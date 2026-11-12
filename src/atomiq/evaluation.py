"""Nested repeated cross-validation and artifact writing for classical Stage 1."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterSampler, RepeatedStratifiedKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from atomiq.calibration import PositiveSigmoidCalibratedClassifier, positive_class_scores
from atomiq.estimators import ModelSpec, build_model_portfolio
from atomiq.metrics import METRIC_NAMES, evaluate_probabilities, make_reliability_bins
from atomiq.schema import TrainingData


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 42
    outer_splits: int = 3
    outer_repeats: int = 4
    inner_splits: int = 3
    search_iterations: int = 3
    calibration_method: str = "positive_slope_sigmoid"
    top_k: int = 10
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.outer_splits < 2 or self.outer_repeats < 1 or self.inner_splits < 2:
            raise ValueError("CV split counts must be >=2 and repeat count must be >=1.")
        if self.search_iterations < 2 or self.top_k < 1 or self.n_jobs != 1:
            raise ValueError(
                "Search iterations must be >=2 to compare imbalance treatments; "
                "top_k must be positive and n_jobs must be 1."
            )
        if self.calibration_method != "positive_slope_sigmoid":
            raise ValueError(
                "The predeclared calibration method is positive_slope_sigmoid."
            )


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Read and validate the explicit experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    section = raw.get("phase1_classical", {})
    return EvaluationConfig(**section)


def make_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"atomiq.phase1.{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
        stream_handler = logging.StreamHandler()
        for handler in (file_handler, stream_handler):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def make_outer_splits(
    target: pd.Series, *, seed: int, n_splits: int, n_repeats: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Materialize identical repeated stratified splits for every estimator."""
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    dummy_x = np.zeros((len(target), 1), dtype=np.uint8)
    splits = []
    for split_number, (train_indices, validation_indices) in enumerate(
        splitter.split(dummy_x, target)
    ):
        repeat_number = split_number // n_splits + 1
        fold_number = split_number % n_splits + 1
        splits.append((repeat_number, fold_number, train_indices, validation_indices))
    return splits


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, type):
        return value.__name__
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _training_fingerprint(data: TrainingData) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(data.customer_ids, index=False).values.tobytes())
    digest.update(pd.util.hash_pandas_object(data.features, index=False).values.tobytes())
    digest.update(pd.util.hash_pandas_object(data.target, index=False).values.tobytes())
    return digest.hexdigest()


def _set_fold_seed(estimator: Any, seed: int) -> Any:
    parameters = estimator.get_params(deep=True)
    updates = {}
    if "random_state" in parameters:
        updates["random_state"] = seed
    elif "model__random_state" in parameters:
        updates["model__random_state"] = seed
    if updates:
        estimator.set_params(**updates)
    return estimator


def _candidate_parameters(
    parameter_distributions: dict[str, list[Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Draw a compact search set and guarantee both class-weight regimes are tested."""
    if not parameter_distributions:
        return []
    weight_key = next(
        (
            key
            for key in ("model__class_weight", "model__auto_class_weights", "auto_class_weights")
            if key in parameter_distributions
        ),
        None,
    )
    candidates: list[dict[str, Any]] = []
    if weight_key is not None:
        weights = parameter_distributions[weight_key]
        if count < len(weights):
            raise ValueError(
                f"Search budget {count} is too small to compare all weights for {weight_key}."
            )
        other_parameters = {
            key: values for key, values in parameter_distributions.items() if key != weight_key
        }
        for index, weight in enumerate(weights):
            sampled = next(
                iter(
                    ParameterSampler(
                        other_parameters,
                        n_iter=1,
                        random_state=seed + 997 * (index + 1),
                    )
                )
            )
            sampled[weight_key] = weight
            candidates.append(sampled)
    while len(candidates) < count:
        remaining = count - len(candidates)
        for candidate in ParameterSampler(
            parameter_distributions,
            n_iter=remaining,
            random_state=seed + 7919,
        ):
            if candidate not in candidates:
                candidates.append(candidate)
                if len(candidates) == count:
                    break
        else:
            raise ValueError("Could not draw enough unique hyperparameter configurations.")
    return candidates[:count]


def _grid_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, list[Any]]]:
    """GridSearchCV representation of an explicit, already seeded candidate budget."""
    return [{key: [value] for key, value in candidate.items()} for candidate in candidates]


def _probabilities_and_source(
    estimator: Any, X: pd.DataFrame
) -> tuple[np.ndarray, str]:
    return positive_class_scores(estimator, X)


def _positive_probabilities(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    return _probabilities_and_source(estimator, X)[0]


def _trial_rows(
    search: GridSearchCV,
    *,
    model_name: str,
    repeat: int,
    fold: int,
    seed: int,
) -> list[dict[str, Any]]:
    results = search.cv_results_
    rows = []
    for index, parameters in enumerate(results["params"]):
        rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
                "search_seed": seed,
                "candidate_index": index,
                "parameters_json": json.dumps(parameters, default=_json_default, sort_keys=True),
                "inner_pr_auc_mean": float(results["mean_test_score"][index]),
                "inner_pr_auc_std": float(results["std_test_score"][index]),
                "rank": int(results["rank_test_score"][index]),
                "selected": index == int(search.best_index_),
            }
        )
    return rows


def _write_partial_artifacts(
    output_dir: Path,
    fold_metrics: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> None:
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(trials).to_csv(output_dir / "tuning_trials.csv", index=False)
    pd.DataFrame(assignments).drop_duplicates().to_csv(
        output_dir / "fold_assignments.csv", index=False
    )


def _make_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, prediction_type), group in fold_metrics.groupby(
        ["model", "prediction_type"], sort=True
    ):
        row: dict[str, Any] = {
            "model": model,
            "prediction_type": prediction_type,
            "n_outer_folds": int(len(group)),
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_calibration_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return paired per-fold calibrated-minus-raw differences for every metric."""
    rows: list[dict[str, Any]] = []
    for model, group in fold_metrics.groupby("model", sort=True):
        paired = group.pivot(index=["repeat", "fold"], columns="prediction_type")
        for metric in METRIC_NAMES:
            differences = (
                paired[metric]["sigmoid_calibrated"] - paired[metric]["raw"]
            ).to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_paired_outer_folds": int(len(differences)),
                    "calibrated_minus_raw_mean": float(np.mean(differences)),
                    "calibrated_minus_raw_std": float(np.std(differences, ddof=1)),
                    "folds_improved": int(np.sum(differences > 0))
                    if metric in {"roc_auc", "pr_auc", "precision_at_10", "recall_at_10"}
                    else int(np.sum(differences < 0)),
                }
            )
    return pd.DataFrame(rows)


def _fit_final_candidate(
    spec: ModelSpec,
    data: TrainingData,
    config: EvaluationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Tune and calibrate the selected classical candidate on training rows only."""
    search_seed = config.seed + 900_001
    estimator = _set_fold_seed(clone(spec.estimator), search_seed)
    inner = StratifiedKFold(n_splits=config.inner_splits, shuffle=True, random_state=search_seed)
    inner_splits = list(inner.split(data.features, data.target))
    search: GridSearchCV | None = None
    if spec.parameter_distributions:
        candidates = _candidate_parameters(
            spec.parameter_distributions,
            count=config.search_iterations,
            seed=search_seed,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=_grid_from_candidates(candidates),
            scoring="average_precision",
            cv=inner_splits,
            n_jobs=config.n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(data.features, data.target)
        selected_estimator = search.best_estimator_
        best_params = search.best_params_
        full_data_trials = _trial_rows(
            search,
            model_name=spec.name,
            repeat=0,
            fold=0,
            seed=search_seed,
        )
        pd.DataFrame(full_data_trials).to_csv(output_dir / "final_fit_tuning_trials.csv", index=False)
    else:
        selected_estimator = estimator.fit(data.features, data.target)
        best_params = {}

    calibration_splitter = StratifiedKFold(
        n_splits=config.inner_splits, shuffle=True, random_state=search_seed + 1
    )
    calibration_splits = list(calibration_splitter.split(data.features, data.target))
    final_calibrator = PositiveSigmoidCalibratedClassifier(selected_estimator).fit(
        data.features, data.target, cv_splits=calibration_splits
    )
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "selected_model_calibrated.joblib"
    joblib.dump(final_calibrator, model_path, compress=3)
    return {
        "model": spec.name,
        "calibration": config.calibration_method,
        "best_params": best_params,
        "model_path": str(model_path.as_posix()),
        "training_rows": len(data.target),
        "training_positive_count": int(data.target.sum()),
        "train_test_prediction_generated": False,
    }


def run_nested_repeated_cv(
    data: TrainingData,
    config: EvaluationConfig,
    output_dir: str | Path,
    *,
    model_names: Iterable[str] | None = None,
    model_specs: Sequence[ModelSpec] | None = None,
) -> dict[str, Any]:
    """Run nested tuning and sigmoid calibration with shared repeated outer folds."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    logger = make_logger(destination)
    if model_names is not None and model_specs is not None:
        raise ValueError("Provide model_names or model_specs, not both.")
    specs = list(model_specs) if model_specs is not None else build_model_portfolio(config.seed)
    if model_names is not None:
        requested = set(model_names)
        specs = [spec for spec in specs if spec.name in requested]
        missing = requested - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"Unknown model names: {sorted(missing)}")
    if not specs:
        raise ValueError("At least one model is required.")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Model names must be unique within one experiment run.")

    config_path = destination / "run_config.yaml"
    config_dump = {"phase1_classical": asdict(config), "models": [s.name for s in specs]}
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config_dump, stream, sort_keys=False)

    folds = make_outer_splits(
        data.target,
        seed=config.seed,
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
    )
    fold_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    run_id = datetime.now(timezone.utc).strftime("phase1-%Y%m%dT%H%M%SZ")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "all_installed_package_versions": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        "training_fingerprint_sha256": _training_fingerprint(data),
        "training_rows": int(len(data.target)),
        "training_positive_count": int(data.target.sum()),
        "training_negative_count": int((data.target == 0).sum()),
        "feature_columns": list(data.features.columns),
        "identifier_used_as_predictor": False,
        "test_sheet_read": False,
        "test_labels_or_aggregate_label_counts_used": False,
        "config": asdict(config),
        "model_names": [spec.name for spec in specs],
        "outer_folds": len(folds),
        "outer_fold_validation_rows": [int(len(valid)) for _, _, _, valid in folds],
        "calibration_method": config.calibration_method,
        "search_scoring": "average_precision",
        "runtime_limitations": "CPU only; n_jobs=1; no hardware backend used",
    }
    manifest_path = destination / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, default=_json_default)

    try:
        with threadpool_limits(limits=1):
            for repeat, fold, train_indices, validation_indices in folds:
                X_train = data.features.iloc[train_indices]
                y_train = data.target.iloc[train_indices]
                X_valid = data.features.iloc[validation_indices]
                y_valid = data.target.iloc[validation_indices].to_numpy(dtype=int)
                ids_valid = data.customer_ids.iloc[validation_indices].to_numpy(dtype=str)
                fold_seed = config.seed + repeat * 10_000 + fold * 101
                inner_splitter = StratifiedKFold(
                    n_splits=config.inner_splits,
                    shuffle=True,
                    random_state=fold_seed,
                )
                inner_splits = list(inner_splitter.split(X_train, y_train))

                for model_index, spec in enumerate(specs):
                    search_seed = fold_seed + model_index
                    estimator = _set_fold_seed(clone(spec.estimator), search_seed)
                    logger.info(
                        "fit model=%s repeat=%d fold=%d train=%d validation=%d seed=%d",
                        spec.name,
                        repeat,
                        fold,
                        len(train_indices),
                        len(validation_indices),
                        search_seed,
                    )
                    search: GridSearchCV | None = None
                    if spec.parameter_distributions:
                        candidates = _candidate_parameters(
                            spec.parameter_distributions,
                            count=config.search_iterations,
                            seed=search_seed,
                        )
                        search = GridSearchCV(
                            estimator=estimator,
                            param_grid=_grid_from_candidates(candidates),
                            scoring="average_precision",
                            cv=inner_splits,
                            n_jobs=config.n_jobs,
                            refit=True,
                            return_train_score=False,
                            error_score="raise",
                        )
                        search.fit(X_train, y_train)
                        raw_estimator = search.best_estimator_
                        best_params = search.best_params_
                        trials.extend(
                            _trial_rows(
                                search,
                                model_name=spec.name,
                                repeat=repeat,
                                fold=fold,
                                seed=search_seed,
                            )
                        )
                    else:
                        raw_estimator = estimator.fit(X_train, y_train)
                        best_params = {}

                    raw_probabilities, raw_score_method = _probabilities_and_source(
                        raw_estimator, X_valid
                    )
                    calibrator = PositiveSigmoidCalibratedClassifier(
                        clone(raw_estimator)
                    ).fit(
                        X_train,
                        y_train,
                        cv_splits=inner_splits,
                    )
                    calibrated_probabilities = _positive_probabilities(calibrator, X_valid)

                    for prediction_type, values in (
                        ("raw", raw_probabilities),
                        ("sigmoid_calibrated", calibrated_probabilities),
                    ):
                        metrics = evaluate_probabilities(
                            y_valid,
                            values,
                            ids_valid,
                            top_k=config.top_k,
                        )
                        fold_metrics.append(
                            {
                                "run_id": run_id,
                                "model": spec.name,
                                "prediction_type": prediction_type,
                                "repeat": repeat,
                                "fold": fold,
                                "search_seed": search_seed,
                                "raw_score_method": (
                                    raw_score_method
                                    if prediction_type == "raw"
                                    else "cross_fitted_sigmoid_calibration"
                                ),
                                "best_params_json": json.dumps(
                                    best_params, default=_json_default, sort_keys=True
                                ),
                                **metrics,
                            }
                        )
                        for customer_id, truth, probability in zip(
                            ids_valid, y_valid, values, strict=True
                        ):
                            predictions.append(
                                {
                                    "run_id": run_id,
                                    "model": spec.name,
                                    "prediction_type": prediction_type,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "customer_id": customer_id,
                                    "y_true": int(truth),
                                    "probability": float(probability),
                                    "raw_score_method": (
                                        raw_score_method
                                        if prediction_type == "raw"
                                        else "cross_fitted_sigmoid_calibration"
                                    ),
                                }
                            )
                    assignments.extend(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "customer_id": customer_id,
                            "role": "validation",
                        }
                        for customer_id in ids_valid
                    )
                    _write_partial_artifacts(
                        destination, fold_metrics, predictions, trials, assignments
                    )

        metrics_frame = pd.DataFrame(fold_metrics)
        summary = _make_summary(metrics_frame)
        summary.to_csv(destination / "model_summary.csv", index=False)
        _make_calibration_comparison(metrics_frame).to_csv(
            destination / "calibration_comparison.csv", index=False
        )
        prediction_frame = pd.DataFrame(predictions)
        reliability_rows = []
        for (model_name, prediction_type), group in prediction_frame.groupby(
            ["model", "prediction_type"], sort=True
        ):
            rows = make_reliability_bins(
                group["y_true"].to_numpy(), group["probability"].to_numpy()
            )
            for row in rows:
                reliability_rows.append(
                    {"model": model_name, "prediction_type": prediction_type, **row}
                )
        pd.DataFrame(reliability_rows).to_csv(
            destination / "calibration_bins.csv", index=False
        )

        calibrated = summary.loc[summary["prediction_type"] == "sigmoid_calibrated"].copy()
        champion = calibrated.sort_values(
            ["pr_auc_mean", "brier_score_mean", "log_loss_mean"],
            ascending=[False, True, True],
        ).iloc[0]
        chosen_spec = next(spec for spec in specs if spec.name == champion["model"])
        champion_record = _fit_final_candidate(chosen_spec, data, config, destination)
        champion_record["selection_rule"] = (
            "highest nested outer-fold mean PR-AUC; ties by lower Brier then log loss"
        )
        champion_record["outer_cv_calibrated_metrics"] = {
            metric: float(champion[f"{metric}_mean"])
            for metric in METRIC_NAMES
        }
        with (destination / "selected_model.json").open("w", encoding="utf-8") as stream:
            json.dump(champion_record, stream, indent=2, default=_json_default)

        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": [
                    "fold_metrics.csv",
                    "oof_predictions.csv",
                    "tuning_trials.csv",
                    "fold_assignments.csv",
                    "model_summary.csv",
                    "calibration_comparison.csv",
                    "calibration_bins.csv",
                    "selected_model.json",
                    "models/selected_model_calibrated.joblib",
                ],
                "test_prediction_generated": False,
            }
        )
        logger.info("completed run_id=%s champion=%s", run_id, champion_record["model"])
        return champion_record
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        logger.exception("experiment failed")
        raise
    finally:
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, default=_json_default)
