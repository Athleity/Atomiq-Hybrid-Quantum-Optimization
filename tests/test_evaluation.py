import json

import pandas as pd

from atomiq.evaluation import (
    EvaluationConfig,
    _candidate_parameters,
    make_outer_splits,
    run_nested_repeated_cv,
)
from atomiq.calibration import PositiveSigmoidCalibratedClassifier
from atomiq.metrics import stable_top_k_indices
from atomiq.estimators import build_model_portfolio
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from atomiq.metrics import evaluate_probabilities
from atomiq.schema import load_training_data


def test_outer_repeated_folds_share_each_customer_once_per_repeat():
    data = load_training_data("data/raw/bank_campaign_participant_release.xlsx")
    folds = make_outer_splits(data.target, seed=42, n_splits=3, n_repeats=2)
    assert len(folds) == 6
    for repeat in (1, 2):
        validation_indices = [valid for r, _, _, valid in folds if r == repeat]
        assert all(len(indices) == 100 for indices in validation_indices)
        assert sorted(index for indices in validation_indices for index in indices) == list(
            range(300)
        )


def test_compact_search_still_compares_unweighted_and_balanced_candidates():
    candidates = _candidate_parameters(
        {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        count=2,
        seed=17,
    )
    assert {candidate["model__class_weight"] for candidate in candidates} == {
        None,
        "balanced",
    }


def test_monotone_sigmoid_calibration_cannot_reverse_probability_ranking():
    data = load_training_data("data/raw/bank_campaign_participant_release.xlsx")
    estimator = clone(
        next(s.estimator for s in build_model_portfolio(42) if s.name == "random_forest")
    )
    splits = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(
        data.features, data.target
    ))
    calibrated = PositiveSigmoidCalibratedClassifier(estimator).fit(
        data.features, data.target, cv_splits=splits
    )
    raw = estimator.fit(data.features, data.target).predict_proba(data.features)[:, 1]
    mapped = calibrated.predict_proba(data.features)[:, 1]
    assert calibrated.slope_ > 0
    assert stable_top_k_indices(raw, data.customer_ids.to_numpy(), 10).tolist() == (
        stable_top_k_indices(mapped, data.customer_ids.to_numpy(), 10).tolist()
    )


def test_calibration_preserves_logistic_ranking_when_raw_scores_saturate():
    data = load_training_data("data/raw/bank_campaign_participant_release.xlsx")
    estimator = clone(
        next(
            s.estimator
            for s in build_model_portfolio(42)
            if s.name == "logistic_unregularized"
        )
    )
    train_indices = list(range(200))
    validation_indices = list(range(200, 300))
    train_x, train_y = data.features.iloc[train_indices], data.target.iloc[train_indices]
    valid_x, valid_y = data.features.iloc[validation_indices], data.target.iloc[validation_indices]
    inner_splits = list(
        StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(train_x, train_y)
    )
    calibrated = PositiveSigmoidCalibratedClassifier(estimator).fit(
        train_x, train_y, cv_splits=inner_splits
    )
    raw_estimator = clone(estimator).fit(train_x, train_y)
    raw = raw_estimator.predict_proba(valid_x)[:, 1]
    mapped = calibrated.predict_proba(valid_x)[:, 1]
    raw_metrics = evaluate_probabilities(
        valid_y.to_numpy(),
        raw,
        data.customer_ids.iloc[validation_indices].to_numpy(),
    )
    mapped_metrics = evaluate_probabilities(
        valid_y.to_numpy(),
        mapped,
        data.customer_ids.iloc[validation_indices].to_numpy(),
    )
    assert mapped_metrics["roc_auc"] == raw_metrics["roc_auc"]
    assert mapped_metrics["pr_auc"] == raw_metrics["pr_auc"]


def test_small_nested_run_writes_only_training_oof_predictions(tmp_path):
    data = load_training_data("data/raw/bank_campaign_participant_release.xlsx")
    result = run_nested_repeated_cv(
        data,
        EvaluationConfig(
            seed=42,
            outer_splits=3,
            outer_repeats=1,
            inner_splits=2,
            search_iterations=2,
            top_k=10,
        ),
        tmp_path,
        model_names=[
            "logistic_unregularized",
            "logistic_regularized",
            "svm_linear",
        ],
    )

    predictions = pd.read_csv(tmp_path / "oof_predictions.csv")
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(predictions) == 1800
    assert set(predictions["customer_id"]) == set(data.customer_ids)
    assert set(predictions["prediction_type"]) == {"raw", "sigmoid_calibrated"}
    assert manifest["test_sheet_read"] is False
    assert manifest["test_prediction_generated"] is False
    assert result["training_rows"] == 300
    trials = pd.read_csv(tmp_path / "tuning_trials.csv")
    regularized_trials = trials.loc[trials["model"] == "logistic_regularized"]
    weights = {
        json.loads(value)["model__class_weight"]
        for value in regularized_trials["parameters_json"]
    }
    assert weights == {None, "balanced"}
    assert "svm_linear" in set(trials["model"])
    assert (tmp_path / "models/selected_model_calibrated.joblib").exists()
