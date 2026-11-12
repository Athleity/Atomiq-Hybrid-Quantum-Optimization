"""Independently verify saved nested-CV predictions using training data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomiq.metrics import METRIC_NAMES, stable_top_k_indices
from atomiq.schema import load_training_data


def audit_results(workbook: Path, results_dir: Path) -> dict[str, object]:
    data = load_training_data(workbook)
    id_to_target = dict(zip(data.customer_ids.astype(str), data.target, strict=True))
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(results_dir / "oof_predictions.csv", dtype={"customer_id": str})
    stored_metrics = pd.read_csv(results_dir / "fold_metrics.csv")
    assignments = pd.read_csv(results_dir / "fold_assignments.csv", dtype={"customer_id": str})

    if manifest["status"] != "complete":
        raise AssertionError("Experiment manifest is not complete.")
    if manifest["test_sheet_read"] or manifest["test_labels_or_aggregate_label_counts_used"]:
        raise AssertionError("Manifest indicates forbidden test-label access.")
    if "customer_id" in manifest["feature_columns"] or "duration" in manifest["feature_columns"]:
        raise AssertionError("Forbidden identifier/duration appears in model features.")
    if predictions["customer_id"].isna().any() or not set(predictions.customer_id).issubset(
        set(data.customer_ids.astype(str))
    ):
        raise AssertionError("OOF predictions contain IDs outside Stage1_Train.")
    if not predictions["y_true"].eq(predictions.customer_id.map(id_to_target)).all():
        raise AssertionError("Saved OOF labels do not match Stage1_Train labels by ID.")
    if not np.isfinite(predictions.probability).all() or not predictions.probability.between(
        0.0, 1.0
    ).all():
        raise AssertionError("Saved probabilities are non-finite or outside [0, 1].")

    expected_fold_rows = manifest["outer_folds"] * len(manifest["model_names"])
    if len(stored_metrics) != expected_fold_rows * 2:
        raise AssertionError("Metric file row count differs from model/fold/type design.")
    if len(assignments) != manifest["outer_folds"] * 100:
        raise AssertionError("Fold assignments do not contain 100 IDs per validation fold.")

    maximum_metric_error = 0.0
    independently_checked_metrics = (
        "roc_auc",
        "pr_auc",
        "precision_at_10",
        "recall_at_10",
        "brier_score",
        "log_loss",
        "ece_10_bins",
    )
    checked_folds = 0
    maximum_top10_disagreements = 0
    for (model, repeat, fold), group in predictions.groupby(
        ["model", "repeat", "fold"], sort=True
    ):
        validation_order = assignments.loc[
            assignments.repeat.eq(repeat) & assignments.fold.eq(fold), "customer_id"
        ].tolist()
        expected_ids = set(validation_order)
        if set(group.customer_id) != expected_ids:
            raise AssertionError(f"Prediction coverage differs for {model}, {repeat}, {fold}.")
        by_type = group.pivot(index=["customer_id", "y_true"], columns="prediction_type").loc[
            validation_order
        ]
        ids = by_type.index.get_level_values("customer_id").to_numpy(dtype=str)
        y = by_type.index.get_level_values("y_true").to_numpy(dtype=int)
        raw = by_type["probability"]["raw"].to_numpy(dtype=float)
        calibrated = by_type["probability"]["sigmoid_calibrated"].to_numpy(dtype=float)
        raw_top = stable_top_k_indices(raw, ids, 10)
        calibrated_top = stable_top_k_indices(calibrated, ids, 10)
        maximum_top10_disagreements = max(
            maximum_top10_disagreements,
            len(set(ids[raw_top]) ^ set(ids[calibrated_top])),
        )
        if set(ids[raw_top]) != set(ids[calibrated_top]):
            raise AssertionError(f"Calibration changed top-ten membership for {model}, {repeat}, {fold}.")

        for prediction_type, probability in (
            ("raw", raw),
            ("sigmoid_calibrated", calibrated),
        ):
            top_indices = stable_top_k_indices(probability, ids, 10)
            bin_indices = np.minimum((probability * 10).astype(int), 9)
            ece = sum(
                float(np.mean(bin_indices == bin_index))
                * abs(
                    float(y[bin_indices == bin_index].mean())
                    - float(probability[bin_indices == bin_index].mean())
                )
                for bin_index in range(10)
                if np.any(bin_indices == bin_index)
            )
            actual = {
                "roc_auc": float(roc_auc_score(y, probability)),
                "pr_auc": float(average_precision_score(y, probability)),
                "precision_at_10": float(y[top_indices].mean()),
                "recall_at_10": float(y[top_indices].sum() / y.sum()),
                "brier_score": float(brier_score_loss(y, probability)),
                "log_loss": float(log_loss(y, probability, labels=[0, 1])),
                "ece_10_bins": float(ece),
            }
            row = stored_metrics.loc[
                stored_metrics.model.eq(model)
                & stored_metrics.repeat.eq(repeat)
                & stored_metrics.fold.eq(fold)
                & stored_metrics.prediction_type.eq(prediction_type)
            ]
            if len(row) != 1:
                raise AssertionError("Expected exactly one stored metric row per model/fold/type.")
            for metric in independently_checked_metrics:
                error = abs(float(row.iloc[0][metric]) - actual[metric])
                maximum_metric_error = max(maximum_metric_error, error)
                if error > 1e-10:
                    raise AssertionError(f"Stored {metric} mismatch for {model}, {repeat}, {fold}.")
            for metric in set(METRIC_NAMES) - set(independently_checked_metrics):
                if not np.isfinite(float(row.iloc[0][metric])):
                    raise AssertionError(f"Stored diagnostic {metric} is not finite.")
        checked_folds += 1

    if maximum_top10_disagreements != 0:
        raise AssertionError("Raw and calibrated rankings differ in top-ten membership.")
    summary = pd.read_csv(results_dir / "model_summary.csv")
    expected_summary_rows = len(manifest["model_names"]) * 2
    if len(summary) != expected_summary_rows:
        raise AssertionError("Model summary is missing rows.")
    return {
        "status": "passed",
        "training_rows": len(data.target),
        "training_positive_count": int(data.target.sum()),
        "test_sheet_read_by_audit": False,
        "checked_model_fold_predictions": checked_folds,
        "checked_metric_rows": len(stored_metrics),
        "independently_checked_metric_names": list(independently_checked_metrics),
        "maximum_absolute_metric_recomputation_error": maximum_metric_error,
        "maximum_top10_membership_disagreements": maximum_top10_disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook", type=Path, default=Path("data/raw/bank_campaign_participant_release.xlsx")
    )
    parser.add_argument("--results", type=Path, default=Path("results/stage1_classical"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit_results(args.workbook, args.results)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
