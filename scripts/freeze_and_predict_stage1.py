"""Freeze the selected training-only Stage 1 model and score test features once."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from atomiq.schema import load_stage1_inference_data, load_training_data


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "raw" / "bank_campaign_participant_release.xlsx"
RESULTS = ROOT / "results" / "stage1_classical"
MODEL = RESULTS / "models" / "selected_model_calibrated.joblib"
SELECTION = RESULTS / "selected_model.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))
    training = load_training_data(WORKBOOK)
    # Freeze provenance from the labeled training partition only.
    training_bytes = pd.concat(
        [training.customer_ids, training.features, training.target], axis=1
    ).to_csv(index=False, lineterminator="\n").encode("utf-8")
    training_digest = hashlib.sha256(training_bytes).hexdigest()
    model_digest = sha256(MODEL)
    model = joblib.load(MODEL)
    if len(training.target) != selected["training_rows"]:
        raise ValueError("Frozen model metadata and training row count disagree.")
    if int(training.target.sum()) != selected["training_positive_count"]:
        raise ValueError("Frozen model metadata and training positive count disagree.")

    freeze = {
        "status": "frozen_before_stage1_test_inference",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_artifact": MODEL.relative_to(ROOT).as_posix(),
        "model_sha256": model_digest,
        "training_partition_sha256": training_digest,
        "training_rows": len(training.target),
        "training_positive_count": int(training.target.sum()),
        "training_data_source": "Stage1_Train only",
        "selection_metadata": SELECTION.relative_to(ROOT).as_posix(),
        "selection_rule": selected["selection_rule"],
        "selected_model": selected["model"],
        "calibration": selected["calibration"],
        "best_params": selected["best_params"],
        "outer_cv_calibrated_metrics": selected["outer_cv_calibrated_metrics"],
        "prediction_boundary": "freeze metadata written before opening Stage1_Test",
        "test_label_access": False,
    }
    freeze_path = RESULTS / "frozen_model.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

    # This loader's usecols contains customer_id plus the 15 predictors only.
    customer_ids, features = load_stage1_inference_data(WORKBOOK)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(features), 2) or len(positions) != 1:
        raise ValueError("Frozen classifier returned malformed binary probabilities.")
    probabilities = probabilities[:, int(positions[0])]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Frozen classifier returned probabilities outside [0, 1].")

    scored = pd.DataFrame({"customer_id": customer_ids, "probability": probabilities})
    # Stable deterministic tie rule: descending probability, then ascending ID.
    scored = scored.sort_values(
        ["probability", "customer_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1, dtype=int))
    scored["selected_top_10"] = scored["rank"] <= 10
    predictions_path = RESULTS / "stage1_test_predictions.csv"
    scored.to_csv(predictions_path, index=False, float_format="%.17g")

    inference_manifest = {
        "status": "completed_inference_after_frozen_model",
        "model_sha256": model_digest,
        "training_partition_sha256": training_digest,
        "stage1_test_rows_scored": int(len(scored)),
        "stage1_test_columns_requested": ["customer_id", *features.columns.tolist()],
        "test_labels_requested_or_loaded": False,
        "test_data_used_for_selection_or_tuning": False,
        "predictions_file": predictions_path.relative_to(ROOT).as_posix(),
        "predictions_sha256": sha256(predictions_path),
        "top_ten_customer_ids": scored.head(10).customer_id.tolist(),
        "probability_range": [float(scored.probability.min()), float(scored.probability.max())],
    }
    (RESULTS / "inference_manifest.json").write_text(
        json.dumps(inference_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "freeze_record": str(freeze_path.relative_to(ROOT)),
        "inference_manifest": "results/stage1_classical/inference_manifest.json",
        "predictions": str(predictions_path.relative_to(ROOT)),
        "rows_scored": len(scored),
        "top_ten": scored.head(10).to_dict(orient="records"),
        "test_labels_requested_or_loaded": False,
        "model_sha256": model_digest,
    }, indent=2))


if __name__ == "__main__":
    main()
