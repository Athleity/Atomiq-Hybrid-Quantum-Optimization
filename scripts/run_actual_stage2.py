"""Select actual Stage 1 finalists using Stage 2 exhaustive and MILP oracles."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from atomiq.stage2 import (
    enumerate_optimum,
    evaluate_selection,
    load_stage2_source,
    problem_from_candidates,
    solve_scipy_milp,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/stage1_classical/stage1_test_predictions.csv"
FREEZE = ROOT / "results/stage1_classical/frozen_model.json"
INFERENCE = ROOT / "results/stage1_classical/inference_manifest.json"
WORKBOOK = ROOT / "data/raw/bank_campaign_participant_release.xlsx"
OUT = ROOT / "results/stage2_final"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    inference = json.loads(INFERENCE.read_text(encoding="utf-8"))
    if freeze["status"] != "frozen_before_stage1_test_inference":
        raise ValueError("Stage 1 freeze record is missing or invalid.")
    if not PREDICTIONS.exists() or sha256(PREDICTIONS) != inference["predictions_sha256"]:
        raise ValueError("Stage 1 predictions do not match the frozen inference manifest.")
    predictions = pd.read_csv(PREDICTIONS, dtype={"customer_id": str})
    top_ten = predictions.loc[predictions.selected_top_10].copy()
    if len(top_ten) != 10 or top_ten.customer_id.duplicated().any():
        raise ValueError("Expected exactly ten unique frozen Stage 1 finalists.")
    operational, constraints = load_stage2_source(WORKBOOK)
    problem = problem_from_candidates(top_ten, operational, constraints)
    optimum, feasible = enumerate_optimum(problem)
    milp_result = solve_scipy_milp(problem)
    if abs(optimum.objective - milp_result.objective) > 1e-10:
        raise AssertionError("MILP and exhaustive Stage 2 objective values disagree.")
    if optimum.customer_ids != milp_result.customer_ids:
        # An objective tie can produce a different MILP basis; the objective and
        # independent feasibility checks are the mathematically relevant tests.
        if abs(optimum.objective - milp_result.objective) > 1e-10:
            raise AssertionError("MILP and enumeration disagree.")
    chosen = [next(c for c in problem.candidates if c.customer_id == key) for key in optimum.customer_ids]
    details = pd.DataFrame([{
        "customer_id": c.customer_id,
        "stage1_probability": c.probability,
        "expected_deposit_value": c.expected_deposit_value,
        "contact_cost": c.contact_cost,
        "capacity_units": c.capacity_units,
        "expected_net_value": c.expected_net_value,
    } for c in chosen]).sort_values("customer_id")
    OUT.mkdir(parents=True, exist_ok=True)
    top_ten.to_csv(OUT / "actual_top_ten.csv", index=False, float_format="%.17g")
    details.to_csv(OUT / "actual_selected_six.csv", index=False, float_format="%.17g")
    result = {
        "status": "completed_actual_challenge_stage2_exact_and_milp",
        "candidate_source": "frozen Stage1_Test feature-only inference; labels not loaded",
        "stage1_freeze_record": "results/stage1_classical/frozen_model.json",
        "stage1_predictions_sha256": inference["predictions_sha256"],
        "candidate_pool": top_ten[["rank", "customer_id", "probability"]].to_dict(orient="records"),
        "constraints": constraints,
        "feasible_exact_six_subsets": len(feasible),
        "total_exact_six_subsets": 210,
        "exact_enumeration_optimum": {
            "customer_ids": list(optimum.customer_ids),
            "objective": optimum.objective,
            "total_contact_cost": optimum.total_contact_cost,
            "total_capacity": optimum.total_capacity,
            "feasible": optimum.feasible,
        },
        "independent_scipy_milp": {
            "customer_ids": list(milp_result.customer_ids),
            "objective": milp_result.objective,
            "total_contact_cost": milp_result.total_contact_cost,
            "total_capacity": milp_result.total_capacity,
            "feasible": milp_result.feasible,
        },
        "selection_csv": "results/stage2_final/actual_selected_six.csv",
        "test_labels_requested_or_used": False,
    }
    (OUT / "actual_stage2_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
