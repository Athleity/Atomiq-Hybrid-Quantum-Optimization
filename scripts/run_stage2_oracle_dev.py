"""Run the exact Stage 2 oracle on a training-only development candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomiq.stage2 import (
    enumerate_optimum,
    load_stage2_source,
    problem_from_candidates,
    solve_scipy_milp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path("data/raw/bank_campaign_participant_release.xlsx"))
    parser.add_argument("--stage1-results", type=Path, default=Path("results/stage1_classical"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_development"))
    args = parser.parse_args()

    prediction_path = args.stage1_results / "oof_predictions.csv"
    # Deliberately do not load or copy y_true, any test data, or test labels.
    predictions = pd.read_csv(
        prediction_path,
        usecols=["customer_id", "prediction_type", "repeat", "probability"],
        dtype={"customer_id": str},
    )
    predictions = predictions.loc[
        predictions.prediction_type.eq("sigmoid_calibrated"),
        ["customer_id", "repeat", "probability"],
    ]
    probability_by_id = (
        predictions.groupby("customer_id", as_index=False, sort=False)
        .agg(probability=("probability", "mean"), oof_repeat_count=("repeat", "nunique"))
    )
    if len(probability_by_id) != 300 or not probability_by_id.oof_repeat_count.eq(3).all():
        raise ValueError("Development fixture expects three calibrated OOF probabilities for each of 300 training customers.")
    probability_by_id = probability_by_id.sort_values(
        ["probability", "customer_id"], ascending=[False, True], kind="stable"
    ).head(10)
    candidates = probability_by_id.loc[:, ["customer_id", "probability"]].copy()

    operational, constraints = load_stage2_source(args.workbook)
    problem = problem_from_candidates(candidates, operational, constraints)
    optimum, feasible = enumerate_optimum(problem)
    milp_result = solve_scipy_milp(problem)
    if abs(milp_result.objective - optimum.objective) > 1e-8:
        raise AssertionError("SciPy MILP and exhaustive enumeration disagree on the optimum.")
    args.output.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    for candidate in problem.candidates:
        candidate_rows.append({
            "customer_id": candidate.customer_id,
            "probability": candidate.probability,
            "expected_deposit_value": candidate.expected_deposit_value,
            "contact_cost": candidate.contact_cost,
            "capacity_units": candidate.capacity_units,
            "expected_net_value_if_selected": candidate.expected_net_value,
        })
    pd.DataFrame(candidate_rows).sort_values(
        ["probability", "customer_id"], ascending=[False, True], kind="stable"
    ).to_csv(args.output / "development_candidate_pool.csv", index=False)

    selection = [row for row in candidate_rows if row["customer_id"] in optimum.customer_ids]
    pd.DataFrame(selection).sort_values("customer_id").to_csv(
        args.output / "development_exact_selection.csv", index=False
    )
    result = {
        "status": "complete_development_only",
        "candidate_source": "top ten mean calibrated out-of-fold probabilities from Stage1_Train only",
        "stage1_frozen": False,
        "stage1_test_sheet_read": False,
        "stage1_test_labels_or_predictions_read": False,
        "selection_exactly_six": optimum.selected_count == 6,
        "selected_customer_ids": list(optimum.customer_ids),
        "total_contact_cost": optimum.total_contact_cost,
        "budget_limit": problem.budget_limit,
        "total_capacity": optimum.total_capacity,
        "capacity_limit": problem.capacity_limit,
        "objective": optimum.objective,
        "feasible_exact_six_subsets": len(feasible),
        "all_fixed_cardinality_subsets_enumerated": 210,
        "independent_scipy_milp_objective": milp_result.objective,
        "milp_matches_exhaustive_objective": abs(milp_result.objective - optimum.objective) <= 1e-8,
        "objective_definition": "sum(probability * expected_deposit_value - contact_cost)",
        "constraint_definition": "exactly six; total contact_cost <= budget; total capacity_units <= capacity limit",
        "note": "Development fixture only; do not report these candidates or selection as the final challenge result.",
    }
    (args.output / "exact_oracle_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
