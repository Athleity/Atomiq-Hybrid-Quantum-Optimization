"""Independently re-check frozen inference metadata and final Stage 2/3 outputs."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from atomiq.stage2 import (
    enumerate_optimum,
    evaluate_selection,
    load_stage2_source,
    problem_from_candidates,
    solve_scipy_milp,
)
from atomiq.stage2_qiskit_circuits import build_parameterized_qaoa, exact_distribution
from atomiq.stage3_quantum import (
    poisson_binomial_distribution,
    probability_at_least,
    stage3_statevector_distribution,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enumerated_distribution(probabilities: np.ndarray) -> np.ndarray:
    histogram = np.zeros(len(probabilities) + 1)
    for outcome in itertools.product((0, 1), repeat=len(probabilities)):
        mass = float(np.prod([p if bit else 1 - p for p, bit in zip(probabilities, outcome, strict=True)]))
        histogram[sum(outcome)] += mass
    return histogram


def distribution_from_counts(counts: dict[str, int], n: int) -> np.ndarray:
    result = np.zeros(1 << n, dtype=float)
    total = sum(counts.values())
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if len(bits) != n:
            raise AssertionError(f"Unexpected Stage 2 count key {bitstring!r}.")
        result[int(bits, 2)] += count / total
    return result


def independent_stage2_metrics(problem, distribution, optimum):
    feasible = np.zeros(len(distribution), dtype=bool)
    rewards = np.zeros(len(distribution), dtype=float)
    for mask in range(len(distribution)):
        ids = [c.customer_id for q, c in enumerate(problem.candidates) if (mask >> q) & 1]
        checked = evaluate_selection(problem, ids)
        feasible[mask] = checked.feasible
        rewards[mask] = checked.objective
    p_feasible = float(distribution[feasible].sum())
    conditional = float(distribution[feasible] @ rewards[feasible] / p_feasible) if p_feasible else None
    optimal = feasible & (np.abs(rewards - optimum.objective) <= 1e-10)
    support = feasible & (distribution > 0)
    return {
        "feasible_shot_probability": p_feasible,
        "expected_objective_unconditional_with_zero_for_infeasible": float(distribution[feasible] @ rewards[feasible]),
        "expected_objective_conditional_on_feasibility": conditional,
        "exact_optimum_sampling_probability": float(distribution[optimal].sum()),
        "best_feasible_measured_objective": float(rewards[support].max()) if support.any() else None,
    }


def wilson_interval(successes: int, shots: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / shots
    denominator = 1 + z * z / shots
    center = (p + z * z / (2 * shots)) / denominator
    half = z * np.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denominator
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def main() -> None:
    freeze_path = ROOT / "results/stage1_classical/frozen_model.json"
    inference_path = ROOT / "results/stage1_classical/inference_manifest.json"
    predictions_path = ROOT / "results/stage1_classical/stage1_test_predictions.csv"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    if freeze["status"] != "frozen_before_stage1_test_inference":
        raise AssertionError("Stage 1 was not frozen before inference.")
    model_path = ROOT / freeze["model_artifact"]
    if sha256(model_path) != freeze["model_sha256"]:
        raise AssertionError("Frozen model artifact SHA-256 mismatch.")
    if sha256(predictions_path) != inference["predictions_sha256"]:
        raise AssertionError("Stage 1 inference file SHA-256 mismatch.")
    if inference["test_labels_requested_or_loaded"] or inference["test_data_used_for_selection_or_tuning"]:
        raise AssertionError("Inference manifest claims prohibited test-label usage.")
    predictions = pd.read_csv(predictions_path, dtype={"customer_id": str})
    if len(predictions) != 100 or predictions.customer_id.duplicated().any():
        raise AssertionError("Expected 100 unique Stage1_Test prediction rows.")
    ranked = predictions.sort_values(["probability", "customer_id"], ascending=[False, True], kind="mergesort")
    if ranked.customer_id.tolist() != predictions.customer_id.tolist():
        raise AssertionError("Predictions are not in deterministic descending-probability order.")
    if predictions.customer_id.head(10).tolist() != inference["top_ten_customer_ids"]:
        raise AssertionError("Top-ten IDs disagree with inference manifest.")

    operational, constraints = load_stage2_source(ROOT / "data/raw/bank_campaign_participant_release.xlsx")
    top_ten = predictions.head(10)
    problem = problem_from_candidates(top_ten, operational, constraints)
    optimum, feasible = enumerate_optimum(problem)
    milp = solve_scipy_milp(problem)
    stage2_path = ROOT / "results/stage2_final/actual_stage2_result.json"
    stage2_record = json.loads(stage2_path.read_text(encoding="utf-8"))
    saved_ids = tuple(stage2_record["exact_enumeration_optimum"]["customer_ids"])
    if saved_ids != optimum.customer_ids or abs(stage2_record["exact_enumeration_optimum"]["objective"] - optimum.objective) > 1e-12:
        raise AssertionError("Persisted Stage 2 result does not match fresh exhaustive recomputation.")
    if abs(milp.objective - optimum.objective) > 1e-10:
        raise AssertionError("Independent MILP disagrees with enumeration.")
    if len(feasible) != stage2_record["feasible_exact_six_subsets"]:
        raise AssertionError("Persisted Stage 2 feasible subset count does not match recomputation.")

    qaoa_record = json.loads((ROOT / "results/stage2_final/actual_stage2_qaoa_results.json").read_text(encoding="utf-8"))
    ansatz = build_parameterized_qaoa(problem, depth=1, kind="xy", topology="cycle")
    qaoa_errors = []
    for experiment in qaoa_record["results"]:
        qaoa_distribution = exact_distribution(ansatz, experiment["parameters_gamma_then_beta"])
        independently_scored = independent_stage2_metrics(problem, qaoa_distribution, optimum)
        for name, value in independently_scored.items():
            stored = experiment["ideal_statevector_metrics"][name]
            qaoa_errors.append(abs(value - stored))
        for condition in ("ideal_finite_shot", "noisy_finite_shot"):
            sampled_distribution = distribution_from_counts(experiment[condition]["counts"], 10)
            recomputed = independent_stage2_metrics(problem, sampled_distribution, optimum)
            for name, value in recomputed.items():
                stored = experiment[condition]["metrics"][name]
                qaoa_errors.append(abs(value - stored))
    qaoa_max_error = float(max(qaoa_errors))
    if qaoa_max_error > 1e-10:
        raise AssertionError(f"Independent QAOA metric recomputation mismatch: {qaoa_max_error}")

    selected = pd.read_csv(ROOT / "results/stage2_final/actual_selected_six.csv", dtype={"customer_id": str})
    probabilities = selected.stage1_probability.to_numpy(dtype=float)
    dp = poisson_binomial_distribution(probabilities)
    enumeration = enumerated_distribution(probabilities)
    stage3_path = ROOT / "results/stage3_final/actual_stage3_results.json"
    stage3_record = json.loads(stage3_path.read_text(encoding="utf-8"))
    qiskit_distribution = np.asarray(stage3_record["statevector_all_64_probabilities"], dtype=float)
    qiskit_histogram = np.zeros(7)
    for state, probability in enumerate(qiskit_distribution):
        qiskit_histogram[state.bit_count()] += probability
    exact_event = float(dp[3:].sum())
    checks = {
        "dp_vs_64_outcome_enumeration_max_abs_error": float(np.max(np.abs(dp - enumeration))),
        "dp_vs_persisted_stage3_histogram_max_abs_error": float(
            np.max(np.abs(dp - np.asarray(stage3_record["exact_poisson_binomial_dp_k0_to6"])))
        ),
        "statevector_vs_exact_weight_histogram_max_abs_error": float(np.max(np.abs(dp - qiskit_histogram))),
        "stage3_p_ge_3_vs_exact_abs_error": float(abs(exact_event - stage3_record["exact_probability_at_least_three"])),
        "selected_six_match_stage2_optimum": sorted(selected.customer_id.tolist()) == list(optimum.customer_ids),
    }
    if max(value for key, value in checks.items() if isinstance(value, float)) > 2e-12:
        raise AssertionError(f"Independent Stage 3 audit error exceeded tolerance: {checks}")
    if not checks["selected_six_match_stage2_optimum"]:
        raise AssertionError("Stage 3 customers do not match the Stage 2 optimum.")
    shot_checks = 0
    for experiment in stage3_record["experiments"]:
        for condition in ("ideal_finite_shot", "noisy_finite_shot"):
            entry = experiment[condition]
            recomputed = entry["event_successes"] / entry["shots"]
            if abs(recomputed - entry["estimate"]) > 1e-14:
                raise AssertionError("Stage 3 finite-shot event rate does not match event counts.")
            if not np.allclose(wilson_interval(entry["event_successes"], entry["shots"]), entry["wilson_95_interval"], atol=1e-14):
                raise AssertionError("Stage 3 Wilson interval does not match stored event counts.")
            shot_checks += 1

    result = {
        "status": "passed_independent_actual_result_audit",
        "stage1_prediction_rows": len(predictions),
        "stage1_predictions_sha256": inference["predictions_sha256"],
        "stage1_test_labels_requested_or_used": False,
        "stage2_subsets_exhaustively_checked": 210,
        "stage2_feasible_exact_six_count": len(feasible),
        "stage2_milp_objective_difference": abs(milp.objective - optimum.objective),
        "stage2_selected_customer_ids": list(optimum.customer_ids),
        "stage2_objective": optimum.objective,
        "stage2_qaoa_variants_independently_recomputed": len(qaoa_record["results"]),
        "stage2_qaoa_distribution_metric_max_abs_error": qaoa_max_error,
        "stage3_exact_probability_at_least_three": exact_event,
        "stage3_finite_shot_rate_and_wilson_checks": shot_checks,
        "checks": checks,
    }
    output = ROOT / "results/final_actual_independent_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
