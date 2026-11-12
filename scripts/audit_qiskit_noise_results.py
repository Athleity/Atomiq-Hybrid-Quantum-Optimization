"""Independently recompute raw Aer metrics and mitigation vectors from saved counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from atomiq.qiskit_noise import mitigate_readout
from atomiq.stage2 import (
    Candidate,
    Stage2Problem,
    enumerate_optimum,
    evaluate_selection,
    load_stage2_source,
    problem_from_candidates,
)
from atomiq.stage2_development_fixtures import four_item_binding_problem
from atomiq.stage3_quantum import poisson_binomial_distribution, probability_at_least


def independent_stage2_metrics(counts: dict[str, int], problem: Stage2Problem, optimum):
    shots = sum(counts.values())
    feasible_shots = optimal_shots = 0
    objective_sum = 0.0
    best = None
    for bitstring, count in counts.items():
        mask = int(bitstring.replace(" ", ""), 2)
        if mask >= (1 << len(problem.candidates)):
            raise ValueError(f"Bitstring width exceeds candidate count: {bitstring!r}")
        ids = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
        checked = evaluate_selection(problem, ids)
        if checked.feasible:
            feasible_shots += count
            objective_sum += count * checked.objective
            optimal_shots += count * (abs(checked.objective - optimum.objective) <= 1e-10)
            best = checked.objective if best is None else max(best, checked.objective)
    feasible_probability = feasible_shots / shots
    expected_feasible_objective = objective_sum / shots
    return {
        "shots": shots,
        "feasible_probability": feasible_probability,
        "expected_feasible_objective": expected_feasible_objective,
        "conditional_feasible_objective": objective_sum / feasible_shots if feasible_shots else None,
        "optimal_probability": optimal_shots / shots,
        "best_feasible_measured_objective": best,
    }


def assert_close(label: str, observed: float, expected: float, tol: float = 1e-10) -> None:
    if abs(observed - expected) > tol:
        raise AssertionError(f"{label} differs: {observed} vs {expected} (tol={tol})")


def audit_four_item() -> dict[str, object]:
    path = Path("results/stage2_qiskit_noise/qiskit_stage2_noise_results.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problem = four_item_binding_problem()
    optimum, _ = enumerate_optimum(problem)
    matrices = [np.asarray(m, dtype=float) for m in manifest["readout_calibration"]["estimated_matrices_observed_given_true"]]
    checks = []
    for run in manifest["runs"]:
        evaluation = run["evaluation"]
        counts = evaluation["raw_noisy_counts"]
        raw = independent_stage2_metrics(counts, problem, optimum)
        stored = evaluation["noisy_finite_shot_metrics"]
        assert_close("4-item raw feasibility", raw["feasible_probability"], stored["feasible_probability_estimate"])
        assert_close("4-item raw optimal probability", raw["optimal_probability"], stored["exact_optimum_sampling_probability"])
        corrected = mitigate_readout(counts, matrices)
        projection = np.asarray(corrected["projected_distribution"])
        projected_feas = 0.0
        for mask, probability in enumerate(projection):
            ids = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
            if evaluate_selection(problem, ids).feasible:
                projected_feas += probability
        assert_close("4-item mitigated feasibility", projected_feas, evaluation["noisy_readout_mitigated_projected_metrics"]["feasible_probability_estimate"])
        evaluation["noisy_readout_mitigated_quasi_distribution"] = np.asarray(corrected["quasi_distribution"]).tolist()
        evaluation["noisy_readout_mitigated_projected_distribution"] = projection.tolist()
        checks.append({
            "ansatz": run["ansatz"], "optimizer": run["optimizer"], "objective": run["objective"],
            "raw_metrics_match_saved": True, "mitigated_metrics_match_saved": True,
            "shots": raw["shots"],
        })
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"path": str(path), "run_checks": checks}


def audit_ten_item() -> dict[str, object]:
    path = Path("results/stage2_qiskit_ten_candidate/ten_candidate_exact_six_qiskit_results.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pool = pd.read_csv("results/stage2_development/development_candidate_pool.csv", dtype={"customer_id": str})
    operational, constraints = load_stage2_source("data/raw/bank_campaign_participant_release.xlsx")
    problem = problem_from_candidates(pool, operational, constraints)
    optimum, _ = enumerate_optimum(problem)
    raw_counts = manifest["noisy_finite_shot"]["counts"]
    raw = independent_stage2_metrics(raw_counts, problem, optimum)
    for metric, expected in (
        ("feasible_probability", manifest["noisy_finite_shot"]["metrics"]["feasible_probability"]),
        ("optimal_probability", manifest["noisy_finite_shot"]["metrics"]["exact_optimum_sampling_probability"]),
    ):
        assert_close(f"10-item raw {metric}", raw[metric], expected)
    matrices = [np.asarray(m, dtype=float) for m in manifest["readout_mitigation"]["assignment_matrices_observed_given_true"]]
    corrected = mitigate_readout(raw_counts, matrices)
    manifest["readout_mitigation"]["quasi_distribution"] = np.asarray(corrected["quasi_distribution"]).tolist()
    manifest["readout_mitigation"]["projected_distribution"] = np.asarray(corrected["projected_distribution"]).tolist()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "path": str(path), "raw_metrics_match_saved": True,
        "shots": raw["shots"], "independently_recomputed_raw_metrics": raw,
    }


def audit_stage3() -> dict[str, object]:
    path = Path("results/stage3_qiskit_noise/qiskit_stage3_noise_results.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    probabilities = [row["success_probability"] for row in manifest["selected_customers"]]
    exact = probability_at_least(probabilities, threshold=3)
    assert_close("Stage 3 Poisson-binomial", exact, manifest["exact_probability_at_least_three"])
    assert_close("Stage 3 Qiskit statevector", exact, manifest["qiskit_statevector_probability_at_least_three"])
    matrices = [np.asarray(m, dtype=float) for m in manifest["calibration"]["matrices_observed_given_true"]]
    rows = []
    for result in manifest["seed_level_results"]:
        counts = result["raw_counts"]
        shots = sum(counts.values())
        successes = sum(count for bits, count in counts.items() if bits.replace(" ", "").count("1") >= 3)
        raw = successes / shots
        assert_close("Stage 3 raw measured threshold probability", raw, result["noisy_finite_shot"]["probability_at_least_three"])
        corrected = mitigate_readout(counts, matrices)
        quasi = np.asarray(corrected["quasi_distribution"])
        projected = np.asarray(corrected["projected_distribution"])
        mitigated = float(sum(p for mask, p in enumerate(projected) if mask.bit_count() >= 3))
        assert_close("Stage 3 projected threshold probability", mitigated, result["noisy_readout_mitigated"]["probability_at_least_three_projected"])
        result["noisy_readout_mitigated"]["quasi_distribution"] = quasi.tolist()
        result["noisy_readout_mitigated"]["projected_distribution"] = projected.tolist()
        rows.append({"seed": result["seed"], "shots": shots, "raw_probability": raw, "mitigated_probability": mitigated})
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"path": str(path), "exact_probability": exact, "statevector_matches_exact": True, "seed_checks": rows}


def main() -> None:
    report = {
        "status": "passed_independent_saved_count_audit",
        "stage1_test_data_read": False,
        "stage2_four_item": audit_four_item(),
        "stage2_ten_item": audit_ten_item(),
        "stage3": audit_stage3(),
    }
    output = Path("results/qiskit_noise_independent_audit.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
