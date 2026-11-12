"""Execute the reusable exact-six XY circuit on the provisional 10-item training-only pool."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yaml

from atomiq.qiskit_noise import (
    bootstrap_metric_variance,
    calibration_assignment_matrices,
    illustrative_profile_for_size,
    mitigate_readout,
    resource_report,
    run_aer_counts,
)
from atomiq.stage2 import (
    enumerate_optimum,
    evaluate_selection,
    load_stage2_source,
    problem_from_candidates,
)
from atomiq.stage2_qiskit_circuits import bind_angles, build_parameterized_qaoa, exact_distribution


def counts_distribution(counts, n):
    p = np.zeros(1 << n, dtype=float)
    total = sum(counts.values())
    for bits, count in counts.items():
        key = bits.replace(" ", "")
        if len(key) != n:
            raise ValueError(f"Invalid {n}-qubit bitstring: {bits!r}")
        p[int(key, 2)] += count / total
    return p


def summarize(p, problem, optimum):
    n = len(problem.candidates)
    feasible = np.zeros(1 << n, dtype=bool)
    objective = np.zeros(1 << n)
    optimal = np.zeros(1 << n, dtype=bool)
    for mask in range(1 << n):
        selected = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
        result = evaluate_selection(problem, selected)
        feasible[mask] = result.feasible
        objective[mask] = result.objective
        optimal[mask] = result.feasible and abs(result.objective - optimum.objective) < 1e-10
    p_feasible = float(p[feasible].sum())
    expected = float(np.dot(p[feasible], objective[feasible]))
    conditional = expected / p_feasible if p_feasible > 0 else None
    return {
        "feasible_probability": p_feasible,
        "expected_feasible_objective": expected,
        "expected_objective_conditional_on_feasible": conditional,
        "exact_optimum_sampling_probability": float(p[optimal].sum()),
        "objective_gap_conditional_on_feasible": float(optimum.objective - conditional) if conditional is not None else None,
        "best_supported_feasible_objective": float(objective[feasible & (p > 1e-10)].max()) if np.any(feasible & (p > 1e-10)) else None,
        "distribution_sum": float(p.sum()),
    }


def main():
    config = yaml.safe_load(Path("configs/qiskit_circuit_noise.yaml").read_text(encoding="utf-8"))["qiskit_circuit_noise"]
    candidate_frame = pd.read_csv("results/stage2_development/development_candidate_pool.csv", dtype={"customer_id": str})
    operational, constraints = load_stage2_source("data/raw/bank_campaign_participant_release.xlsx")
    problem = problem_from_candidates(candidate_frame, operational, constraints)
    optimum, feasible = enumerate_optimum(problem)
    previous = json.loads(Path("results/stage2_quantum_development/xy_qaoa_development_results.json").read_text(encoding="utf-8"))
    seed_record = next(row for row in previous["results"] if row["seed"] == 11)
    parameters = [*seed_record["parameters"]["gamma"], *seed_record["parameters"]["beta"]]
    ansatz = build_parameterized_qaoa(problem, depth=1, kind="xy", topology="cycle")
    if ansatz.circuit.num_qubits != 10 or problem.selected_count != 6:
        raise AssertionError("The provisional development pool must exercise a ten-qubit exact-six circuit.")
    circuit = bind_angles(ansatz, parameters)
    exact = summarize(exact_distribution(ansatz, parameters), problem, optimum)
    profile = illustrative_profile_for_size(10)
    ideal_counts, ideal_runtime, ideal_compiled = run_aer_counts(
        circuit, profile=profile, shots=2048, seed=11, noise=False
    )
    noisy_counts, noisy_runtime, noisy_compiled = run_aer_counts(
        circuit, profile=profile, shots=512, seed=11, noise=True
    )
    calibration, calibration_runtime = calibration_assignment_matrices(
        10, profile=profile, shots=2048, seed=731
    )
    mitigation_started = perf_counter()
    mitigation = mitigate_readout(noisy_counts, calibration)
    mitigation_runtime = perf_counter() - mitigation_started
    feasible_mask = np.zeros(1 << 10, dtype=float)
    for mask in range(1 << 10):
        selected = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
        feasible_mask[mask] = float(evaluate_selection(problem, selected).feasible)
    mitigation_variance = bootstrap_metric_variance(
        noisy_counts, calibration, lambda vector: float(np.dot(vector, feasible_mask)),
        seed=1111, replicates=200,
    )
    result = {
        "status": "completed_provisional_ten_candidate_exact_six_xy_qiskit_circuit",
        "qiskit_version": __import__("qiskit").__version__,
        "qiskit_aer_version": __import__("qiskit_aer").__version__,
        "classification": ["actual 10-qubit Qiskit parameterized XY-QAOA circuit", "ideal Statevector", "ideal finite-shot Aer", "noisy finite-shot Aer", "readout mitigation"],
        "data_boundary": {
            "candidate_source": "mean calibrated repeated OOF probabilities from Stage1_Train only",
            "stage1_frozen": False,
            "stage1_test_sheet_read": False,
            "stage1_test_labels_or_predictions_read": False,
            "candidate_pool_artifact": "results/stage2_development/development_candidate_pool.csv",
            "original_angle_source": "previous depth-one exact-statevector XY-QAOA fit, seed 11; circuit convention verified in unit tests",
        },
        "candidate_count": len(problem.candidates),
        "selected_count": problem.selected_count,
        "feasible_exact_six_count": len(feasible),
        "exact_optimum": optimum.objective,
        "exact_optimum_ids": optimum.customer_ids,
        "parameters_gamma_then_beta": parameters,
        "ideal_statevector_metrics": exact,
        "ideal_finite_shot": {
            "shots": sum(ideal_counts.values()),
            "metrics": summarize(counts_distribution(ideal_counts, 10), problem, optimum),
            "runtime_seconds": ideal_runtime,
            "transpiled_depth": int(ideal_compiled.depth()),
            "counts": ideal_counts,
        },
        "noisy_finite_shot": {
            "shots": sum(noisy_counts.values()),
            "metrics": summarize(counts_distribution(noisy_counts, 10), problem, optimum),
            "runtime_seconds": noisy_runtime,
            "transpiled_depth": int(noisy_compiled.depth()),
            "counts": noisy_counts,
        },
        "readout_mitigation": {
            "method": "per-qubit Aer readout-only prepared-state assignment calibration, tensor-product matrix inversion and simplex projection",
            "calibration_shots_per_state_per_qubit": 2048,
            "calibration_runtime_seconds": calibration_runtime,
            "mitigation_runtime_seconds": mitigation_runtime,
            "assignment_matrices_observed_given_true": [matrix.tolist() for matrix in calibration],
            "bootstrap_feasibility_variance": mitigation_variance,
            "metrics_quasi": summarize(np.asarray(mitigation["quasi_distribution"]), problem, optimum),
            "metrics_projected": summarize(np.asarray(mitigation["projected_distribution"]), problem, optimum),
            "quasi_distribution": np.asarray(mitigation["quasi_distribution"]).tolist(),
            "projected_distribution": np.asarray(mitigation["projected_distribution"]).tolist(),
            "condition_number": mitigation["condition_number"],
            "negative_quasi_mass": mitigation["negative_quasi_mass"],
        },
        "resources": resource_report(circuit, profile, seed=11),
        "noise_profile": profile.to_dict(),
        "limitations": [
            "The provisional OOF pool has all 210 exact-six selections feasible, so resource feasibility cannot be evaluated on this candidate pool.",
            "Angle fit is reused from the earlier exact-sector development result; the present circuit-level resource and noise results are newly executed.",
            "The ten-qubit noise profile repeats assumed seven-qubit parameters for qubits 7-9; it is a declared synthetic assumption, not device calibration.",
            "The original phase oracle and mixer are compiled to a line, with substantial routing/state-preparation overhead; this is a simulator feasibility run, not an efficient hardware circuit.",
        ],
    }
    output = Path("results/stage2_qiskit_ten_candidate")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "ten_candidate_exact_six_qiskit_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"ideal_finite_shot", "noisy_finite_shot", "readout_mitigation"}}, indent=2))
    print("finite_shot", json.dumps({
        "ideal": result["ideal_finite_shot"]["metrics"],
        "noisy": result["noisy_finite_shot"]["metrics"],
        "mitigated": result["readout_mitigation"]["metrics_projected"],
    }, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
