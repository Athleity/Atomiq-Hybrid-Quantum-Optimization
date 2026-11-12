"""Execute parameterized Stage 2 QAOA circuits with ideal/noisy Aer and readout correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from atomiq.qiskit_noise import (
    IllustrativeBackendProfile,
    bootstrap_metric_variance,
    calibration_assignment_matrices,
    mitigate_readout,
    resource_report,
    run_aer_counts,
)
from atomiq.stage2 import enumerate_optimum, evaluate_selection
from atomiq.stage2_development_fixtures import four_item_binding_problem
from atomiq.stage2_qiskit_circuits import (
    bind_angles,
    build_parameterized_qaoa,
    exact_distribution,
    interpolated_parameters,
    optimize_qaoa,
    score_distribution,
)


def counts_distribution(counts: dict[str, int], n: int) -> np.ndarray:
    values = np.zeros(1 << n, dtype=float)
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("No measurements were returned.")
    for key, count in counts.items():
        bitstring = key.replace(" ", "")
        if len(bitstring) != n or set(bitstring) - {"0", "1"}:
            raise ValueError(f"Invalid Qiskit bitstring {key!r} for {n} qubits.")
        values[int(bitstring, 2)] += count / shots
    return values


def summarize_distribution(distribution: np.ndarray, problem, optimum) -> dict[str, object]:
    feasible = np.zeros(distribution.size, dtype=bool)
    exact = np.zeros(distribution.size, dtype=float)
    optimal = np.zeros(distribution.size, dtype=bool)
    for mask in range(distribution.size):
        ids = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
        result = evaluate_selection(problem, ids)
        feasible[mask] = result.feasible
        exact[mask] = result.objective
        optimal[mask] = result.feasible and abs(result.objective - optimum.objective) <= 1e-10
    feasible_probability = float(distribution[feasible].sum())
    expected_feasible_objective = float(np.dot(distribution[feasible], exact[feasible]))
    conditional = expected_feasible_objective / feasible_probability if feasible_probability > 0 else None
    conditional_probabilities = distribution[feasible] / feasible_probability if feasible_probability > 0 else None
    cvar_objective = (
        _cvar_objective(exact[feasible], distribution[feasible])
        if feasible_probability > 0 else None
    )
    return {
        "feasible_probability_estimate": feasible_probability,
        "expected_objective_over_all_shots_feasible_only": expected_feasible_objective,
        "expected_objective_conditional_on_feasible": conditional,
        "cvar_25_percent_feasible_objective": cvar_objective,
        "exact_optimum_sampling_probability": float(distribution[optimal].sum()),
        "objective_gap_conditional_on_feasible": (
            float(optimum.objective - conditional) if conditional is not None else None
        ),
        "best_feasible_state_with_nonzero_estimated_mass": (
            float(max(exact[m] for m in range(len(exact)) if feasible[m] and distribution[m] > 1e-10))
            if np.any(feasible & (distribution > 1e-10)) else None
        ),
        "distribution_sum": float(distribution.sum()),
    }


def best_feasible_measured_objective(counts: dict[str, int], problem):
    best = None
    for bitstring, count in counts.items():
        if count <= 0:
            continue
        mask = int(bitstring.replace(" ", ""), 2)
        ids = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
        evaluation = evaluate_selection(problem, ids)
        if evaluation.feasible:
            best = evaluation.objective if best is None else max(best, evaluation.objective)
    return best


def _cvar_objective(values: np.ndarray, probabilities: np.ndarray, alpha: float = 0.25) -> float | None:
    mass = float(probabilities.sum())
    if mass <= 0:
        return None
    probabilities = probabilities / mass
    order = np.argsort(values)[::-1]
    remaining = alpha
    total = 0.0
    for index in order:
        take = min(remaining, float(probabilities[index]))
        total += take * float(values[index])
        remaining -= take
        if remaining <= 1e-14:
            break
    return total / alpha


def evaluate_conditions(ansatz, parameters, *, profile, shots, seed, calibration_matrices, calibration_runtime, problem, optimum, gate_scale=1.0):
    circuit = bind_angles(ansatz, parameters)
    state_distribution = exact_distribution(ansatz, parameters)
    exact = summarize_distribution(state_distribution, problem, optimum)
    ideal_counts, ideal_runtime, ideal_compiled = run_aer_counts(
        circuit, profile=profile, shots=shots, seed=seed, noise=False
    )
    ideal_shot = summarize_distribution(counts_distribution(ideal_counts, len(problem.candidates)), problem, optimum)
    ideal_shot["best_feasible_measured_objective"] = best_feasible_measured_objective(ideal_counts, problem)
    noisy_counts, noisy_runtime, noisy_compiled = run_aer_counts(
        circuit, profile=profile, shots=shots, seed=seed, noise=True, gate_scale=gate_scale
    )
    noisy_distribution = counts_distribution(noisy_counts, len(problem.candidates))
    noisy = summarize_distribution(noisy_distribution, problem, optimum)
    noisy["best_feasible_measured_objective"] = best_feasible_measured_objective(noisy_counts, problem)
    mitigation_started = perf_counter()
    mitigation = mitigate_readout(noisy_counts, calibration_matrices)
    mitigation_runtime = perf_counter() - mitigation_started
    quasi = summarize_distribution(np.asarray(mitigation["quasi_distribution"]), problem, optimum)
    projected_distribution = np.asarray(mitigation["projected_distribution"])
    projected = summarize_distribution(projected_distribution, problem, optimum)
    projected["best_feasible_measured_objective"] = None
    feasibility_mask = np.array([
        evaluate_selection(
            problem,
            [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1],
        ).feasible
        for mask in range(1 << len(problem.candidates))
    ])
    variance = bootstrap_metric_variance(
        noisy_counts, calibration_matrices,
        lambda p: float(np.dot(p, feasibility_mask)), seed=seed + 500, replicates=100,
    )
    return {
        "classification": {
            "ideal_statevector": True,
            "ideal_finite_shot_aer": True,
            "noisy_finite_shot_aer": True,
            "readout_mitigated_noisy_aer": True,
        },
        "shots": shots,
        "seed": seed,
        "gate_noise_scale": gate_scale,
        "ideal_statevector_metrics": exact,
        "ideal_finite_shot_metrics": ideal_shot,
        "noisy_finite_shot_metrics": noisy,
        "noisy_readout_mitigated_quasi_metrics": quasi,
        "noisy_readout_mitigated_projected_metrics": projected,
        "noisy_readout_mitigated_quasi_distribution": np.asarray(mitigation["quasi_distribution"]).tolist(),
        "noisy_readout_mitigated_projected_distribution": projected_distribution.tolist(),
        "readout_mitigation": {
            "condition_number": mitigation["condition_number"],
            "negative_quasi_mass": mitigation["negative_quasi_mass"],
            "calibration_shots_per_prepared_state_per_qubit": 4096,
            "calibration_runtime_seconds": calibration_runtime,
            "mitigation_runtime_seconds": mitigation_runtime,
            "bootstrap_feasibility_variance": variance,
        },
        "runtime_seconds": {
            "ideal_finite_shot_simulation": ideal_runtime,
            "noisy_finite_shot_simulation": noisy_runtime,
        },
        "transpilation": {
            "ideal_depth": int(ideal_compiled.depth()),
            "noisy_depth": int(noisy_compiled.depth()),
            "noisy_gate_counts": {str(k): int(v) for k, v in noisy_compiled.count_ops().items()},
        },
        "resource_report": resource_report(circuit, profile, seed=seed),
        "raw_noisy_counts": noisy_counts,
    }


def noisy_spsa_fit(ansatz, *, profile, seed, iterations, shots, objective_kind, alpha):
    rng = np.random.default_rng(seed)
    parameters = rng.uniform(0.0, 2.0 * np.pi, len(ansatz.gamma) + len(ansatz.beta))
    evaluations = 0
    started = perf_counter()

    def objective(x):
        nonlocal evaluations
        counts, _, _ = run_aer_counts(
            bind_angles(ansatz, x), profile=profile, shots=shots,
            seed=seed + evaluations, noise=True,
        )
        evaluations += 1
        p = counts_distribution(counts, int(np.log2(ansatz.costs.size)))
        return score_distribution(ansatz, p, objective_kind=objective_kind, alpha=alpha)

    best_x = parameters.copy()
    best = objective(best_x)
    for step in range(iterations):
        c = 0.18 / (step + 1) ** 0.101
        a = 0.12 / (step + 11) ** 0.602
        delta = rng.choice((-1.0, 1.0), size=parameters.size)
        high = objective(parameters + c * delta)
        low = objective(parameters - c * delta)
        parameters = (parameters - a * ((high - low) / (2 * c)) / delta) % (2 * np.pi)
        current = objective(parameters)
        if current < best:
            best, best_x = current, parameters.copy()
    return {
        "optimizer": "SPSA",
        "objective": objective_kind,
        "noise_optimized": True,
        "noise_profile": profile.name,
        "shots_per_objective_evaluation": shots,
        "iterations": iterations,
        "objective_evaluations": evaluations,
        "best_training_objective": best,
        "parameters": best_x.tolist(),
        "seed": seed,
        "runtime_seconds": perf_counter() - started,
        "termination": "fixed SPSA budget completed; not a convergence certificate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qiskit_circuit_noise.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_qiskit_noise"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["qiskit_circuit_noise"]
    profile = IllustrativeBackendProfile()
    problem = four_item_binding_problem()
    optimum, feasible_solutions = enumerate_optimum(problem)
    calibration_matrices, calibration_runtime = calibration_assignment_matrices(
        len(problem.candidates), profile=profile,
        shots=settings["readout_calibration_shots"], seed=settings["calibration_seed"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "qiskit_stage2_noise_in_progress.json"
    runs = []
    fit_table = []
    for kind, optimizer, objective in settings["optimization_variants"]:
        ansatz = build_parameterized_qaoa(
            problem, depth=settings["depth"], kind=kind, topology="cycle"
        )
        fit = optimize_qaoa(
            ansatz, optimizer=optimizer, objective_kind=objective,
            seed=settings["seed"], max_evaluations=settings["optimizer_evaluations"],
            alpha=settings["cvar_alpha"],
        )
        fit_table.append({
            "ansatz": kind, "optimizer": optimizer, "objective": objective,
            "parameters": fit.parameters, "training_score": fit.objective_value,
            "evaluations": fit.evaluations, "success": fit.success,
            "termination": fit.message, "runtime_seconds": fit.runtime_seconds,
        })
        runs.append({
            "ansatz": kind, "optimizer": optimizer, "objective": objective,
            "fit": fit_table[-1],
            "evaluation": evaluate_conditions(
                ansatz, fit.parameters, profile=profile, shots=settings["evaluation_shots"],
                seed=settings["evaluation_seed"], calibration_matrices=calibration_matrices,
                calibration_runtime=calibration_runtime,
                problem=problem, optimum=optimum,
            ),
        })
        checkpoint_path.write_text(json.dumps({
            "status": "in_progress_partial_results_not_benchmark",
            "configuration": settings,
            "optimizer_comparison": fit_table,
            "completed_runs": runs,
        }, indent=2, default=list), encoding="utf-8")

    # Warm-start/interpolation from the best depth-one XY expectation fit to depth two.
    xy_expectation = [row for row in fit_table if row["ansatz"] == "xy" and row["objective"] == "expectation"]
    selected_fit = min(xy_expectation, key=lambda row: row["training_score"])
    depth_two = build_parameterized_qaoa(
        problem, depth=2, kind="xy", topology="cycle"
    )
    warm_initial = interpolated_parameters(selected_fit["parameters"], 2)
    depth_two_fit = optimize_qaoa(
        depth_two, optimizer="COBYLA", objective_kind="expectation", seed=settings["seed"],
        max_evaluations=settings["optimizer_evaluations"], initial_parameters=warm_initial,
    )
    depth_two_result = {
        "ansatz": "xy", "depth": 2, "initialization": "linear_interpolation_from_depth_1",
        "initial_parameters": warm_initial, "fit": depth_two_fit.__dict__,
        "evaluation": evaluate_conditions(
            depth_two, depth_two_fit.parameters, profile=profile,
            shots=settings["evaluation_shots"], seed=settings["evaluation_seed"],
            calibration_matrices=calibration_matrices, calibration_runtime=calibration_runtime,
            problem=problem, optimum=optimum,
        ),
    }
    checkpoint_path.write_text(json.dumps({
        "status": "in_progress_partial_results_not_benchmark",
        "configuration": settings,
        "optimizer_comparison": fit_table,
        "completed_runs": runs,
        "depth_two_interpolated_warm_start": depth_two_result,
    }, indent=2, default=list), encoding="utf-8")

    # Same parameter point across mixer graph connectivities isolates routing cost.
    topology_cost = []
    for topology in ("path", "cycle", "complete"):
        topo = build_parameterized_qaoa(problem, depth=1, kind="xy", topology=topology)
        # Shared angles permit a topology-only resource comparison.
        topo_circuit = bind_angles(topo, selected_fit["parameters"])
        topology_cost.append({
            "topology": topology,
            "resource_report": resource_report(topo_circuit, profile, seed=settings["evaluation_seed"]),
        })

    # A small finite-shot noisy-training run measures the gap between ideal- and noise-trained angles.
    noise_fit = noisy_spsa_fit(
        build_parameterized_qaoa(problem, depth=1, kind="xy", topology="cycle"),
        profile=profile, seed=settings["noise_optimizer_seed"],
        iterations=settings["noise_optimizer_iterations"],
        shots=settings["noise_optimizer_shots"], objective_kind="expectation",
        alpha=settings["cvar_alpha"],
    )
    noise_ansatz = build_parameterized_qaoa(problem, depth=1, kind="xy", topology="cycle")
    noise_fit["evaluation"] = evaluate_conditions(
        noise_ansatz, noise_fit["parameters"], profile=profile,
        shots=settings["evaluation_shots"], seed=settings["evaluation_seed"],
        calibration_matrices=calibration_matrices, calibration_runtime=calibration_runtime,
        problem=problem, optimum=optimum,
    )

    # Compare the same ideal-trained parameters under two explicit gate-noise strengths.
    strength_sweep = []
    for scale in settings["gate_noise_scales"]:
        circuit = bind_angles(noise_ansatz, selected_fit["parameters"])
        counts, elapsed, compiled = run_aer_counts(
            circuit, profile=profile, shots=settings["evaluation_shots"],
            seed=settings["evaluation_seed"], noise=True, gate_scale=scale,
        )
        dist = counts_distribution(counts, len(problem.candidates))
        strength_sweep.append({
            "gate_noise_scale": scale,
            "metrics": summarize_distribution(dist, problem, optimum),
            "shots": sum(counts.values()), "runtime_seconds": elapsed,
            "transpiled_depth": int(compiled.depth()),
        })

    manifest = {
        "status": "completed_qiskit_parameterized_stage2_noise_diagnostic",
        "qiskit_version": __import__("qiskit").__version__,
        "qiskit_aer_version": __import__("qiskit_aer").__version__,
        "execution_classification": [
            "actual Qiskit QuantumCircuit", "ideal Statevector simulation",
            "ideal finite-shot Aer", "noisy finite-shot Aer", "readout assignment calibration Aer",
        ],
        "problem": {
            "synthetic_fixture": True, "stage1_or_challenge_data_read": False,
            "candidates": len(problem.candidates), "selected_count": problem.selected_count,
            "feasible_exact_cardinality_states": len(feasible_solutions),
            "exact_optimum": optimum.objective, "exact_optimum_ids": optimum.customer_ids,
        },
        "profile": profile.to_dict(),
        "configuration": settings,
        "readout_calibration": {
            "method": "Aer readout-only calibration circuits for prepared 0 and 1 on each physical qubit; gate errors disabled to isolate assignment channel",
            "calibration_runtime_seconds": calibration_runtime,
            "shots_per_prepared_state_per_qubit": settings["readout_calibration_shots"],
            "estimated_matrices_observed_given_true": [m.tolist() for m in calibration_matrices],
        },
        "optimizer_comparison": fit_table,
        "runs": runs,
        "depth_two_interpolated_warm_start": depth_two_result,
        "xy_mixer_connectivity_resource_comparison": topology_cost,
        "noise_optimized_spsa": noise_fit,
        "noise_strength_sensitivity": strength_sweep,
        "limitations": [
            "The noise profile is a hypothetical illustrative backend, not retrieved calibration; no physical QPU was used.",
            "Stage 2 experiment is a four-item synthetic binding-constraint instance, not a challenge-sized benchmark.",
            "The exact cost phase is built from state-selective multi-controlled phase gates; transpiled swap/depth results expose its small-instance synthesis overhead.",
            "SPSA completion at a fixed iteration limit does not certify convergence; report the endpoint and budget.",
            "Readout mitigation is local assignment-matrix inversion plus simplex projection; it cannot correct coherent, crosstalk, or gate noise.",
        ],
    }
    path = args.output / "qiskit_stage2_noise_results.json"
    path.write_text(json.dumps(manifest, indent=2, default=list), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"runs", "noise_optimized_spsa", "xy_mixer_connectivity_resource_comparison"}}, indent=2, default=list))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
