"""Run a bounded actual-candidate exact-six QAOA comparison in Qiskit/Aer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

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
from atomiq.stage2_qiskit_circuits import (
    bind_angles,
    build_parameterized_qaoa,
    cvar_lower_tail,
    exact_distribution,
    optimize_qaoa,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage2_final"


def distribution_from_counts(counts: dict[str, int], n: int) -> np.ndarray:
    distribution = np.zeros(1 << n, dtype=float)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Aer returned no samples.")
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        if len(cleaned) != n:
            raise ValueError(f"Expected {n}-bit Stage 2 result, got {bitstring!r}.")
        distribution[int(cleaned, 2)] += count / total
    return distribution


def score(problem, distribution, optimum, cvar_alpha=0.25):
    feasible = np.zeros(1 << len(problem.candidates), dtype=bool)
    objectives = np.zeros_like(feasible, dtype=float)
    for mask in range(len(feasible)):
        ids = [c.customer_id for q, c in enumerate(problem.candidates) if (mask >> q) & 1]
        result = evaluate_selection(problem, ids)
        feasible[mask] = result.feasible
        objectives[mask] = result.objective
    p_feasible = float(distribution[feasible].sum())
    expected_feasible_reward = float(distribution[feasible] @ objectives[feasible])
    conditional_reward = expected_feasible_reward / p_feasible if p_feasible else None
    optimal = feasible & (np.abs(objectives - optimum.objective) <= 1e-10)
    best_support = float(objectives[feasible & (distribution > 0)].max()) if np.any(feasible & (distribution > 0)) else None
    return {
        "feasible_shot_probability": p_feasible,
        "expected_objective_unconditional_with_zero_for_infeasible": expected_feasible_reward,
        "expected_objective_conditional_on_feasibility": conditional_reward,
        "cvar_objective_lower_quartile_conditional_on_feasibility": cvar_lower_tail(
            -objectives[feasible], distribution[feasible], cvar_alpha
        ) * -1 if p_feasible > 0 else None,
        "exact_optimum_sampling_probability": float(distribution[optimal].sum()),
        "objective_gap_conditional_on_feasibility": optimum.objective - conditional_reward if conditional_reward is not None else None,
        "best_feasible_measured_objective": best_support,
        "probability_normalization": float(distribution.sum()),
    }


def main() -> None:
    candidates = pd.read_csv(OUT / "actual_top_ten.csv", dtype={"customer_id": str})
    operational, constraints = load_stage2_source(ROOT / "data/raw/bank_campaign_participant_release.xlsx")
    problem = problem_from_candidates(candidates, operational, constraints)
    optimum, feasible = enumerate_optimum(problem)
    if len(feasible) != 210:
        raise ValueError("Expected the actual challenge pool's 210 feasible subsets.")
    ansatz = build_parameterized_qaoa(problem, depth=1, kind="xy", topology="cycle")
    if not ansatz.circuit.metadata["fixed_sector_feasible_additive_cost"]:
        raise AssertionError("Feasible fixed-K pool should use its exact additive separator.")

    fit_records = []
    for optimizer, objective_kind in (
        ("COBYLA", "expectation"), ("SPSA", "expectation"),
        ("COBYLA", "cvar"), ("SPSA", "cvar"),
    ):
        fit = optimize_qaoa(
            ansatz, optimizer=optimizer, objective_kind=objective_kind,
            seed=11, max_evaluations=10, alpha=0.25,
        )
        exact = exact_distribution(ansatz, fit.parameters)
        circuit = bind_angles(ansatz, fit.parameters)
        ideal_counts, ideal_runtime, ideal_compiled = run_aer_counts(
            circuit, profile=illustrative_profile_for_size(10), shots=2048, seed=11, noise=False,
        )
        noisy_counts, noisy_runtime, noisy_compiled = run_aer_counts(
            circuit, profile=illustrative_profile_for_size(10), shots=2048, seed=11, noise=True,
        )
        fit_records.append({
            "optimizer": optimizer,
            "training_objective": objective_kind,
            "parameters_gamma_then_beta": list(fit.parameters),
            "seed": fit.seed,
            "max_evaluations": 10,
            "actual_evaluations": fit.evaluations,
            "convergence_certified": fit.success,
            "optimizer_message": fit.message,
            "optimization_runtime_seconds": fit.runtime_seconds,
            "ideal_statevector_metrics": score(problem, exact, optimum),
            "ideal_finite_shot": {
                "shots": sum(ideal_counts.values()),
                "metrics": score(problem, distribution_from_counts(ideal_counts, 10), optimum),
                "runtime_seconds": ideal_runtime,
                "transpiled_depth": int(ideal_compiled.depth()),
                "counts": ideal_counts,
            },
            "noisy_finite_shot": {
                "shots": sum(noisy_counts.values()),
                "metrics": score(problem, distribution_from_counts(noisy_counts, 10), optimum),
                "runtime_seconds": noisy_runtime,
                "transpiled_depth": int(noisy_compiled.depth()),
                "counts": noisy_counts,
            },
            "circuit": circuit,
        })

    profile = illustrative_profile_for_size(10)
    matrices, calibration_runtime = calibration_assignment_matrices(
        10, profile=profile, shots=2048, seed=731,
    )
    for record in fit_records:
        mitigation_started = perf_counter()
        mitigation = mitigate_readout(record["noisy_finite_shot"]["counts"], matrices)
        mitigation_runtime = perf_counter() - mitigation_started
        distribution = np.asarray(mitigation["projected_distribution"])
        record["noisy_readout_mitigated"] = {
            "metrics": score(problem, distribution, optimum),
            "condition_number": float(mitigation["condition_number"]),
            "negative_quasi_mass": float(mitigation["negative_quasi_mass"]),
            "runtime_seconds": mitigation_runtime,
            "bootstrap_feasibility_variance": bootstrap_metric_variance(
                record["noisy_finite_shot"]["counts"], matrices,
                lambda p: float(sum(p[m] for m in range(1 << 10)
                    if evaluate_selection(problem, [c.customer_id for q, c in enumerate(problem.candidates) if (m >> q) & 1]).feasible)),
                seed=1111, replicates=100,
            ),
        }
        record["resources"] = resource_report(
            record.pop("circuit"), profile, seed=11,
        )

    result = {
        "status": "completed_actual_challenge_stage2_qiskit_aer_qaoa",
        "classification": ["actual ten-qubit parameterized Qiskit exact-six XY-QAOA circuit", "ideal statevector", "ideal finite-shot Aer", "illustrative noisy Aer", "readout-calibrated mitigation"],
        "candidate_source": "frozen Stage 1 feature-only inference; test labels not used",
        "candidate_pool": candidates[["customer_id", "probability"]].to_dict(orient="records"),
        "exact_classical_optimum": {"ids": list(optimum.customer_ids), "objective": optimum.objective},
        "feasible_exact_six_subsets": len(feasible),
        "ansatz": {"kind": "XY-QAOA", "topology": "cycle", "depth": 1,
                   "initial_state": "uniform Dicke state over all 10 choose 6 strings",
                   "mixer": "cycle XY exchange interactions preserving exact cardinality",
                   "cost_phase": "exact additive expected-net-value phase on the feasible fixed-six sector",
                   "noise_profile": profile.to_dict(),
                   "readout_calibration_shots_per_state_per_qubit": 2048,
                   "readout_calibration_runtime_seconds": calibration_runtime},
        "results": [{key: value for key, value in record.items() if key != "noisy_finite_shot" or True} for record in fit_records],
        "limitations": [
            "All 210 fixed-six subsets are feasible, so the fixed-sector objective is exactly additive; this pool cannot evaluate binding resource constraints.",
            "Optimization used a deliberately small ten-call deterministic budget per variant; none is a convergence certificate or exhaustive parameter search.",
            "Finite-shot conditions use one common 2,048-shot seed per variant; optimizer outcomes are diagnostics, not a statistically powered comparison.",
            "Illustrative repeated 10-qubit line profile is not hardware calibration; no physical QPU was used.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "actual_stage2_qaoa_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "saved": str(path.relative_to(ROOT)),
        "exact_optimum": result["exact_classical_optimum"],
        "results": [{
            "optimizer": row["optimizer"], "objective": row["training_objective"],
            "evaluations": row["actual_evaluations"], "success": row["convergence_certified"],
            "ideal_statevector": row["ideal_statevector_metrics"],
            "ideal_finite": row["ideal_finite_shot"]["metrics"],
            "noisy": row["noisy_finite_shot"]["metrics"],
            "mitigated": row["noisy_readout_mitigated"]["metrics"],
            "resources": row["resources"],
        } for row in fit_records],
    }, indent=2))


if __name__ == "__main__":
    main()
