"""Run standard XY-QAOA controls on the shared synthetic binding fixture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from atomiq.stage2 import enumerate_optimum, evaluate_selection
from atomiq.stage2_development_fixtures import four_item_binding_problem
from atomiq.stage2_quantum import (
    build_xy_qaoa_circuit,
    circuit_resource_summary,
    exact_qaoa_distribution,
    illustrative_noise_model,
    optimize_xy_qaoa,
    sample_aer,
)


def _summarize(counts, problem, optimum):
    total = sum(counts.values())
    feasible = optimal = 0
    best = None
    for raw_bits, frequency in counts.items():
        mask = int(raw_bits.replace(" ", ""), 2)
        ids = [c.customer_id for index, c in enumerate(problem.candidates) if (mask >> index) & 1]
        result = evaluate_selection(problem, ids)
        if result.feasible:
            feasible += frequency
            optimal += frequency * (abs(result.objective - optimum.objective) <= 1e-10)
            best = result.objective if best is None else max(best, result.objective)
    return {
        "shots": total,
        "feasible_rate": feasible / total,
        "optimal_selection_probability": optimal / total,
        "best_sampled_feasible_objective": best,
        "raw_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_xy_binding_development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_xy_binding_development"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["stage2_xy_binding_development"]
    problem = four_item_binding_problem()
    optimum, feasible = enumerate_optimum(problem)
    started = time.perf_counter()
    rows = []
    for seed in config["random_seeds"]:
        optimized = optimize_xy_qaoa(
            problem, depth=config["depth"], seed=seed,
            max_evaluations=config["max_objective_evaluations"], topology="cycle",
        )
        circuit = build_xy_qaoa_circuit(problem, optimized.gamma, optimized.beta, topology="cycle")
        ideal = sample_aer(circuit, shots=config["ideal_shots"], seed=seed)
        noisy = sample_aer(
            circuit, shots=config["illustrative_noisy_shots"], seed=seed,
            noise_model=illustrative_noise_model(),
        )
        distribution = exact_qaoa_distribution(problem, circuit)
        rows.append({
            "seed": seed,
            "gamma": optimized.gamma,
            "beta": optimized.beta,
            "optimizer_success": optimized.optimizer_success,
            "optimizer_message": optimized.optimizer_message,
            "objective_evaluations": optimized.evaluations,
            "exact_expected_objective": optimized.exact_expected_objective,
            "exact_feasible_probability": optimized.exact_feasible_probability,
            "exact_optimal_solution_probability": optimized.exact_optimal_solution_probability,
            "approximation_gap": (optimum.objective - optimized.exact_expected_objective) / max(abs(optimum.objective), 1e-12),
            "ideal_finite_shot": _summarize(ideal, problem, optimum),
            "illustrative_noisy_finite_shot": _summarize(noisy, problem, optimum),
            "resources": circuit_resource_summary(circuit, seed=seed),
            "state_norm": float(distribution.sum()),
        })
    manifest = {
        "status": "completed_synthetic_binding_xy_control",
        "instance_is_synthetic": True,
        "stage1_or_challenge_data_read": False,
        "candidate_count": len(problem.candidates),
        "selected_count": problem.selected_count,
        "budget_limit": problem.budget_limit,
        "capacity_limit": problem.capacity_limit,
        "feasible_exact_cardinality_subsets": len(feasible),
        "total_exact_cardinality_subsets": 6,
        "exact_optimum": optimum.objective,
        "exact_optimum_ids": optimum.customer_ids,
        "configuration": config,
        "runtime_seconds": time.perf_counter() - started,
        "noise_model_is_illustrative_not_hardware_calibrated": True,
        "results": rows,
        "limitations": [
            "Tiny four-candidate exact-simulation control only.",
            "Illustrative Aer noise is a stress-test assumption, not device calibration.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "xy_qaoa_binding_development_results.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "results"}, indent=2))
    for row in rows:
        print(json.dumps({
            key: row[key] for key in (
                "seed", "optimizer_success", "objective_evaluations", "exact_expected_objective",
                "exact_feasible_probability", "exact_optimal_solution_probability",
                "ideal_finite_shot", "illustrative_noisy_finite_shot",
            )
        }, default=list))


if __name__ == "__main__":
    main()
