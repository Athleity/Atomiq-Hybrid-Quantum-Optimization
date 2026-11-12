"""Run the QTG state generator and AAM-QAOA mixer on a shared binding fixture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from atomiq.stage2 import enumerate_optimum
from atomiq.stage2_development_fixtures import four_item_binding_problem
from atomiq.stage2_qtg_qaoa import qtg_initial_state, run_qtg_qaoa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_qtg_qaoa_development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_qtg_qaoa_development"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["stage2_qtg_qaoa_development"]
    problem = four_item_binding_problem()
    optimum, feasible = enumerate_optimum(problem)
    started = time.perf_counter()
    tree_summaries = []
    rows = []
    for bias in config["bias_values"]:
        masks, amplitudes = qtg_initial_state(problem, bias_b=bias)
        tree_summaries.append({
            "bias_b": bias,
            "positive_feasible_states": len(masks),
            "state_norm": float((abs(amplitudes) ** 2).sum()),
            "amplitude_min": float(amplitudes.real.min()),
            "amplitude_max": float(amplitudes.real.max()),
        })
        for seed in config["random_seeds"]:
            result = run_qtg_qaoa(
                problem,
                seed=seed,
                bias_b=bias,
                depth=config["depth"],
                max_evaluations=config["max_objective_evaluations"],
                shots=config["finite_shots_per_seed"],
            )
            rows.append(asdict(result))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "completed_synthetic_binding_constraint_methodology_diagnostic",
        "method": "QTG biased-Hadamard tree state preparation with paper's AAM-QAOA rank-one mixer",
        "exact_k_two_inequality_adaptation": "suffix feasibility dynamic program checks remaining exact count and both resource limits; reversible implementation would require compiled lookup/arithmetic controls",
        "high_level_simulation": "complex amplitude propagation over only feasible exact-K assignments; mirrors the QTG paper's high-level statevector simulator",
        "hardware_gate_decomposition_available": False,
        "stage1_or_challenge_data_read": False,
        "instance_is_synthetic": True,
        "candidate_count": len(problem.candidates),
        "selected_count": problem.selected_count,
        "constraint_limits": {"budget": problem.budget_limit, "capacity": problem.capacity_limit},
        "feasible_states": len(feasible),
        "exact_optimum": optimum.objective,
        "exact_optimum_ids": optimum.customer_ids,
        "qtg_initial_state_checks": tree_summaries,
        "configuration": config,
        "runtime_seconds": time.perf_counter() - started,
        "results": rows,
        "limitations": [
            "The completion predicate is evaluated by memoized classical dynamic programming in this simulator. A quantum implementation needs reversible suffix-feasibility logic, count/cost/capacity registers and compiled arithmetic.",
            "No gate-level decomposition, finite-shot hardware noise model, or physical quantum processor data is reported.",
            "This is a two-resource exact-K adaptation of a one-capacity knapsack QTG/AAM-QAOA paper, not a reproduction of its benchmark instances.",
        ],
    }
    (args.output / "qtg_qaoa_development_results.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "results"}, indent=2))
    for result in rows:
        print(json.dumps({
            key: result[key] for key in (
                "seed", "bias_b", "gamma", "beta", "optimizer_success",
                "expected_objective", "feasible_probability", "optimal_solution_probability",
                "approximation_gap", "sampled_optimal_hits", "best_sampled_objective",
            )
        }))


if __name__ == "__main__":
    main()
