"""Run DLA warm-start shared/multi-angle experiments and random-init ablations."""

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
from atomiq.stage2_dla_warm_start import analyze_dla, run_dla_warm_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_dla_warm_start_development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_dla_warm_start_development"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["stage2_dla_warm_start_development"]
    problem = four_item_binding_problem()
    optimum, feasible = enumerate_optimum(problem)
    dla = analyze_dla(len(problem.candidates), problem.selected_count, max_dimension=64)
    started = time.perf_counter()
    rows = []
    for parameterization in config["parameterizations"]:
        for seed in config["random_seeds"]:
            result = run_dla_warm_start(
                problem, seed=seed, parameterization=parameterization,
                depth=config["depth"],
                restricted_steps=config["restricted_adam_steps"],
                full_steps=config["full_adam_steps"],
                learning_rate=config["learning_rate"],
            )
            rows.append(asdict(result))
    manifest = {
        "status": "completed_synthetic_binding_constraint_methodology_diagnostic",
        "method": "DLA-informed warm-start QAOA: cycle XY + local Z pretrain; zero-added all-pairs ZZ; transfer and fine-tune",
        "instance_is_synthetic": True,
        "stage1_or_challenge_data_read": False,
        "constraint_semantics": "exactly two; contact cost <=3; capacity <=3",
        "all_ansatz_generators_preserve_hamming_weight": True,
        "ansatz_generator_counts": {
            "restricted": len(problem.candidates) + len(problem.candidates),
            "full": len(problem.candidates) + len(problem.candidates) + len(problem.candidates) * (len(problem.candidates) - 1) // 2,
        },
        "dla_numerical_closure": asdict(dla),
        "exact_optimum": optimum.objective,
        "exact_optimum_ids": optimum.customer_ids,
        "feasible_exact_cardinality_subsets": len(feasible),
        "total_exact_cardinality_subsets": 6,
        "configuration": config,
        "runtime_seconds": time.perf_counter() - started,
        "seeds": rows,
        "limitations": [
            "DLA closure is numerically computed in the six-dimensional fixed-K sector for the 4-item diagnostic; conclusions do not establish asymptotic trainability.",
            "Exact statevector simulation only; finite-shot and noisy comparisons remain outstanding.",
            "This is an exact-K adaptation of the cited DLA warm-start circuit, not a reproduction of its original problem instances.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dla_warm_start_development_results.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "seeds"}, indent=2))
    for row in rows:
        print(json.dumps({
            key: row[key] for key in (
                "seed", "parameterization", "restricted_initial_cost", "restricted_final_cost",
                "full_initial_cost", "full_final_cost", "random_full_initial_cost",
                "random_full_final_cost", "full_expected_feasible_objective",
                "random_full_expected_feasible_objective", "exact_feasible_probability",
                "exact_optimal_solution_probability", "sector_leakage",
            )
        }))


if __name__ == "__main__":
    main()
