"""Run IWS-XY on the same synthetic binding instance used by other methods."""

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
from atomiq.stage2_iws_xy import run_iws_xy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_iws_binding_development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_iws_binding_development"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["stage2_iws_binding_development"]
    problem = four_item_binding_problem()
    optimum, feasible = enumerate_optimum(problem)
    started = time.perf_counter()
    rows = [asdict(run_iws_xy(
        problem, seed=seed, depth=config["depth"],
        shots_per_iteration=config["shots_per_iteration"],
        total_shots=config["total_shots_per_seed"],
        inverse_temperature=config["inverse_temperature"], epsilon=config["epsilon"],
        max_evaluations=config["max_objective_evaluations"],
    )) for seed in config["random_seeds"]]
    manifest = {
        "status": "completed_synthetic_binding_constraint_iws_diagnostic",
        "method": "exact-K weighted Johnson IWS-XY with IF-style objective cost",
        "instance_is_synthetic": True,
        "stage1_or_challenge_data_read": False,
        "candidate_count": len(problem.candidates),
        "selected_count": problem.selected_count,
        "feasible_exact_cardinality_subsets": len(feasible),
        "total_exact_cardinality_subsets": 6,
        "exact_optimum": optimum.objective,
        "exact_optimum_ids": optimum.customer_ids,
        "configuration": config,
        "runtime_seconds": time.perf_counter() - started,
        "results": rows,
        "limitations": [
            "The source paper's shallow one-hot hardware mixer is not retained by the exact-K Johnson graph generalization.",
            "Exact compressed-sector simulation and ideal draws only; no finite-gate or noisy hardware simulation.",
            "This tiny case is for correctness and method screening, not a generalization or quantum-advantage claim.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "iws_xy_binding_development_results.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "results"}, indent=2))
    for row in rows:
        print(json.dumps({
            "seed": row["seed"], "optimizer_success": row["optimizer_success"],
            "evaluations": row["optimization_evaluations"],
            "iterations": row["iterations"],
            "ground_overlap": row["mixer_ground_overlap"],
        }))


if __name__ == "__main__":
    main()
