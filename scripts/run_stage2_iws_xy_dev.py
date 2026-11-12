"""Run the exact-K Johnson-graph IWS-XY adaptation on the training pool."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from atomiq.stage2 import enumerate_optimum, load_stage2_source, problem_from_candidates
from atomiq.stage2_iws_xy import run_iws_xy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path("data/raw/bank_campaign_participant_release.xlsx"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_iws_xy_development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_iws_xy_development"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["stage2_iws_xy_development"]
    pool = pd.read_csv(
        "results/stage2_development/development_candidate_pool.csv",
        usecols=["customer_id", "probability"], dtype={"customer_id": str},
    )
    operational, constraints = load_stage2_source(args.workbook)
    problem = problem_from_candidates(pool, operational, constraints)
    optimum, feasible = enumerate_optimum(problem)
    started = time.perf_counter()
    rows = []
    for seed in config["random_seeds"]:
        result = run_iws_xy(
            problem,
            seed=seed,
            depth=config["depth"],
            shots_per_iteration=config["shots_per_iteration"],
            total_shots=config["total_shots_per_seed"],
            inverse_temperature=config["inverse_temperature"],
            epsilon=config["epsilon"],
            max_evaluations=config["max_objective_evaluations"],
        )
        rows.append(asdict(result))
    manifest = {
        "status": "complete_development_only_exact_k_adaptation",
        "method": "IWS-XY adaptation with weighted Johnson graph and paper-style Boltzmann marginal updates",
        "exact_k_adaptation": "subset vertices are K-of-N bitstrings; graph edges swap one selected and one unselected item; per-edge warm-start XY block uses q=P(S)/(P(S)+P(T)); sum normalized by degree K(N-K)",
        "mixer_ground_state": "sqrt(P(S)) is unique ground state at eigenvalue -1 for connected Johnson graph",
        "source_paper_shallow_circuit_claim_applies": False,
        "simulation_representation": "exact 210-dimensional cardinality-sector dense matrix evolution; not decomposed to a gate-level hardware circuit",
        "stage1_frozen": False,
        "stage1_test_sheet_or_labels_read": False,
        "synthetic_constraints": False,
        "candidate_count": len(problem.candidates),
        "selected_count": problem.selected_count,
        "feasible_exact_cardinality_subsets": len(feasible),
        "total_exact_cardinality_subsets": 210,
        "all_fixed_k_subsets_feasible": len(feasible) == 210,
        "exact_optimum": optimum.objective,
        "exact_optimum_ids": optimum.customer_ids,
        "configuration": config,
        "runtime_seconds": time.perf_counter() - started,
        "seeds": rows,
        "limitations": [
            "The exact-K Johnson-graph generalization is mathematically adapted from the paper's one-hot edge Hamiltonian, not an exact reproduction of its one-hot problem or shallow circuit.",
            "The development pool has no binding budget/capacity constraints, so this experiment does not test constrained feasibility handling.",
            "Dense compressed-sector simulation and ideal sampling do not estimate hardware gate noise or device runtime.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "iws_xy_development_results.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "seeds"}, indent=2))
    for row in rows:
        print(json.dumps({
            "seed": row["seed"], "gamma": row["gamma"], "beta": row["beta"],
            "optimizer_success": row["optimizer_success"],
            "iterations": row["iterations"],
            "ground_overlap": row["mixer_ground_overlap"],
        }))


if __name__ == "__main__":
    main()
