"""Run QPE-based IF-QAOA on a deterministic binding-constraint diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml
from qiskit import transpile

from atomiq.stage2 import enumerate_optimum, evaluate_selection
from atomiq.stage2_development_fixtures import four_item_binding_problem
from atomiq.stage2_if_qaoa import (
    build_if_qaoa_circuit,
    optimize_if_qaoa,
    selector_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_if_qaoa_development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_if_qaoa_development"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["stage2_if_qaoa_development"]
    problem = four_item_binding_problem()
    optimum, feasible = enumerate_optimum(problem)
    rows = []
    for seed in config["random_seeds"]:
        result = optimize_if_qaoa(
            problem, depth=config["depth"], seed=seed,
            max_evaluations=config["max_evaluations"],
            topology=config["mixer_topology"],
        )
        circuit = build_if_qaoa_circuit(problem, result.gamma, result.beta)
        distribution = selector_distribution(problem, circuit)
        supported_feasible = []
        for mask, probability in enumerate(distribution):
            if probability <= 1e-12:
                continue
            chosen = [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]
            evaluation = evaluate_selection(problem, chosen)
            if evaluation.feasible:
                supported_feasible.append(evaluation.objective)
        compiled = transpile(
            circuit, basis_gates=["cx", "rz", "sx", "x"],
            optimization_level=0, seed_transpiler=seed,
        )
        rows.append({
            "seed": seed,
            "parameters": {"gamma": result.gamma, "beta": result.beta},
            "optimizer_success": result.optimizer_success,
            "optimizer_message": result.optimizer_message,
            "objective_evaluations": result.objective_evaluations,
            "expected_feasible_objective": result.expected_objective,
            "feasible_probability": result.feasible_probability,
            "optimal_solution_probability": result.optimal_solution_probability,
            "best_supported_feasible_objective": max(supported_feasible),
            "logical_qubits": result.logical_qubits,
            "circuit_depth": compiled.depth(),
            "gate_counts": {str(name): int(count) for name, count in compiled.count_ops().items()},
        })
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "completed_synthetic_binding_constraint_diagnostic",
        "method": "IF-QAOA with exact signed-integer QPE indicators, controlled objective phase, and uncomputation",
        "constraint_semantics": "exactly two selected; budget <= 3; capacity <= 3",
        "fixture_is_synthetic": True,
        "stage1_data_or_test_data_read": False,
        "candidate_count": len(problem.candidates),
        "feasible_exact_cardinality_subsets": len(feasible),
        "total_exact_cardinality_subsets": 6,
        "exact_optimum": optimum.objective,
        "exact_optimum_ids": optimum.customer_ids,
        "configuration": config,
        "results": rows,
        "limitations": [
            "Tiny synthetic instance establishes oracle arithmetic and end-to-end circuit behavior; it is not a performance comparison on the challenge-sized instance.",
            "No physical quantum hardware or noise model was used.",
            "The four-item synthetic fixture is a correctness diagnostic only; it is not the challenge-sized candidate pool.",
            "COBYLA reached the configured 100-evaluation cap; optimizer_success is false and this run is budget-truncated.",
        ],
    }
    (args.output / "if_qaoa_development_results.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
