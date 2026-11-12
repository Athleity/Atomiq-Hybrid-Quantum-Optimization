"""Screen ideal/shot/noisy XY-QAOA on the training-only Stage 2 fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomiq.stage2 import evaluate_selection, enumerate_optimum, load_stage2_source, problem_from_candidates
from atomiq.stage2_quantum import (
    build_xy_qaoa_circuit,
    circuit_resource_summary,
    exact_qaoa_distribution,
    illustrative_noise_model,
    optimize_xy_qaoa,
    sample_aer,
)


def _summarize_counts(counts: dict[str, int], problem, optimum) -> dict[str, object]:
    total = sum(counts.values())
    feasible_shots = 0
    optimal_shots = 0
    best_observed = None
    for raw_bits, frequency in counts.items():
        mask = int(raw_bits.replace(" ", ""), 2)
        ids = [
            candidate.customer_id
            for index, candidate in enumerate(problem.candidates)
            if (mask >> index) & 1
        ]
        evaluation = evaluate_selection(problem, ids)
        if evaluation.feasible:
            feasible_shots += frequency
            if abs(evaluation.objective - optimum.objective) <= 1e-10:
                optimal_shots += frequency
            if best_observed is None or evaluation.objective > best_observed["objective"]:
                best_observed = {"customer_ids": evaluation.customer_ids, "objective": evaluation.objective}
    return {
        "shots": total,
        "feasible_shots": feasible_shots,
        "feasible_rate": feasible_shots / total,
        "optimal_shots": optimal_shots,
        "optimal_selection_probability": optimal_shots / total,
        "best_feasible_observed": best_observed,
        "raw_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=Path("data/raw/bank_campaign_participant_release.xlsx"))
    parser.add_argument("--stage1-results", type=Path, default=Path("results/stage1_classical"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_quantum_development.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/stage2_quantum_development"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["stage2_quantum_development"]

    predictions = pd.read_csv(
        args.stage1_results / "oof_predictions.csv",
        usecols=["customer_id", "prediction_type", "repeat", "probability"],
        dtype={"customer_id": str},
    )
    predictions = predictions.loc[
        predictions.prediction_type.eq("sigmoid_calibrated"),
        ["customer_id", "repeat", "probability"],
    ]
    means = predictions.groupby("customer_id", as_index=False).agg(
        probability=("probability", "mean"), repeats=("repeat", "nunique")
    )
    if len(means) != 300 or not means.repeats.eq(3).all():
        raise ValueError("Only three-repeat Stage1_Train OOF predictions are accepted for this fixture.")
    candidates = means.sort_values(
        ["probability", "customer_id"], ascending=[False, True], kind="stable"
    ).head(10)[["customer_id", "probability"]]
    operational, constraints = load_stage2_source(args.workbook)
    problem = problem_from_candidates(candidates, operational, constraints)
    optimum, feasible = enumerate_optimum(problem)
    args.output.mkdir(parents=True, exist_ok=True)

    summary = []
    for seed in config["random_seeds"]:
        optimized = optimize_xy_qaoa(
            problem,
            depth=config["depth"],
            seed=seed,
            max_evaluations=config["max_evaluations"],
            topology="cycle",
        )
        circuit = build_xy_qaoa_circuit(problem, optimized.gamma, optimized.beta, topology="cycle")
        exact = exact_qaoa_distribution(problem, circuit)
        finite = sample_aer(circuit, shots=config["finite_shots"], seed=seed)
        noisy = sample_aer(
            circuit,
            shots=config["noisy_shots"],
            seed=seed,
            noise_model=illustrative_noise_model(),
        )
        summary.append({
            "seed": seed,
            "optimizer": "COBYLA",
            "max_evaluations": config["max_evaluations"],
            "parameters": {"gamma": optimized.gamma, "beta": optimized.beta},
            "optimizer_success": optimized.optimizer_success,
            "optimizer_message": optimized.optimizer_message,
            "objective_evaluations": optimized.evaluations,
            "ideal_statevector": {
                "expected_objective": optimized.exact_expected_objective,
                "optimal_selection_probability": optimized.exact_optimal_solution_probability,
                "feasible_probability": optimized.exact_feasible_probability,
                "best_supported_objective": optimized.exact_best_supported_objective,
            },
            "ideal_finite_shot": _summarize_counts(finite, problem, optimum),
            "illustrative_noisy_finite_shot": _summarize_counts(noisy, problem, optimum),
            "circuit_resources": circuit_resource_summary(circuit, seed=seed),
        })

    manifest = {
        "status": "complete_development_only",
        "method": "standard shared-angle cycle XY-QAOA with exact Dicke initial state",
        "stage1_frozen": False,
        "stage1_test_sheet_read": False,
        "stage1_test_labels_or_predictions_read": False,
        "training_only_development_pool": True,
        "candidate_count": len(problem.candidates),
        "selected_count": problem.selected_count,
        "candidate_feasible_subsets": len(feasible),
        "exact_classical_optimum": optimum.objective,
        "exact_classical_optimum_customer_ids": optimum.customer_ids,
        "noise_model_is_illustrative_not_hardware_calibrated": True,
        "qiskit_version": __import__("qiskit").__version__,
        "results": summary,
        "limitations": [
            "The candidate pool uses repeated OOF training probabilities and is not the frozen challenge top ten.",
            "This standard XY-QAOA run uses an inequality penalty if inequalities bind; it is not IF-QAOA.",
            "No physical quantum hardware was used.",
        ],
    }
    (args.output / "xy_qaoa_development_results.json").write_text(
        json.dumps(manifest, indent=2, default=lambda value: list(value) if isinstance(value, tuple) else value),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "results"}, indent=2))
    for row in summary:
        print(json.dumps({k: row[k] for k in ("seed", "parameters", "ideal_statevector", "optimizer_success")}, default=list))


if __name__ == "__main__":
    main()
