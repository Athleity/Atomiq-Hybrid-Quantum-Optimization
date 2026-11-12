"""Run a six-qubit Bernoulli Stage 3 circuit from the provisional training-only Stage 2 selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from atomiq.qiskit_noise import (
    IllustrativeBackendProfile,
    bootstrap_metric_variance,
    calibration_assignment_matrices,
    mitigate_readout,
    resource_report,
    run_aer_counts,
)
from atomiq.stage3_quantum import (
    build_stage3_circuit,
    poisson_binomial_distribution,
    probability_at_least,
    probability_from_counts,
    probability_from_distribution,
    stage3_statevector_distribution,
    successes_from_counts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qiskit_circuit_noise.yaml"))
    parser.add_argument("--selection", type=Path, default=Path("results/stage2_development/development_exact_selection.csv"))
    parser.add_argument("--oracle", type=Path, default=Path("results/stage2_development/exact_oracle_result.json"))
    parser.add_argument("--output", type=Path, default=Path("results/stage3_qiskit_noise"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["qiskit_circuit_noise"]
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    selection = pd.read_csv(args.selection, dtype={"customer_id": str})
    ids = selection.customer_id.astype(str).tolist()
    probabilities = selection.probability.to_numpy(dtype=float)
    if ids != oracle["selected_customer_ids"]:
        raise ValueError("Stage 3 selection IDs do not match the recorded Stage 2 oracle artifact.")
    profile = IllustrativeBackendProfile()
    circuit = build_stage3_circuit(probabilities)
    exact_distribution = poisson_binomial_distribution(probabilities)
    exact_probability = probability_at_least(probabilities, threshold=3)
    statevector_distribution = stage3_statevector_distribution(probabilities)
    statevector_probability = probability_from_distribution(statevector_distribution, threshold=3)
    if abs(statevector_probability - exact_probability) > 1e-12:
        raise ArithmeticError("Qiskit Stage 3 statevector disagrees with exact Poisson-binomial reference.")

    calibration, calibration_runtime = calibration_assignment_matrices(
        6, profile=profile, shots=settings["readout_calibration_shots"],
        seed=settings["calibration_seed"],
    )
    conditions = []
    seed_shot_pairs = [(seed, settings["evaluation_shots"]) for seed in settings["stage3_seeds"]]
    seed_shot_pairs += [(settings["stage3_seeds"][0], shots) for shots in settings["stage3_shot_sweep"]]
    for seed, shot_count in seed_shot_pairs:
        ideal_counts, ideal_runtime, ideal_compiled = run_aer_counts(
            circuit, profile=profile, shots=shot_count, seed=seed, noise=False
        )
        noisy_counts, noisy_runtime, noisy_compiled = run_aer_counts(
            circuit, profile=profile, shots=shot_count, seed=seed, noise=True
        )
        # Independent decoder and bitstring validation operate on every observed key.
        ideal_weight_counts = successes_from_counts(ideal_counts)
        noisy_weight_counts = successes_from_counts(noisy_counts)
        noisy_probability_raw = probability_from_counts(noisy_counts, threshold=3)
        mitigation_started = perf_counter()
        mitigated = mitigate_readout(noisy_counts, calibration)
        mitigation_runtime = perf_counter() - mitigation_started
        p_mitigated = probability_from_distribution(
            mitigated["projected_distribution"], threshold=3
        )
        variance = bootstrap_metric_variance(
            noisy_counts, calibration,
            lambda vector: probability_from_distribution(vector, threshold=3),
            seed=seed + 1000, replicates=300,
        )
        shots = int(sum(noisy_counts.values()))
        conditions.append({
            "seed": seed,
            "shot_budget": shot_count,
            "shots": shots,
            "ideal_finite_shot": {
                "probability_at_least_three": probability_from_counts(ideal_counts, threshold=3),
                "success_count_histogram_k_0_to_6": ideal_weight_counts.tolist(),
                "runtime_seconds": ideal_runtime,
                "transpiled_depth": int(ideal_compiled.depth()),
            },
            "noisy_finite_shot": {
                "probability_at_least_three": noisy_probability_raw,
                "binomial_standard_error": (noisy_probability_raw * (1 - noisy_probability_raw) / shots) ** 0.5,
                "success_count_histogram_k_0_to_6": noisy_weight_counts.tolist(),
                "runtime_seconds": noisy_runtime,
                "transpiled_depth": int(noisy_compiled.depth()),
            },
            "noisy_readout_mitigated": {
                "probability_at_least_three_projected": p_mitigated,
                "probability_at_least_three_quasi": probability_from_distribution(
                    mitigated["quasi_distribution"], threshold=3
                ),
                "quasi_distribution": np.asarray(mitigated["quasi_distribution"]).tolist(),
                "projected_distribution": np.asarray(mitigated["projected_distribution"]).tolist(),
                "negative_quasi_mass": mitigated["negative_quasi_mass"],
                "assignment_condition_number": mitigated["condition_number"],
                "runtime_seconds": mitigation_runtime,
                "bootstrap_variance": variance,
            },
            "raw_counts": noisy_counts,
        })

    resources = resource_report(circuit.remove_final_measurements(inplace=False), profile, seed=settings["evaluation_seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "completed_six_qubit_qiskit_stage3_noise_diagnostic",
        "qiskit_version": __import__("qiskit").__version__,
        "qiskit_aer_version": __import__("qiskit_aer").__version__,
        "classification": ["actual Qiskit QuantumCircuit", "ideal statevector", "ideal finite-shot Aer", "noisy finite-shot Aer", "readout-mitigated Aer counts"],
        "data_boundary": {
            "selection_artifact": str(args.selection),
            "candidate_source": oracle["candidate_source"],
            "stage1_frozen": oracle["stage1_frozen"],
            "stage1_test_sheet_read": oracle["stage1_test_sheet_read"],
            "stage1_test_labels_or_predictions_read": oracle["stage1_test_labels_or_predictions_read"],
            "interpretation": "provisional selection from training-only calibrated OOF probabilities; not final challenge customers",
        },
        "selected_customers": [
            {"customer_id": customer_id, "success_probability": float(probability)}
            for customer_id, probability in zip(ids, probabilities, strict=True)
        ],
        "success_threshold": 3,
        "exact_poisson_binomial_distribution_k_0_to_6": exact_distribution.tolist(),
        "exact_probability_at_least_three": exact_probability,
        "qiskit_statevector_probability_at_least_three": statevector_probability,
        "probability_absolute_difference": abs(statevector_probability - exact_probability),
        "circuit_resource_report": resources,
        "profile": profile.to_dict(),
        "calibration": {
            "method": "Aer readout-only prepared 0/1 calibration circuits on every physical qubit; gate errors disabled during assignment fit",
            "shots_per_state_per_qubit": settings["readout_calibration_shots"],
            "runtime_seconds": calibration_runtime,
            "matrices_observed_given_true": [matrix.tolist() for matrix in calibration],
        },
        "configuration": settings,
        "seed_level_results": conditions,
        "limitations": [
            "Backend profile is illustrative and not retrieved calibration; no physical QPU was used.",
            "The probabilities are training-only OOF values in a provisional Stage 2 selection, not probabilities from a frozen final Stage 1 model.",
            "Readout mitigation corrects local assignment errors only; it does not mitigate gate noise, coherent error, crosstalk, or probability-model uncertainty.",
        ],
    }
    path = args.output / "qiskit_stage3_noise_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "seed_level_results"}, indent=2))
    for row in conditions:
        print(json.dumps({
            "seed": row["seed"], "shots": row["shots"],
            "exact_probability": exact_probability,
            "ideal_finite": row["ideal_finite_shot"]["probability_at_least_three"],
            "noisy": row["noisy_finite_shot"]["probability_at_least_three"],
            "mitigated": row["noisy_readout_mitigated"]["probability_at_least_three_projected"],
        }))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
