"""Run final actual-customer Stage 3 probability, Aer noise, and mitigation study."""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib_config"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
import yaml

from atomiq.qiskit_noise import (
    IllustrativeBackendProfile,
    assignment_matrix,
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


OUT = ROOT / "results/stage3_final"
REPORTS = ROOT / "reports"
SELECTED = ROOT / "results/stage2_final/actual_selected_six.csv"
PROFILE = IllustrativeBackendProfile()
SHOTS = (512, 2048, 8192, 32768)
SEEDS = (11, 23, 42)
THRESHOLD = 3


def exact_enumeration(probabilities: np.ndarray) -> np.ndarray:
    """Independently enumerate all 64 independent Bernoulli outcomes."""
    result = np.zeros(len(probabilities) + 1, dtype=float)
    for outcome in itertools.product((0, 1), repeat=len(probabilities)):
        probability = float(np.prod([
            p if bit else 1.0 - p for p, bit in zip(probabilities, outcome, strict=True)
        ]))
        result[sum(outcome)] += probability
    return result


def wilson_interval(successes: int, shots: int, z: float = 1.959963984540054) -> list[float]:
    if shots < 1 or not 0 <= successes <= shots:
        raise ValueError("Invalid binomial counts.")
    estimate = successes / shots
    denominator = 1 + z * z / shots
    center = (estimate + z * z / (2 * shots)) / denominator
    half = z * np.sqrt(estimate * (1 - estimate) / shots + z * z / (4 * shots * shots)) / denominator
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def mitigation_bootstrap(
    counts: dict[str, int], matrices: list[np.ndarray], seed: int, replicates: int = 2000
) -> dict[str, object]:
    """Bootstrap measured bitstring frequencies through the fixed readout inverse."""
    started = perf_counter()
    n = len(matrices)
    dimension = 1 << n
    vector = np.zeros(dimension, dtype=float)
    for bitstring, count in counts.items():
        vector[int(bitstring.replace(" ", ""), 2)] += count
    shots = int(vector.sum())
    frequencies = vector / shots
    transform = np.linalg.inv(assignment_matrix(matrices))
    observed_events = np.asarray([i.bit_count() >= THRESHOLD for i in range(dimension)])
    quasi_point = transform @ frequencies
    projected_point = project_simplex(quasi_point)
    estimate = float(observed_events @ projected_point)
    quasi_estimate = float(observed_events @ quasi_point)
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(shots, frequencies, size=replicates) / shots
    quasi = samples @ transform.T
    # Readout mitigation returns a projected physical distribution; the CI
    # reflects finite count sampling and the fixed estimated assignment matrix.
    estimates = np.asarray([
        observed_events @ project for project in (project_simplex(row) for row in quasi)
    ])
    return {
        "estimate_projected": estimate,
        "estimate_quasi": quasi_estimate,
        "bootstrap_95_percentile_interval": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
        "bootstrap_standard_error": float(estimates.std(ddof=1)),
        "bootstrap_replicates": replicates,
        "bootstrap_runtime_seconds": perf_counter() - started,
        "assignment_calibration_uncertainty_in_interval": False,
    }


def project_simplex(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1
    indices = np.arange(1, len(values) + 1)
    valid = ordered - cumulative / indices > 0
    rho = np.flatnonzero(valid)[-1]
    return np.maximum(values - cumulative[rho] / (rho + 1), 0.0)


def main() -> None:
    settings = yaml.safe_load((ROOT / "configs/qiskit_circuit_noise.yaml").read_text(encoding="utf-8"))[
        "qiskit_circuit_noise"
    ]
    selected = pd.read_csv(SELECTED, dtype={"customer_id": str})
    if len(selected) != 6 or selected.customer_id.duplicated().any():
        raise ValueError("Final Stage 3 input must contain six unique Stage 2 customers.")
    ids = selected.customer_id.astype(str).tolist()
    probabilities = selected.stage1_probability.to_numpy(dtype=float)
    if len(np.unique(probabilities)) != 6:
        raise ValueError("Expected six distinct frozen Stage 1 probabilities as a mapping guard.")

    circuit = build_stage3_circuit(probabilities)
    if circuit.num_qubits != 6 or circuit.num_clbits != 6:
        raise AssertionError("Stage 3 must execute six qubits and six measured classical bits.")
    exact_dp = poisson_binomial_distribution(probabilities)
    exact_enum = exact_enumeration(probabilities)
    if not np.allclose(exact_dp, exact_enum, atol=2e-15, rtol=2e-14):
        raise ArithmeticError("Dynamic programming and independent 64-outcome enumeration disagree.")
    exact_probability = probability_at_least(probabilities, THRESHOLD)
    statevector = stage3_statevector_distribution(probabilities)
    statevector_event = probability_from_distribution(statevector, THRESHOLD)
    statevector_error = float(abs(statevector_event - exact_probability))
    if statevector_error > 1e-12:
        raise ArithmeticError("Qiskit ideal statevector disagrees with exact Poisson-binomial probability.")
    statevector_marginals = [
        float(sum(statevector[mask] for mask in range(64) if (mask >> q) & 1))
        for q in range(6)
    ]
    if not np.allclose(statevector_marginals, probabilities, atol=1e-12, rtol=0):
        raise ArithmeticError("Qiskit statevector individual marginals disagree with Stage 1 probabilities.")

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    calibration, calibration_runtime = calibration_assignment_matrices(
        6, profile=PROFILE, shots=settings["readout_calibration_shots"], seed=settings["calibration_seed"]
    )
    # Compare with the explicitly declared profile truth to quantify calibration shot error.
    calibration_reference = [PROFILE.readout_assignment(q) for q in range(6)]
    calibration_abs_error = [float(np.max(np.abs(a - b))) for a, b in zip(calibration, calibration_reference, strict=True)]
    results = []
    for shots in SHOTS:
        for seed in SEEDS:
            row = {"shots": shots, "seed": seed}
            for condition, noisy in (("ideal_finite_shot", False), ("noisy_finite_shot", True)):
                started = perf_counter()
                counts, runtime, compiled = run_aer_counts(
                    circuit.remove_final_measurements(inplace=False),
                    profile=PROFILE, shots=shots, seed=seed, noise=noisy,
                )
                observed = sum(counts.values())
                event_count = sum(count for bits, count in counts.items() if bits.replace(" ", "").count("1") >= THRESHOLD)
                estimate = event_count / observed
                entry = {
                    "estimate": estimate,
                    "absolute_error": abs(estimate - exact_probability),
                    "relative_error": abs(estimate - exact_probability) / exact_probability if exact_probability else None,
                    "event_successes": event_count,
                    "shots": observed,
                    "wilson_95_interval": wilson_interval(event_count, observed),
                    "runtime_seconds": runtime,
                    "transpiled_depth": int(compiled.depth()),
                    "transpiled_gate_counts": {str(g): int(v) for g, v in compiled.count_ops().items()},
                    "elapsed_including_summary_seconds": perf_counter() - started,
                }
                row[condition] = entry
                if noisy:
                    mitigation_started = perf_counter()
                    mitigated = mitigate_readout(counts, calibration)
                    mitigation_runtime = perf_counter() - mitigation_started
                    bootstrap = mitigation_bootstrap(counts, calibration, seed=seed + 100_000 + shots)
                    mitigated_estimate = float(probability_from_distribution(
                        np.asarray(mitigated["projected_distribution"]), THRESHOLD
                    ))
                    row["noisy_readout_mitigated"] = {
                        "estimate_projected": mitigated_estimate,
                        "estimate_quasi": float(probability_from_distribution(
                            np.asarray(mitigated["quasi_distribution"]), THRESHOLD
                        )),
                        "absolute_error": abs(mitigated_estimate - exact_probability),
                        "relative_error": abs(mitigated_estimate - exact_probability) / exact_probability if exact_probability else None,
                        "bootstrap_95_interval": bootstrap["bootstrap_95_percentile_interval"],
                        "bootstrap_standard_error": bootstrap["bootstrap_standard_error"],
                        "bootstrap_replicates": bootstrap["bootstrap_replicates"],
                        "assignment_calibration_uncertainty_in_interval": False,
                        "negative_quasi_mass": float(mitigated["negative_quasi_mass"]),
                        "condition_number": float(mitigated["condition_number"]),
                        "mitigation_runtime_seconds": mitigation_runtime,
                        "bootstrap_runtime_seconds": bootstrap["bootstrap_runtime_seconds"],
                        "bootstrap_variance": bootstrap_metric_variance(
                            counts, calibration,
                            lambda vector: probability_from_distribution(vector, THRESHOLD),
                            seed=seed + 200_000 + shots, replicates=500,
                        ),
                    }
            results.append(row)

    resources = resource_report(
        circuit.remove_final_measurements(inplace=False), PROFILE, seed=settings["evaluation_seed"]
    )
    summaries = []
    for shot_count in SHOTS:
        selected_rows = [row for row in results if row["shots"] == shot_count]
        summary = {"shots": shot_count, "independent_seeds": len(selected_rows)}
        for key, metric in (
            ("ideal_finite_shot", "estimate"),
            ("noisy_finite_shot", "estimate"),
            ("noisy_readout_mitigated", "estimate_projected"),
        ):
            values = [float(row[key][metric]) for row in selected_rows]
            summary[key] = {
                "mean_estimate": float(np.mean(values)),
                "mean_signed_bias": float(np.mean(values) - exact_probability),
                "mean_absolute_error": float(np.mean(np.abs(np.asarray(values) - exact_probability))),
                "seed_standard_deviation": float(np.std(values, ddof=1)),
            }
        summaries.append(summary)
    result = {
        "status": "completed_actual_challenge_stage3_aer_noise_study",
        "classification": [
            "exact classical Poisson-binomial dynamic program",
            "independent classical enumeration of all 64 outcomes",
            "actual Qiskit QuantumCircuit ideal Statevector",
            "actual Qiskit Aer ideal finite-shot simulation",
            "actual Qiskit Aer noisy finite-shot simulation",
            "actual Aer readout-calibrated mitigation of noisy counts",
        ],
        "data_boundary": {
            "selected_ids_from": "Stage 2 exhaustive optimum based on frozen Stage 1 feature-only predictions",
            "stage1_test_labels_requested_or_used": False,
            "probability_artifact": "results/stage1_classical/stage1_test_predictions.csv",
        },
        "selected_customers_in_qubit_order": [
            {"qubit": q, "classical_bit": q, "customer_id": key, "stage1_success_probability": float(p),
             "ry_radians": float(2 * np.arcsin(np.sqrt(p))), "statevector_marginal": statevector_marginals[q]}
            for q, (key, p) in enumerate(zip(ids, probabilities, strict=True))
        ],
        "bitstring_order": "Qiskit displayed counts are c5...c0; each measured q_i maps to c_i. Integer index i has q0 as least-significant bit.",
        "threshold": {"successes_at_least": THRESHOLD, "objective": "P(K >= 3)"},
        "exact_poisson_binomial_dp_k0_to6": exact_dp.tolist(),
        "explicit_64_outcome_enumeration_k0_to6": exact_enum.tolist(),
        "exact_probability_at_least_three": exact_probability,
        "qiskit_ideal_statevector_probability_at_least_three": statevector_event,
        "statevector_absolute_difference": statevector_error,
        "statevector_normalization": float(statevector.sum()),
        "statevector_all_64_probabilities": statevector.tolist(),
        "confidence_interval_method": "Wilson score 95% interval for raw finite-shot Bernoulli threshold events; mitigated intervals use parametric multinomial bootstrap after fixed assignment-matrix inversion and simplex projection.",
        "noise_profile": PROFILE.to_dict(),
        "calibration": {
            "method": "per-qubit prepared 0/1 readout-only Aer assignment calibration, asymmetric 2x2 matrices; no gate noise during calibration",
            "shots_per_prepared_state_per_qubit": int(settings["readout_calibration_shots"]),
            "runtime_seconds": calibration_runtime,
            "estimated_matrices_observed_given_true": [m.tolist() for m in calibration],
            "max_abs_calibration_error_against_illustrative_model": calibration_abs_error,
            "readout_mitigation_tensor_condition_number": float(np.linalg.cond(assignment_matrix(calibration))),
        },
        "circuit_resources": resources,
        "experiments": results,
        "shot_count_summaries": summaries,
        "limitations": [
            "Noise profile is illustrative and is not measured hardware calibration; no physical QPU was used.",
            "No two-qubit gates exist in this product-state circuit; no two-qubit error channel was applied to Stage 3 runs.",
            "Mitigated bootstrap intervals treat the finite-shot estimated readout assignment matrices as fixed, so do not include calibration-matrix uncertainty.",
            "The exact dynamic program is the authoritative reference; this circuit does not establish quantum advantage.",
        ],
    }
    result_path = OUT / "actual_stage3_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    make_figures(result)
    write_report(result)
    print(json.dumps({
        "exact_probability": exact_probability,
        "statevector_absolute_difference": statevector_error,
        "resource_report": resources,
        "calibration_runtime_seconds": calibration_runtime,
        "experiment_count": len(results),
        "results_path": str(result_path.relative_to(ROOT)),
        "figures": ["reports/figures/actual_stage3_probability_estimates.png", "reports/figures/actual_stage3_probability_estimates.pdf", "reports/figures/actual_stage3_absolute_error.png", "reports/figures/actual_stage3_absolute_error.pdf"],
    }, indent=2))


def make_figures(result: dict[str, object]) -> None:
    figure_dir = REPORTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = result["experiments"]
    styles = {
        "ideal_finite_shot": ("Ideal finite-shot", "#3973ac", "o"),
        "noisy_finite_shot": ("Noisy raw", "#d65f5f", "s"),
        "noisy_readout_mitigated": ("Noisy + readout mitigation", "#27836a", "^"),
    }
    exact = float(result["exact_probability_at_least_three"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    ax.axhline(exact, color="#222222", linestyle="--", linewidth=1.4, label=f"Exact classical reference ({exact:.4f})")
    offsets = {"ideal_finite_shot": -0.035, "noisy_finite_shot": 0.0, "noisy_readout_mitigated": 0.035}
    for key, (label, color, marker) in styles.items():
        xs, ys, lows, highs = [], [], [], []
        for row in records:
            item = row[key]
            estimate = item["estimate"] if key != "noisy_readout_mitigated" else item["estimate_projected"]
            interval = item["wilson_95_interval"] if key != "noisy_readout_mitigated" else item["bootstrap_95_interval"]
            xs.append(row["shots"] * (1 + offsets[key]))
            ys.append(estimate)
            lows.append(max(0.0, estimate - interval[0]))
            highs.append(max(0.0, interval[1] - estimate))
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt=marker, color=color, alpha=0.66, markersize=4, capsize=1.6, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Shots per independent run (three seeds per budget)")
    ax.set_ylabel("Estimated probability of at least three successes")
    ax.set_title("Stage 3 estimates and 95% sampling intervals")
    ax.set_xticks(SHOTS, labels=[f"{s:,}" for s in SHOTS])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"actual_stage3_probability_estimates.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for key, (label, color, marker) in styles.items():
        x_by_shots: list[int] = []
        means, stds = [], []
        for shots in SHOTS:
            errors = []
            for row in records:
                if row["shots"] != shots:
                    continue
                item = row[key]
                estimate = item["estimate"] if key != "noisy_readout_mitigated" else item["estimate_projected"]
                errors.append(abs(estimate - exact))
            x_by_shots.append(shots)
            means.append(float(np.mean(errors)))
            stds.append(float(np.std(errors, ddof=1)))
        ax.errorbar(x_by_shots, means, yerr=stds, marker=marker, color=color, capsize=3, label=f"{label}: mean ± seed SD")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Shots per independent run")
    ax.set_ylabel("Absolute error from exact reference (log scale)")
    ax.set_title("Finite-shot accuracy by condition")
    ax.set_xticks(SHOTS, labels=[f"{s:,}" for s in SHOTS])
    ax.grid(which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / f"actual_stage3_absolute_error.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def write_report(result: dict[str, object]) -> None:
    lines = [
        "# Final actual challenge results: Stages 1–3",
        "",
        "## Stage 1 freeze and inference",
        "",
        "The CatBoost native-categorical candidate selected by nested repeated CV was frozen before test-feature inference. Selection used Stage1_Train labels only; the inference loader explicitly requested customer_id and the 15 permitted predictors. Stage1_Test labels were not requested or loaded. Duration and customer_id were excluded as predictors. The selected model's nested outer-CV PR-AUC was 0.2417, ROC-AUC 0.6555, Brier score 0.1012, and log loss 0.3526. These are training validation estimates, not test-set metrics.",
        "",
        "The selected model artifact, training-partition digest, model digest, and prediction digest are in `results/stage1_classical/frozen_model.json`. The 100 inference-only scores are in `results/stage1_classical/stage1_test_predictions.csv`.",
        "",
        "## Stage 2 exact challenge solution",
        "",
        "The ten highest frozen Stage 1 probabilities were joined by customer ID to Stage2_Data. All 210 exact-six subsets were enumerated and independently checked against SciPy MILP. All 210 were feasible, so the constraints are nonbinding over this candidate pool. Both methods selected C0028, C0031, C0079, C0081, C0327, and C0338 with expected net objective 16.1710314161, contact cost 16/25, and capacity 15/22. This is the exact optimum of the stated Stage 2 objective for this top-ten pool; the probability inputs remain model estimates.",
        "",
        "Full candidates and selected customer details: `results/stage2_final/actual_top_ten.csv` and `results/stage2_final/actual_selected_six.csv`. Machine-readable independent oracle result: `results/stage2_final/actual_stage2_result.json`.",
        "",
        "The actual ten-candidate pool was also exercised with a ten-qubit exact-six XY-QAOA circuit. It uses the exact Dicke state and an additive objective phase, which is exact on this fixed-six sector because all 210 selections are feasible. Four one-seed COBYLA/SPSA × expectation/CVaR fits were capped at 10 evaluations; none certified convergence. Ideal statevector expected rewards were 10.420–11.400 against the exact 16.171 optimum. Under 2,048 illustrative noisy Aer shots, observed feasible rates were about 0.11–0.12 and the exact optimum was not sampled. Each transpiled circuit used depth 6,010, 7,089 gates, 3,081 CX gates, and 981 routing swaps. Full evidence: `results/stage2_final/actual_stage2_qaoa_results.json`.",
        "",
        "## Stage 3 probability and circuit results",
        "",
        f"The independent Poisson-binomial dynamic program and explicit enumeration of all 64 outcome strings both give P(K ≥ 3) = **{result['exact_probability_at_least_three']:.12f}** for the six selected probabilities. The Qiskit ideal statevector gives {result['qiskit_ideal_statevector_probability_at_least_three']:.12f}; absolute difference {result['statevector_absolute_difference']:.3g}. Each qubit is prepared with RY(2 asin √pᵢ), so its success marginal is pᵢ. All six qubits are measured qᵢ→cᵢ. Displayed Qiskit count keys are c5…c0; each qubit's bit is therefore at index 5−i in the string.",
        "",
        "The executed circuit uses only single-qubit RY rotations and measurements. Stage 3 therefore has no two-qubit gate errors or connectivity-induced SWAPs. No entangling operations were added. The noise model is illustrative, not hardware calibration: see the result JSON for assumed per-qubit T1/T2, asymmetric readout assignment probabilities, gate durations, single-qubit depolarization, and measurement window.",
        "",
        "Readout mitigation estimates per-qubit asymmetric assignment matrices from prepared 0/1 Aer circuits, applies tensor-product matrix inversion, then projects the quasi-distribution to the probability simplex. Raw estimates use Wilson 95% binomial intervals. Mitigated intervals use a multinomial bootstrap through the fixed estimated assignment matrix and projection. They exclude assignment-calibration uncertainty, which is reported separately. Readout mitigation corrects readout assignment errors only; it does not correct RY gate errors or thermal relaxation. The raw-vs-mitigated bias and variance results are in `results/stage3_final/actual_stage3_results.json`.",
        "",
        "In these results mitigation did not consistently reduce absolute error. Mean noisy raw estimates at 512/2,048/8,192/32,768 shots were 0.180339/0.207520/0.213053/0.214742; projected mitigated means were 0.163128/0.195841/0.198778/0.200688. Mitigation was worse at 512 and 2,048 shots; at the two largest budgets it corrected positive raw bias but still slightly increased mean absolute error. The estimated mitigation-to-raw metric variance ratio was 1.326 on average. This is an empirical trade-off, not evidence of a general mitigation benefit.",
        "",
        "![Stage 3 threshold-probability estimates and confidence intervals](figures/actual_stage3_probability_estimates.png)",
        "",
        "![Stage 3 absolute errors by shot count](figures/actual_stage3_absolute_error.png)",
        "",
        "## Interpretation and scope",
        "",
        "The exact classical dynamic program is authoritative for independent Bernoulli trials. The statevector comparison validates the circuit preparation and bit mapping. Finite-shot deviations are sampling error; noisy runs add readout, single-qubit gate, and relaxation effects; differences from the exact value can also include any model error in the Stage 1 probabilities. In this run, the mean noisy raw estimate over three seeds was 0.1803 at 512 shots versus 0.1631 after mitigation, both below the exact 0.2079; at 2,048 shots the raw mean was 0.2075 and mitigated mean 0.1958; at 8,192 and 32,768 shots raw means were 0.2131 and 0.2147, while mitigated means were 0.1988 and 0.2007. Thus mitigation reduced high-shot upward raw bias somewhat but did not consistently reduce absolute error (and was worse at the first two budgets); the estimated mitigation variance was 1.33× raw on average. This is an empirical diagnostic, not evidence that mitigation generally improves results. The experiment demonstrates correct probabilistic circuit execution and noise sensitivity, not quantum advantage. No physical QPU was used.",
        "",
        "The user-provided attached execution brief available in this workspace ends mid-sentence at its graph request (it stops after 'I want publication'); the figures and report here cover all complete requirements preceding that truncation.",
        "",
        "Independent final audit: `results/final_actual_independent_audit.json`. It re-enumerates Stage 2, re-solves the MILP, recomputes the QAOA statevector and finite-shot metrics from saved parameters/counts, checks Stage 3 dynamic programming against all 64 outcomes and the statevector weight histogram, verifies raw threshold rates and Wilson intervals, and confirms the selected six match the Stage 2 optimum. It does not open Stage1_Test or load labels.",
        "",
    ]
    (REPORTS / "final_actual_challenge_results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
