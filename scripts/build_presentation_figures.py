"""Create missing report figures from committed, independently audited outputs only.

This script does not fit models, run optimizers, open Stage1_Test, or execute Aer.
The two existing Stage 3 figures are intentionally left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
BLUE = "#275D8C"
TEAL = "#238A82"
RED = "#C85652"
GOLD = "#D3942F"
MID = "#737D87"
LIGHT = "#EAF0F5"
NAVY = "#172B43"
PALETTE = [BLUE, TEAL, RED, GOLD]


def require_verified_inputs() -> None:
    audit = json.loads((ROOT / "results/final_actual_independent_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed_independent_actual_result_audit":
        raise RuntimeError("Final-results audit is not marked passed; refusing to plot.")
    if audit.get("stage1_test_labels_requested_or_used") is not False:
        raise RuntimeError("Audit records test-label usage; refusing to plot.")
    if audit.get("stage2_qaoa_distribution_metric_max_abs_error", 1) > 1e-10:
        raise RuntimeError("Persisted QAOA metrics fail the independent audit.")
    if audit.get("checks", {}).get("dp_vs_64_outcome_enumeration_max_abs_error", 1) > 1e-12:
        raise RuntimeError("Stage 3 exact references do not agree.")


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stage1_figures() -> None:
    summary = pd.read_csv(ROOT / "results/stage1_classical/model_summary.csv")
    raw = summary.loc[summary.prediction_type == "raw"].copy()
    raw = raw.sort_values("pr_auc_mean", ascending=True)
    calibrated = summary.loc[summary.prediction_type == "sigmoid_calibrated"].set_index("model")
    names = raw.model.tolist()
    labels = [name.replace("_", " ") for name in names]
    y = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=True)
    metric_specs = [
        (axes[0, 0], "roc_auc_mean", "roc_auc_std", "ROC-AUC", None),
        (axes[0, 1], "pr_auc_mean", "pr_auc_std", "Average precision (PR-AUC)", None),
        (axes[1, 0], "precision_at_10_mean", "precision_at_10_std", "Precision@10", None),
    ]
    for ax, mean_col, std_col, title, _ in metric_specs:
        for yi, row in enumerate(raw.itertuples(index=False)):
            value = float(getattr(row, mean_col))
            sd = float(getattr(row, std_col))
            color = RED if row.model == "catboost_native" else BLUE
            ax.errorbar(value, yi, xerr=sd, fmt="o", color=color, ecolor=color, capsize=2.4,
                        markersize=4.5, linewidth=1.1)
        ax.set_yticks(y, labels)
        ax.set_title(title, loc="left", color=NAVY, fontweight="bold", fontsize=10)
        ax.grid(axis="x", color=LIGHT, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7.1)
        ax.tick_params(axis="x", labelsize=8)

    ax = axes[1, 1]
    for yi, name in enumerate(names):
        raw_row = raw.loc[raw.model == name].iloc[0]
        cal_row = calibrated.loc[name]
        ax.plot([raw_row.brier_score_mean, cal_row.brier_score_mean], [yi, yi], color="#C8D2DC", linewidth=1.1, zorder=1)
        ax.scatter(raw_row.brier_score_mean, yi, color=MID, marker="o", s=23, label="Raw" if yi == 0 else None, zorder=2)
        ax.scatter(cal_row.brier_score_mean, yi, color=TEAL, marker="D", s=22, label="Calibrated" if yi == 0 else None, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title("Brier score: raw vs calibrated", loc="left", color=NAVY, fontweight="bold", fontsize=10)
    ax.set_xlabel("Lower is better")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7.1)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Stage 1: repeated nested-CV model comparison", color=NAVY, fontsize=14, fontweight="bold")
    fig.text(0.01, -0.018, "Points are means over nine outer folds; whiskers show fold standard deviations, not confidence intervals.", fontsize=8, color=MID)
    save(fig, "presentation_stage1_model_metrics.png")

    oof = pd.read_csv(ROOT / "results/stage1_classical/oof_predictions.csv")
    data = oof.loc[
        (oof.model == "catboost_native")
        & (oof.prediction_type == "sigmoid_calibrated")
    ]
    repeats = sorted(data.repeat.unique())
    if len(repeats) != 3:
        raise ValueError(f"Expected three saved Stage 1 repeats, got {repeats}.")
    fpr_grid = np.linspace(0.0, 1.0, 301)
    recall_grid = np.linspace(0.0, 1.0, 301)
    tpr_by_repeat, precision_by_repeat = [], []
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    for j, repeat in enumerate(repeats):
        part = data.loc[data.repeat == repeat]
        if part.customer_id.nunique() != 300 or len(part) != 300:
            raise ValueError(f"Repeat {repeat} is not a single 300-row OOF pass.")
        fpr, tpr, _ = roc_curve(part.y_true, part.probability)
        pr, recall, _ = precision_recall_curve(part.y_true, part.probability)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        precision_interp = np.interp(recall_grid, recall[::-1], pr[::-1])
        tpr_by_repeat.append(tpr_interp)
        precision_by_repeat.append(precision_interp)
        axes[0].plot(fpr, tpr, color=BLUE, alpha=0.25, linewidth=0.9)
        axes[1].plot(recall, pr, color=TEAL, alpha=0.25, linewidth=0.9)
    tpr_values = np.asarray(tpr_by_repeat)
    precision_values = np.asarray(precision_by_repeat)
    axes[0].fill_between(fpr_grid, tpr_values.min(axis=0), tpr_values.max(axis=0), color=BLUE, alpha=0.15)
    axes[0].plot(fpr_grid, tpr_values.mean(axis=0), color=BLUE, linewidth=2, label="Mean across 3 repeats")
    axes[0].plot([0, 1], [0, 1], color=MID, linestyle="--", linewidth=1, label="Chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    prevalence = float(data.loc[data.repeat == repeats[0], "y_true"].mean())
    axes[1].fill_between(recall_grid, precision_values.min(axis=0), precision_values.max(axis=0), color=TEAL, alpha=0.15)
    axes[1].plot(recall_grid, precision_values.mean(axis=0), color=TEAL, linewidth=2, label="Mean across 3 repeats")
    axes[1].axhline(prevalence, color=MID, linestyle="--", linewidth=1, label=f"Training prevalence ({prevalence:.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for ax in axes:
        ax.grid(color=LIGHT, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Selected CatBoost: repeat-wise out-of-fold curves", color=NAVY, fontsize=14, fontweight="bold")
    fig.text(0.02, -0.018, "Each line is one 300-row pooled outer-OOF repeat; shading is the repeat range, not a confidence interval. Reported model scores are means of nine fold metrics.", fontsize=7.8, color=MID)
    save(fig, "presentation_stage1_roc_pr.png")

    bins = pd.read_csv(ROOT / "results/stage1_classical/calibration_bins.csv")
    selected_bins = bins.loc[bins.model == "catboost_native"].copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=MID, linestyle="--", linewidth=1.2, label="Perfect calibration")
    for prediction_type, color, label, marker in (
        ("raw", RED, "Raw", "o"),
        ("sigmoid_calibrated", TEAL, "Positive-slope sigmoid", "D"),
    ):
        part = selected_bins.loc[selected_bins.prediction_type == prediction_type].copy()
        part["weighted_pred"] = part.mean_predicted * part["count"]
        part["weighted_obs"] = part.observed_positive_rate * part["count"]
        grouped = part.groupby("bin", as_index=False).agg(count=("count", "sum"), weighted_pred=("weighted_pred", "sum"), weighted_obs=("weighted_obs", "sum"))
        grouped["mean_predicted"] = grouped.weighted_pred / grouped["count"]
        grouped["observed"] = grouped.weighted_obs / grouped["count"]
        sizes = 24 + 76 * grouped["count"] / grouped["count"].max()
        ax.scatter(grouped.mean_predicted, grouped.observed, s=sizes, color=color, marker=marker, edgecolor="white", linewidth=0.7, label=label)
        ax.plot(grouped.mean_predicted, grouped.observed, color=color, alpha=0.65, linewidth=1)
    ax.set(xlim=(0, 0.65), ylim=(0, 0.65), xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="CatBoost out-of-fold reliability bins")
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.02, -0.02, "Bins pool repeated outer-OOF records (each training row contributes once per repeat); point size reflects bin count. Reliability is descriptive, with small-sample uncertainty.", fontsize=7.6, color=MID)
    save(fig, "presentation_stage1_calibration.png")


def stage2_figures() -> None:
    top = pd.read_csv(ROOT / "results/stage2_final/actual_top_ten.csv")
    selected = pd.read_csv(ROOT / "results/stage2_final/actual_selected_six.csv")
    final = json.loads((ROOT / "results/stage2_final/actual_stage2_result.json").read_text(encoding="utf-8"))
    selected_ids = set(final["exact_enumeration_optimum"]["customer_ids"])
    fig, ax = plt.subplots(figsize=(9.8, 5.1), constrained_layout=True)
    top = top.sort_values("rank", ascending=False)
    colors = [TEAL if cid in selected_ids else "#B7C2CC" for cid in top.customer_id]
    ax.barh(top.customer_id, top.probability, color=colors, height=0.68)
    for y, row in enumerate(top.itertuples(index=False)):
        selected_label = "selected" if row.customer_id in selected_ids else "not selected"
        ax.text(row.probability + 0.006, y, f"{row.probability:.3f}  |  {selected_label}", va="center", fontsize=8.2, color=NAVY)
    ax.set_xlim(0, max(top.probability) * 1.38)
    ax.set_xlabel("Frozen Stage 1 probability")
    ax.set_title("Frozen top ten and exact Stage 2 selection", loc="left", color=NAVY, fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.02, -0.02, "Green marks the six-customer exhaustive optimum. All 210 fixed-six subsets satisfy the stated constraints in this candidate pool.", fontsize=8, color=MID)
    save(fig, "presentation_stage2_top_ten.png")

    qaoa = json.loads((ROOT / "results/stage2_final/actual_stage2_qaoa_results.json").read_text(encoding="utf-8"))
    runs = qaoa["results"]
    labels = [f"{r['optimizer']}\n{r['training_objective'].upper()}" for r in runs]
    conditions = [
        ("ideal_statevector_metrics", "Statevector", "expected_objective_conditional_on_feasibility", "cvar_objective_lower_quartile_conditional_on_feasibility", "feasible_shot_probability"),
        ("ideal_finite_shot", "Ideal Aer", "expected_objective_conditional_on_feasibility", "cvar_objective_lower_quartile_conditional_on_feasibility", "feasible_shot_probability"),
        ("noisy_finite_shot", "Noisy Aer", "expected_objective_conditional_on_feasibility", "cvar_objective_lower_quartile_conditional_on_feasibility", "feasible_shot_probability"),
        ("noisy_readout_mitigated", "Mitigated", "expected_objective_conditional_on_feasibility", "cvar_objective_lower_quartile_conditional_on_feasibility", "feasible_shot_probability"),
    ]
    exact = float(qaoa["exact_classical_optimum"]["objective"])
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.1), constrained_layout=True)
    x = np.arange(len(runs))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(conditions))
    for (key, label, reward_key, cvar_key, feasible_key), offset, color in zip(conditions, offsets, PALETTE, strict=True):
        rewards, feasibilities, cvars = [], [], []
        for r in runs:
            obj = r[key]["metrics"] if key in {"ideal_finite_shot", "noisy_finite_shot", "noisy_readout_mitigated"} else r[key]
            rewards.append(obj[reward_key])
            feasibilities.append(obj[feasible_key])
            cvars.append(obj[cvar_key])
        axes[0].bar(x + offset, rewards, width, color=color, label=label)
        axes[1].bar(x + offset, feasibilities, width, color=color, label=label)
        axes[2].bar(x + offset, cvars, width, color=color, label=label)
    axes[0].axhline(exact, color=NAVY, linestyle="--", linewidth=1.2, label="Exact classical optimum")
    axes[0].set_title("Expected feasible objective", color=NAVY, fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Objective value")
    axes[1].set_title("Feasible-shot probability", color=NAVY, fontweight="bold", fontsize=10)
    axes[1].set_ylim(0, 1.08)
    axes[2].axhline(exact, color=NAVY, linestyle="--", linewidth=1.2, label="Exact optimum")
    axes[2].set_title("Lower-quartile CVaR objective", color=NAVY, fontweight="bold", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7.2)
        ax.grid(axis="y", color=LIGHT, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=6.4, loc="best")
    fig.suptitle("Actual-pool XY-QAOA: exact and sampled conditions", color=NAVY, fontsize=14, fontweight="bold")
    fig.text(0.01, -0.025, "Four one-seed fits, 10 optimizer evaluations each; finite-shot conditions use 2,048 shots. Noise is illustrative, not hardware calibration.", fontsize=8, color=MID)
    save(fig, "presentation_stage2_qaoa_comparison.png")

    fig, ax = plt.subplots(figsize=(8.9, 4.8), constrained_layout=True)
    y = np.arange(len(runs))
    calls = [int(r["actual_evaluations"]) for r in runs]
    ax.barh(y, calls, color=[RED if not r["convergence_certified"] else TEAL for r in runs], height=0.55)
    for yi, r in enumerate(runs):
        ax.text(r["actual_evaluations"] + 0.15, yi, f"{r['actual_evaluations']}/{r['max_evaluations']} calls; convergence certified = {str(r['convergence_certified']).lower()}", va="center", fontsize=8.2, color=NAVY)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 16)
    ax.set_xlabel("Objective evaluations")
    ax.set_title("Optimizer endpoint audit (not a convergence trace)", loc="left", color=NAVY, fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.02, -0.02, "The saved actual-pool result contains final parameters and evaluation counts, but no per-iteration objective history. All four reached their cap; no trajectory can be reconstructed.", fontsize=7.8, color=MID)
    save(fig, "presentation_stage2_optimizer_status.png")

    noise = json.loads((ROOT / "results/stage2_qiskit_noise/qiskit_stage2_noise_results.json").read_text(encoding="utf-8"))
    resources = noise["xy_mixer_connectivity_resource_comparison"]
    sensitivity = noise["noise_strength_sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)
    topo = [item["topology"] for item in resources]
    depth = [item["resource_report"]["transpiled_depth"] for item in resources]
    cx = [item["resource_report"]["transpiled_two_qubit_gates"] for item in resources]
    swaps = [item["resource_report"]["routing_swap_count_before_basis_translation"] for item in resources]
    xpos = np.arange(len(topo))
    axes[0].bar(xpos - 0.18, depth, 0.36, color=BLUE, label="Depth")
    ax2 = axes[0].twinx()
    ax2.bar(xpos + 0.18, cx, 0.36, color=TEAL, label="CX count")
    for i, value in enumerate(swaps):
        axes[0].text(i, depth[i] + 14, f"{value} SWAPs", ha="center", fontsize=7.4, color=NAVY)
    axes[0].set_xticks(xpos, topo)
    axes[0].set_ylabel("Transpiled depth")
    ax2.set_ylabel("Two-qubit CX gates")
    axes[0].set_title("Mixer topology resources", loc="left", color=NAVY, fontweight="bold")
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=TEAL)], ["Depth", "CX gates"], frameon=False, fontsize=8, loc="upper left")
    axes[0].text(0.02, 0.02, "Four-item synthetic fixture", transform=axes[0].transAxes, fontsize=7.5, color=MID)
    scales = [x["gate_noise_scale"] for x in sensitivity]
    feasible = [x["metrics"]["feasible_probability_estimate"] for x in sensitivity]
    cond_reward = [x["metrics"]["expected_objective_conditional_on_feasible"] for x in sensitivity]
    axes[1].plot(scales, feasible, color=RED, marker="o", linewidth=2, label="Feasible-shot probability")
    ax3 = axes[1].twinx()
    ax3.plot(scales, cond_reward, color=GOLD, marker="D", linewidth=2, label="Conditional feasible objective")
    axes[1].set(xlabel="Gate-noise scale", ylabel="Feasible-shot probability", ylim=(0, 0.35), title="Illustrative noise sensitivity")
    ax3.set_ylabel("Objective given feasible")
    axes[1].text(0.02, 0.02, "Four-item synthetic fixture; one seed; 2,048 shots", transform=axes[1].transAxes, fontsize=7.5, color=MID)
    axes[1].legend([plt.Line2D([0], [0], color=RED, marker="o"), plt.Line2D([0], [0], color=GOLD, marker="D")], ["Feasibility", "Conditional objective"], frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Stage 2 circuit cost and noise stress test", color=NAVY, fontsize=14, fontweight="bold")
    fig.text(0.02, -0.02, "Connectivity and noise-scale results are from the saved synthetic binding fixture, not the final challenge candidate pool.", fontsize=8, color=MID)
    save(fig, "presentation_stage2_resources_noise.png")


def stage3_figures() -> None:
    result = json.loads((ROOT / "results/stage3_final/actual_stage3_results.json").read_text(encoding="utf-8"))
    selected = result["selected_customers_in_qubit_order"]
    names = [row["customer_id"] for row in selected]
    probs = np.asarray([row["stage1_success_probability"] for row in selected])
    marginals = np.asarray([row["statevector_marginal"] for row in selected])
    dist = np.asarray(result["exact_poisson_binomial_dp_k0_to6"])
    exact = float(result["exact_probability_at_least_three"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
    bar_colors = [TEAL if p >= 0.3 else BLUE for p in probs]
    axes[0].bar(names, probs, color=bar_colors, width=0.68)
    for i, p in enumerate(probs):
        axes[0].text(i, p + 0.012, f"{p:.3f}", ha="center", fontsize=8, color=NAVY)
    axes[0].set_ylim(0, 0.62)
    axes[0].set_ylabel("Success probability")
    axes[0].set_title("Six frozen Stage 1 probabilities", loc="left", color=NAVY, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(np.arange(7), dist, color=[MID if k < 3 else TEAL for k in range(7)], width=0.7)
    axes[1].set_xticks(np.arange(7), [f"{k}" for k in range(7)])
    axes[1].set_xlabel("Number of successes, K")
    axes[1].set_ylabel("Poisson-binomial probability")
    axes[1].set_title("Exact independent-response distribution", loc="left", color=NAVY, fontweight="bold")
    axes[1].text(0.96, 0.93, f"P(K >= 3) = {exact:.4f}", transform=axes[1].transAxes, ha="right", va="top", bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": TEAL}, color=NAVY, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color=LIGHT, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Stage 3 selected portfolio and exact probability", color=NAVY, fontsize=14, fontweight="bold")
    save(fig, "presentation_stage3_probabilities_distribution.png")

    max_abs = float(np.max(np.abs(probs - marginals)))
    fig, ax = plt.subplots(figsize=(6.6, 5.7), constrained_layout=True)
    ax.plot([0, 0.55], [0, 0.55], color=MID, linestyle="--", linewidth=1.3, label="Identity")
    ax.scatter(probs, marginals, s=52, color=TEAL, edgecolor="white", linewidth=0.7, zorder=3)
    offsets = {"C0028": (-6, 7), "C0031": (7, -10), "C0079": (7, 4), "C0081": (7, 7), "C0327": (-42, -16), "C0338": (7, -20)}
    for x, y, name in zip(probs, marginals, names, strict=True):
        ax.annotate(name, (x, y), xytext=offsets[name], textcoords="offset points", fontsize=7.7, color=NAVY)
    ax.set(xlim=(0, 0.55), ylim=(0, 0.55), xlabel="Stage 1 probability supplied to qubit", ylabel="Ideal statevector measurement marginal", title="Individual-qubit probability validation")
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.04, 0.84, f"Max |p - marginal| = {max_abs:.2e}\n6 distinct p values; q_i -> c_i", transform=ax.transAxes, fontsize=8, color=NAVY, bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": LIGHT})
    save(fig, "presentation_stage3_qubit_validation.png")

    experiments = result["experiments"]
    shot_levels = sorted({int(row["shots"]) for row in experiments})
    series = [
        ("ideal_finite_shot", "estimate", "Ideal Aer", BLUE),
        ("noisy_finite_shot", "estimate", "Noisy Aer", RED),
        ("noisy_readout_mitigated", "estimate_projected", "Noisy + readout mitigation", TEAL),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), constrained_layout=True)
    for key, estimate_key, label, color in series:
        means, sds, mean_abs_errors, abs_sds = [], [], [], []
        for shots in shot_levels:
            subset = [r for r in experiments if int(r["shots"]) == shots]
            values = [float(r[key][estimate_key]) for r in subset]
            errors = [abs(v - exact) for v in values]
            means.append(float(np.mean(values)))
            sds.append(float(np.std(values, ddof=1)))
            mean_abs_errors.append(float(np.mean(errors)))
            abs_sds.append(float(np.std(errors, ddof=1)))
        axes[0].errorbar(shot_levels, means, yerr=sds, marker="o", color=color, linewidth=1.8, capsize=3, label=label)
        axes[1].errorbar(shot_levels, mean_abs_errors, yerr=abs_sds, marker="o", color=color, linewidth=1.8, capsize=3, label=label)
    axes[0].axhline(exact, color=NAVY, linestyle="--", linewidth=1.2, label=f"Exact ({exact:.4f})")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(shot_levels, [f"{n:,}" for n in shot_levels])
    axes[0].set(xlabel="Shots per run", ylabel="Mean estimated P(K >= 3)", title="Estimate by shot budget")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(shot_levels, [f"{n:,}" for n in shot_levels])
    axes[1].set(xlabel="Shots per run", ylabel="Mean absolute error (log scale)", title="Error from exact reference")
    for ax in axes:
        ax.grid(which="both", color=LIGHT, linewidth=0.75)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Stage 3 finite-shot convergence and noise comparison", color=NAVY, fontsize=14, fontweight="bold")
    fig.text(0.01, -0.02, "Markers show three-seed means; whiskers show seed standard deviation, not confidence intervals. Raw per-run Wilson and mitigated bootstrap intervals are in the existing Stage 3 figure and results JSON.", fontsize=7.7, color=MID)
    save(fig, "presentation_stage3_shot_convergence.png")


def main() -> None:
    require_verified_inputs()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })
    stage1_figures()
    stage2_figures()
    stage3_figures()
    print("Created missing presentation figures from saved results; no experiment was run.")
    for path in sorted(FIGURES.glob("presentation_*.png")):
        print(f"{path.relative_to(ROOT)} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
