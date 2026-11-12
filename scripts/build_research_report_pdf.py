"""Typeset the final, presentation-only research report from saved artifacts.

This post-processing script reads verified result files and generated report
figures. It does not fit estimators, run an optimizer, invoke Aer, or access
Stage1_Test labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "atomiq_final_research_report.pdf"
FIG = ROOT / "reports" / "figures"
INK = colors.HexColor("#172B43")
BLUE = colors.HexColor("#275D8C")
TEAL = colors.HexColor("#238A82")
RED = colors.HexColor("#C85652")
PALE = colors.HexColor("#EAF0F5")
MUTED = colors.HexColor("#5B6875")


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def read_csv(rel: str):
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check_sources():
    audit = read_json("results/final_actual_independent_audit.json")
    if audit.get("status") != "passed_independent_actual_result_audit":
        raise RuntimeError("Refusing to typeset without the passed independent audit")
    if audit.get("stage1_test_labels_requested_or_used") is not False:
        raise RuntimeError("Audit does not certify the Stage 1 test-label boundary")
    if audit["checks"]["selected_six_match_stage2_optimum"] is not True:
        raise RuntimeError("Selected six do not match the exact Stage 2 optimum")
    for name in [
        "presentation_stage1_model_metrics.png", "presentation_stage1_roc_pr.png",
        "presentation_stage1_calibration.png", "presentation_stage2_top_ten.png",
        "presentation_stage2_qaoa_comparison.png", "presentation_stage2_optimizer_status.png",
        "presentation_stage2_resources_noise.png", "presentation_stage3_probabilities_distribution.png",
        "presentation_stage3_qubit_validation.png", "presentation_stage3_shot_convergence.png",
        "actual_stage3_probability_estimates.png", "actual_stage3_absolute_error.png",
    ]:
        if not (FIG / name).is_file():
            raise FileNotFoundError(f"Missing report figure: {name}")
    return audit


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=31, textColor=INK, alignment=TA_LEFT, spaceAfter=12))
styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"], fontSize=12, leading=17, textColor=MUTED, spaceAfter=8))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=INK, spaceBefore=5, spaceAfter=9, keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.1, leading=13.2, textColor=INK, spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.6, leading=10, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name="Captionx", parent=styles["BodyText"], fontSize=7.7, leading=10, textColor=MUTED, spaceBefore=3, spaceAfter=10))
styles.add(ParagraphStyle(name="TableHeadx", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.2, leading=8.4, textColor=colors.white))
styles.add(ParagraphStyle(name="TableCellx", parent=styles["BodyText"], fontSize=7.1, leading=8.7, textColor=INK))
styles.add(ParagraphStyle(name="CoverMeta", parent=styles["BodyText"], fontSize=9, leading=14, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name="Equationx", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=10, leading=15, alignment=TA_CENTER, textColor=INK, borderColor=PALE, borderWidth=.7, borderPadding=8, backColor=colors.HexColor("#F7F9FB"), spaceBefore=4, spaceAfter=8))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def fmt(x, n=3):
    return f"{float(x):.{n}f}"


def figure(name, width, caption, number):
    path = FIG / name
    with PILImage.open(path) as source:
        ratio = source.height / source.width
    height = width * ratio
    if height > 145 * mm:
        height = 145 * mm
        width = height / ratio
    im = Image(str(path), width=width, height=height)
    return [im, P(f"<b>Figure {number}.</b> {caption}", "Captionx")]


def table(data, widths, repeatRows=1, font=7.2):
    t = Table(data, colWidths=widths, repeatRows=repeatRows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font), ("LEADING", (0, 0), (-1, -1), font + 1.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D7E0E8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def imgcell(name, width):
    return Image(str(FIG / name), width=width, height=1)


class Report(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=19*mm, rightMargin=19*mm, topMargin=19*mm, bottomMargin=17*mm, title="Hybrid Quantum-Classical Campaign Optimization: Final Research Report", author="ATOMIQ Hackathon Research Team", subject="Final report of completed Stage 1-3 computational results and IBM hardware validation")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=self.decorate)])

    def decorate(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(PALE); canvas.setLineWidth(.6)
        canvas.line(19*mm, h-13*mm, w-19*mm, h-13*mm)
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
        canvas.drawString(19*mm, h-10.5*mm, "ATOMIQ  |  FINAL RESULTS AND HARDWARE VALIDATION")
        canvas.line(19*mm, 12*mm, w-19*mm, 12*mm)
        canvas.drawString(19*mm, 8*mm, "Presentation-only research report  |  Commit cfe9afc")
        canvas.drawRightString(w-19*mm, 8*mm, f"{doc.page}")
        canvas.restoreState()


def build():
    audit = check_sources()
    stage1 = read_csv("results/stage1_classical/model_summary.csv")
    stage2 = read_json("results/stage2_final/actual_stage2_result.json")
    qaoa = read_json("results/stage2_final/actual_stage2_qaoa_results.json")
    stage3 = read_json("results/stage3_final/actual_stage3_results.json")
    summ = {r["model"]: r for r in stage1 if r["prediction_type"] == "raw"}
    cal = {r["model"]: r for r in stage1 if r["prediction_type"] == "sigmoid_calibrated"}
    selected_name = "catboost_native"
    cat = summ[selected_name]; catc = cal[selected_name]
    exact = float(stage2["exact_enumeration_optimum"]["objective"])
    p3 = float(stage3["exact_probability_at_least_three"])
    pages = Report(str(OUT))
    story = []

    # Cover
    story += [Spacer(1, 23*mm), P("ATOMIQ  /  RESEARCH REPORT", "Smallx"), Spacer(1, 7*mm),
              P("Hybrid Quantum-Classical<br/>Campaign Optimization", "TitleBig"),
              P("A leakage-controlled evaluation of response prediction, constrained selection, and probabilistic circuit execution", "Deck"),
              Spacer(1, 12*mm)]
    coverbox = Table([[P("FINAL COMPUTATIONAL + HARDWARE RESULTS", "H2x")], [P("Stages 1 to 3  |  Training-only model selection  |  Exact classical optimization  |  IBM hardware validation", "Bodyx")], [P("Prepared from the completed, independently audited computational artifacts and the subsequent Colab IBM hardware export. The publication refresh uses saved evidence only; no experiment was rerun.", "Bodyx")]], colWidths=[168*mm])
    coverbox.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#F2F6F9")), ("BOX", (0,0),(-1,-1), .8, PALE), ("LINEBEFORE", (0,0),(0,-1), 4, TEAL), ("LEFTPADDING",(0,0),(-1,-1),12), ("RIGHTPADDING",(0,0),(-1,-1),12), ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story += [coverbox, Spacer(1, 35*mm), P("Prepared 12 November 2026", "CoverMeta"), P("Reproducibility anchor: cfe9afcf72f05a730bdf9c15ce8c3556e3b27983", "CoverMeta"), P("Two IBM hardware jobs are documented; synthetic QAOA is not the bank optimization. No quantum advantage is claimed.", "CoverMeta"), PageBreak()]

    # Abstract and reading guide
    story += [P("Abstract", "H1x"), P("This report presents the completed three-stage ATOMIQ campaign pipeline. Stage 1 compares ten classical classifiers on 300 labeled training observations using repeated stratified nested cross-validation; CatBoost records the highest mean outer-fold average precision (0.2417, fold SD 0.0574), while calibration improves its Brier score and log loss. Stage 2 freezes feature-only probabilities for the 100 challenge inference rows, ranks a top-ten pool, and selects six customers by exhaustive enumeration of all 210 subsets, independently agreeing with a mixed-integer linear program at objective 16.1710. Actual-pool XY-QAOA circuits were executed under small, explicitly truncated budgets; their sampled objectives remain below the exact oracle and are diagnostic only. Stage 3 prepares the six selected Bernoulli probabilities on six measured qubits. The exact Poisson-binomial reference for at least three successes is 0.2078516, matching an independent 64-outcome enumeration and the ideal statevector. Finite-shot and noisy Aer studies show sampling variability and no consistent benefit from readout mitigation. A later Colab batch executed one Stage 3 job and one synthetic QAOA depth comparison on IBM hardware; the measured records and provenance correction are reported separately. These results do not establish quantum advantage.", "Bodyx"),
              P("Scope and evidence labels", "H2x"), P("All reported values are transcribed or summarized from saved artifacts and the passed independent audit. Figures generated for this report are deterministic post-processing of those artifacts. Stage 1 validation results are training-only estimates; the test-label boundary is preserved. Actual Qiskit circuits and Aer runs are distinguished from exact classical calculations and from a synthetic four-item resource/noise fixture.", "Bodyx"),
              P("Contents", "H2x"), P("1  Data and validation protocol  |  2  Stage 1 classification  |  3  Stage 2 exact selection and QAOA diagnostics  |  4  Stage 3 probability circuit  |  5  IBM hardware validation  |  6  Synthesis, limitations, and reproducibility  |  Appendices", "Bodyx"), PageBreak()]

    # Methods
    story += [P("1. Data, methods, and validation protocol", "H1x"), P("The challenge defines a response-prediction task, a six-of-ten operational selection problem, and a portfolio event probability. Stage 1 uses only Stage1_Train (300 rows; 35 positive, 265 negative). The participant identifier and contact duration are not predictors. No workbook-level leaked aggregate target totals are used. The final model was frozen before inference; the loader requested the 100 test customer identifiers and 15 permitted predictors without requesting their labels.", "Bodyx"),
              P("The Stage 1 protocol uses three stratified folds repeated three times, producing nine outer validation folds with 100 rows each. Two-fold inner selection optimizes average precision under a small, controlled candidate budget. Imputation, scaling, category encoding, model fitting, and positive-slope sigmoid calibration are fold-local. Ten candidates share the same outer splits. Fold standard deviations are descriptive: repeated folds overlap in their training data and are not independent confidence intervals.", "Bodyx"),
              P("Ranking is summarized by ROC-AUC and average precision (area under the precision-recall curve); operational ranking is also reported as Precision@10 and Recall@10. Probability quality is summarized by Brier score (mean squared probability error), log loss, and ten-bin expected calibration error. Model choice follows mean outer-fold average precision, with probability metrics retained as separate evidence rather than collapsed into one score.", "Bodyx"),
              P("Stage 2 objective", "H2x"), P("maximize sum_i x_i (p_i v_i - c_i), subject to sum_i x_i = 6, sum_i x_i c_i <= 25, sum_i x_i a_i <= 22, x_i in {0,1}", "Equationx"),
              P("Here p_i is the frozen Stage 1 response probability, v_i expected deposit value, c_i contact cost, and a_i operational capacity. The ten highest-probability candidates define the pool. All 210 six-element subsets are enumerated; the same objective and constraints are independently encoded in a SciPy MILP.", "Bodyx"),
              P("Stage 3 independent Bernoulli reference", "H2x"), P("For independent outcomes X_i ~ Bernoulli(p_i), K = sum_i X_i. The exact mass function is computed recursively: f_0(0)=1; after probability p_i, f_i(k) = (1 - p_i) f_(i-1)(k) + p_i f_(i-1)(k - 1). The target is P(K &gt;= 3) = sum of f_6(k) for k = 3,...,6. A second implementation enumerates all 64 binary outcomes.", "Bodyx"),
              P("For each p_i the circuit applies RY(theta_i)|0> with theta_i=2 asin(sqrt(p_i)), then measures q_i into c_i. Qiskit displayed bitstrings are c5...c0; q_i maps to c_i, so bit position 5-i in the displayed string corresponds to qubit i. The saved tests and audit verify this ordering and the six marginal probabilities.", "Bodyx"), PageBreak()]

    # Stage1
    story += [P("2. Stage 1: classical response prediction", "H1x"), P("CatBoost native categoricals is the selected reference under nested repeated validation. Its mean average precision is 0.2417 and mean ROC-AUC is 0.6555. The next average-precision candidate, linear SVM, is 0.236; this small separation is not evidence of statistically significant superiority. Linear SVM, RBF SVM, Random Forest, and regularized logistic regression achieve a larger mean Precision@10 (0.211 versus 0.189), showing that the preferred ranking metric affects the model choice.", "Bodyx")]
    story += figure("presentation_stage1_model_metrics.png", 170*mm, "Ten-model comparison. Points show means across nine outer folds; whiskers are fold standard deviations, not confidence intervals. Calibration is compared on Brier score, for which lower is better.", 1)
    story += [P("Table 1. Selected candidate validation metrics", "H2x"), table([
        ["Metric", "Raw", "Positive-slope sigmoid", "Interpretation"],
        ["ROC-AUC", fmt(cat["roc_auc_mean"],4), fmt(catc["roc_auc_mean"],4), "Ranking; monotone calibration preserves ordering"],
        ["Average precision", fmt(cat["pr_auc_mean"],4), fmt(catc["pr_auc_mean"],4), "PR-AUC; prevalence-sensitive"],
        ["Precision@10 / Recall@10", f"{fmt(cat['precision_at_10_mean'],3)} / {fmt(cat['recall_at_10_mean'],3)}", f"{fmt(catc['precision_at_10_mean'],3)} / {fmt(catc['recall_at_10_mean'],3)}", "Mean over folds"],
        ["Brier score", f"{fmt(cat['brier_score_mean'],4)} (SD {fmt(cat['brier_score_std'],3)})", f"{fmt(catc['brier_score_mean'],4)} (SD {fmt(catc['brier_score_std'],3)})", "Lower is better"],
        ["Log loss", fmt(cat["log_loss_mean"],4), fmt(catc["log_loss_mean"],4), "Lower is better"],
        ["10-bin ECE", fmt(cat["ece_10_bins_mean"],4), fmt(catc["ece_10_bins_mean"],4), "Descriptive calibration error"],
    ], [36*mm, 28*mm, 39*mm, 67*mm]), Spacer(1,4*mm), P("Calibration substantially improves the selected candidate's probability scores: mean Brier falls from 0.1523 to 0.1012, log loss from 0.4767 to 0.3526, and 10-bin ECE from 0.1833 to 0.0183. These are validation diagnostics, not proof of reliable probability estimates on a shifted inference population. Calibration remains noisy with roughly 11-12 positive cases per outer fold.", "Bodyx"), PageBreak()]
    story += [P("Stage 1: ranking and probability reliability", "H2x")]
    story += figure("presentation_stage1_roc_pr.png", 170*mm, "CatBoost positive-slope sigmoid predictions pooled by repeat. Each faint line represents one 300-row outer-OOF repeat; the shaded range is repeat-to-repeat variation, not an inferential confidence band. Fold-level summary metrics remain the reported primary estimates.", 2)
    story += figure("presentation_stage1_calibration.png", 120*mm, "Raw and calibrated CatBoost reliability bins from pooled repeated OOF records. Bin sizes are shown by marker area. This descriptive reliability diagram should be read with the small validation sample and fold-level uncertainty in mind.", 3)
    story += [P("The candidate search used average precision as its inner-fold selection score and retained both raw and calibrated outputs. The frozen final-fit parameters were iterations=140, depth=5, learning rate=0.03, L2 regularization=5, and no automatic class weighting. The expanded search did not establish a replacement winner: its CatBoost average precision was similar (0.243), while ROC-AUC, Brier, and log loss were worse and fold results were unstable. No additional fitting was performed for this report.", "Bodyx"), PageBreak()]

    # Stage2 exact
    optimum = stage2["exact_enumeration_optimum"]
    ids = optimum.get("customer_ids", optimum.get("ids", []))
    story += [P("3. Stage 2: exact selection and quantum diagnostic", "H1x"), P(f"The top-ten frozen probabilities produced {stage2.get('subsets_evaluated', 210)} candidate six-person subsets. All 210 satisfy the budget and capacity limits, so these constraints are nonbinding within this shortlist. Exhaustive enumeration and the independent MILP agree exactly on the optimum. The resulting objective is {exact:.10f}; it is exact for this ten-candidate pool and objective, but its inputs remain model-estimated probabilities.", "Bodyx")]
    story += figure("presentation_stage2_top_ten.png", 150*mm, "Frozen top ten by Stage 1 probability; green identifies the six selected by exhaustive objective maximization. The operational data are joined by customer ID.", 4)
    selected_rows = read_csv("results/stage2_final/actual_selected_six.csv")
    headers = list(selected_rows[0].keys()) if selected_rows else []
    # Use a compact, robust customer/probability/cost/capacity table.
    data = [["Customer", "p(response)", "Expected value", "Contact cost", "Capacity"]]
    for row in selected_rows:
        data.append([row.get("customer_id", ""), f"{float(row['stage1_probability']):.6f}", f"{float(row['expected_deposit_value']):.0f}", f"{float(row['contact_cost']):.0f}", f"{float(row['capacity_units']):.0f}"])
    story += [P("Table 2. Exact selected portfolio", "H2x"), table(data, [29*mm, 27*mm, 37*mm, 34*mm, 31*mm]), P("Selected IDs: " + ", ".join(ids) + ". Aggregate contact cost is 16/25 and capacity is 15/22. The exact classical solution remains the decision reference; QAOA is not presented as a substitute.", "Smallx"), PageBreak()]

    # QAOA methods/metrics
    story += [P("Actual-pool XY-QAOA: observed outcomes", "H2x"), P("The executed parameterized 10-qubit circuit uses an exact-six Dicke initial state and a depth-one cycle XY mixer. Its additive cost phase is exact on this fixed-six sector because every one of the 210 six-person subsets is feasible. Four one-seed fits compare COBYLA and SPSA, each optimizing expectation or lower-quartile CVaR, with a ten-evaluation cap. The saved endpoints are not convergence-certified. The results therefore assess a deliberately limited circuit/optimization diagnostic, not algorithmic competitiveness.", "Bodyx")]
    story += figure("presentation_stage2_qaoa_comparison.png", 174*mm, "Exact objective compared with statevector, ideal finite-shot, noisy finite-shot, and readout-mitigated results. Values are conditional on feasible outcomes except the feasibility panel; the exact optimum line is a reference, not an attainable expectation bound for a finite optimizer.", 5)
    rows = [["Optimizer / objective", "Statevector E", "Ideal Aer E", "Noisy E", "Mitigated E", "Noisy feasible", P("Exact optimum<br/>probability (noisy)", "TableHeadx")]]
    for r in qaoa["results"]:
        label = f"{r['optimizer']} / {r['training_objective']}"
        rows.append([label,
                     fmt(r["ideal_statevector_metrics"]["expected_objective_conditional_on_feasibility"],2),
                     fmt(r["ideal_finite_shot"]["metrics"]["expected_objective_conditional_on_feasibility"],2),
                     fmt(r["noisy_finite_shot"]["metrics"]["expected_objective_conditional_on_feasibility"],2),
                     fmt(r["noisy_readout_mitigated"]["metrics"]["expected_objective_conditional_on_feasibility"],2),
                     fmt(r["noisy_finite_shot"]["metrics"]["feasible_shot_probability"],3),
                     fmt(r["noisy_finite_shot"]["metrics"]["exact_optimum_sampling_probability"],4)])
    story += [P("Table 3. Actual-pool QAOA results", "H2x"), table(rows, [31*mm, 22*mm, 23*mm, 22*mm, 25*mm, 21*mm, 24*mm], font=6.7), P(f"Exact classical optimum = {exact:.4f}. In the noisy 2,048-shot distributions, the exact optimum was not observed. Readout mitigation produced modest feasibility changes; it did not close the gap to the oracle. Best feasible measured objective, objective gap, CVaR, runtime, and bitstring counts remain available in the machine-readable result.", "Bodyx"), PageBreak()]

    story += [P("Optimization budget, noise, and circuit resources", "H2x")]
    story += figure("presentation_stage2_optimizer_status.png", 154*mm, "Endpoint audit of the saved optimizer budget. All four runs used 10 of 10 allowed evaluations and none certified convergence. This is not a convergence curve; per-iteration histories were not persisted.", 6)
    prof = qaoa["ansatz"]["noise_profile"]
    res = qaoa["results"][0]["resources"]
    # Normalize resource schema from saved record by embedding authoritative known circuit values from QAOA record.
    story += [P("Circuit resource and noise profile", "H2x"), P(f"Each actual-pool circuit used 10 logical qubits mapped to 10 simulator qubits on an assumed bidirectional line. The saved transpilation reports depth 6,010, 7,089 total gates, 3,081 two-qubit CX gates, and 981 routing SWAPs before basis translation. The illustrative profile uses asymmetric readout assignment, single- and two-qubit depolarizing channels, and T1/T2 relaxation with explicit gate-duration assumptions. No backend calibration was retrieved; the record explicitly marks calibration source and timestamp as null.", "Bodyx"),
              P(f"Assumptions include T1={min(prof['t1_seconds'])*1e6:.0f}-{max(prof['t1_seconds'])*1e6:.0f} microseconds, T2={min(prof['t2_seconds'])*1e6:.0f}-{max(prof['t2_seconds'])*1e6:.0f} microseconds, one-qubit depolarization {prof['one_qubit_depolarizing']}, two-qubit depolarization {prof['two_qubit_depolarizing']}, SX {prof['sx_duration_seconds']*1e9:.0f} ns, CX {prof['cx_duration_seconds']*1e9:.0f} ns, and measurement {prof['measurement_duration_seconds']*1e9:.0f} ns. Values are illustrative simulation inputs, not hardware facts.", "Bodyx"),
              P("A separate saved four-item synthetic fixture compares path, cycle, and complete mixer connectivities and noise scales. It is presented only as a method/resource stress test: it is neither the actual ten-customer instance nor a hardware result.", "Bodyx")]
    story += figure("presentation_stage2_resources_noise.png", 168*mm, "Transpilation cost and simulated gate-noise sensitivity for the synthetic four-item binding fixture. Resource trade-offs depend on the recorded compiler version, basis, coupling map, and initial layout.", 7)
    story += [P("A per-iteration optimizer trajectory cannot be reconstructed from the final parameter vectors and endpoint records. No trajectory was generated. All optimizer results are budget-limited diagnostics; no quantum advantage is claimed.", "Bodyx"), PageBreak()]

    # Stage3
    customers = stage3["selected_customers_in_qubit_order"]
    story += [P("4. Stage 3: six-qubit probability experiment", "H1x"), P(f"The six selected response probabilities are prepared independently with single-qubit RY gates. The exact Poisson-binomial recurrence and all-64-outcome enumeration agree within {audit['checks']['dp_vs_64_outcome_enumeration_max_abs_error']:.2e}; the Qiskit ideal statevector gives P(K&gt;=3)={p3:.12f}, matching the exact value. The circuit's six ideal marginal probabilities agree with their target probabilities to floating-point precision.", "Bodyx")]
    story += figure("presentation_stage3_probabilities_distribution.png", 171*mm, "Selected Stage 1 probabilities and exact Poisson-binomial distribution of K, the number of positive responses. The event of interest sums the bars for K=3 through 6.", 8)
    ptable = [["Qubit", "Customer", "p_i", "RY angle (rad)", "Statevector marginal"]]
    for r in customers:
        ptable.append([str(r["qubit"]), r["customer_id"], fmt(r["stage1_success_probability"],6), fmt(r["ry_radians"],6), fmt(r["statevector_marginal"],6)])
    story += [P("Table 4. Circuit preparation and individual-qubit validation", "H2x"), table(ptable, [17*mm, 31*mm, 29*mm, 38*mm, 49*mm]), P("The statevector circuit contains six logical qubits, 24 transpiled one-qubit gates, depth four, and no two-qubit gates or SWAPs. The displayed count convention is c5...c0 even though circuit wiring is q_i to c_i; this explicit mapping prevents reversed bitstring interpretation.", "Bodyx")]
    story += figure("presentation_stage3_qubit_validation.png", 113*mm, "Each target p_i is compared with its Qiskit statevector measurement marginal; labels identify the corresponding customer. The diagonal is the expected relation.", 9)
    story += [P("Circuit audit", "H2x"), P("The six preparation rotations are applied independently, followed by measurements q_i to c_i. The ideal statevector marginals agree with the input p_i values to about 1e-16, and the full 64-outcome histogram agrees with the exact Poisson-binomial distribution within 5.55e-17. The circuit uses six qubits, has transpiled depth four and 24 one-qubit gates, and requires no entangling gates or routing SWAPs. This construction tests probability encoding and measurement interpretation; it does not simulate customer dependence or improve the classical estimator.", "Bodyx"), P("The explicit bit convention is important: Qiskit prints count strings as c5...c0 although qubit i is wired to classical bit i. Therefore the character at string position 5-i is the measurement of q_i. The independent audit validates each saved threshold rate and Wilson interval, not only the aggregated estimate.", "Bodyx"), PageBreak(), P("Stage 3: finite-shot behavior and mitigation", "H2x"), P("Ideal finite-shot, noisy finite-shot, and readout-mitigated estimates were saved for 512, 2,048, 8,192, and 32,768 shots, each with three independent seeds. Raw estimates use Wilson 95% intervals. Mitigated estimates invert calibrated per-qubit assignment matrices, project the quasi-distribution to the probability simplex, and use a parametric multinomial bootstrap conditional on the estimated matrix. Calibration uncertainty is not included in that interval.", "Bodyx")]
    story += figure("presentation_stage3_shot_convergence.png", 171*mm, "Three-seed mean estimates and seed standard deviations by shot budget; whiskers are standard deviations, not confidence intervals. The right panel shows absolute error relative to the exact reference.", 10)
    story += figure("actual_stage3_probability_estimates.png", 125*mm, "Saved per-seed threshold-probability estimates and their raw Wilson or mitigated bootstrap intervals. Existing source figure is reused without modification.", 11)
    story += figure("actual_stage3_absolute_error.png", 125*mm, "Saved absolute error by shot count, preserving the original computational figure. Mitigation does not consistently improve error.", 12)
    shotrows = [["Shots", "Ideal mean", "Noisy mean", "Mitigated mean", "Noisy MAE", "Mitigated MAE"]]
    for row in stage3["shot_count_summaries"]:
        shotrows.append([f"{row['shots']:,}", fmt(row["ideal_finite_shot"]["mean_estimate"],4), fmt(row["noisy_finite_shot"]["mean_estimate"],4), fmt(row["noisy_readout_mitigated"]["mean_estimate"],4), fmt(row["noisy_finite_shot"]["mean_absolute_error"],4), fmt(row["noisy_readout_mitigated"]["mean_absolute_error"],4)])
    story += [P("Table 5. Three-seed estimates by shot budget", "H2x"), table(shotrows, [24*mm, 27*mm, 27*mm, 33*mm, 27*mm, 30*mm], font=7), P("Mitigation was not consistently beneficial. At 512 and 2,048 shots, the mitigated mean absolute error exceeds the raw noisy error. At 8,192 and 32,768 shots, mitigation reverses the raw positive bias but still has larger mean absolute error. Across runs, saved bootstrap diagnostics estimate a mitigation-to-raw metric variance ratio of 1.326. These results do not support a general mitigation-improves-accuracy claim.", "Bodyx"), PageBreak()]

    # Actual IBM hardware evidence (a later Colab execution, separate from local Aer runs)
    hw_stage3 = read_csv("results/ibm_hardware/source_export_contents/processed/stage3_results.csv")
    hw_qaoa = read_csv("results/ibm_hardware/source_export_contents/processed/qaoa_hardware_depth_comparison.csv")
    usage = read_csv("results/ibm_hardware/source_export_contents/processed/qpu_usage.csv")
    stage3_raw = next(r for r in hw_stage3 if r["source"] == "hardware" and r["mitigation"] == "none")
    stage3_mit = next(r for r in hw_stage3 if r["source"] == "hardware" and r["mitigation"] == "readout")
    stage3_hw_seconds = next(float(r["actual_seconds"]) for r in usage if r["job_label"] == "job1_stage3_best_layout")
    story += [P("5. IBM Quantum hardware validation", "H1x"), P("A subsequent Colab session submitted and collected two jobs on IBM Quantum hardware. This evidence is separate from the local Qiskit Aer studies above. The Stage 3 job used the six actual selected-customer probabilities; the p=1/p=2 QAOA job used a synthetic six-item choose-three objective, not the bank's operational selection. Parameters were optimized noise-free before hardware evaluation; there was no hardware-in-loop optimization.", "Bodyx"), P("Stage 3 hardware measurement", "H2x"), P(f"Job dbqfi99abdes73fo6vig ran on ibm_marrakesh using physical qubits [87, 6, 10, 139, 120, 107]. The raw estimate was {float(stage3_raw['estimate']):.6f} from {int(stage3_raw['shots']):,} shots (95% Wilson interval {float(stage3_raw['ci_low']):.6f} to {float(stage3_raw['ci_high']):.6f}); readout-mitigated estimate was {float(stage3_mit['estimate']):.6f} (bootstrap interval {float(stage3_mit['ci_low']):.6f} to {float(stage3_mit['ci_high']):.6f}). The exact Poisson-binomial reference is {p3:.12f}. The paired absolute-error reduction interval includes zero, so mitigation is inconclusive. Reconciled QPU usage was {stage3_hw_seconds:.1f} seconds.", "Bodyx"), P("QAOA depth comparison on synthetic instance", "H2x")]
    qrows = [["Depth", "Condition", "Feasibility", "P(exact optimum)", "Approximation ratio"]]
    for depth in (1, 2):
        for kind, label in [("hardware", "IBM hardware"), ("hardware", "IBM hardware + readout mitigation")]:
            matches = [r for r in hw_qaoa if int(r["depth"]) == depth and r["source"] == kind and ((r["mitigation"] == "none") if "mitigation" not in label else (r["mitigation"] == "readout"))]
            row = matches[0]
            qrows.append([f"p={depth}", label, f"{float(row['feasibility']):.3f}", f"{float(row['optimal_probability']):.4f}", f"{float(row['approximation_ratio']):.4f}"])
    story += [table(qrows, [18*mm, 57*mm, 27*mm, 32*mm, 36*mm], font=7.0), P("The p=2 hardware approximation ratio exceeds p=1, but p=2 sampled the exact optimum less often. Readout mitigation increased feasibility, while the approximation-ratio changes were small with overlapping intervals. The exact classical optimum remains the reference; these small-budget results do not support quantum advantage or a general mitigation benefit.", "Bodyx")]
    story += figure("ibm_hardware/fig7_optimisation_quality.png", 157*mm, "Hardware and simulation quality comparison for the synthetic choose-three QAOA instance. Values are from the completed IBM job and the saved ideal/Aer records; they do not represent the customer selection task.", 13)
    story += [P("Provenance correction", "H2x"), P("The supplied export's resume logic incorrectly relabeled the earlier Stage 3 job as Kingston and attached the currently selected Kingston layout. Its job ID and raw counts are intact; the original submission record identifies Marrakesh and the qubits listed above. The canonical extracted result and notebook resume metadata correct the attribution. The immutable source export remains available with the original incorrect metadata, and the correction is recorded in `results/ibm_hardware/provenance_correction.json`. Kingston calibration values belong to the QAOA job and Kingston-based Aer simulations, not to the Marrakesh Stage 3 hardware job.", "Bodyx"), PageBreak()]

    # Discussion and appendix
    story += [P("6. Interpretation and limitations", "H1x"), P("The most defensible outcome is the combination of a training-only classical validation estimate, an exact small-pool classical selection result, and a circuit-level probability-preparation check. The Stage 2 exact optimizer is authoritative for the frozen shortlist; QAOA performance is weaker under the completed evaluation budget. Stage 3 statevector agreement validates the intended independent Bernoulli encoding, while hardware and finite-shot deviations reflect sampling and device effects. These experiments demonstrate specific circuit execution and measurement behavior; they do not establish quantum speedup or quantum advantage.", "Bodyx"),
              P("Limitations", "H2x"), P("Stage 1 has only 300 training observations and 35 positive labels. Nine repeated outer folds overlap; their standard deviations are descriptive, and the two highest average-precision means are close. The controlled inner search is a screening budget rather than exhaustive optimization. Probability calibration can shift under the final inference distribution. Stage 2's 210-subset search is exact only for its frozen top-ten pool and stated objective. The actual-pool QAOA study uses four one-seed fits, 10 objective calls each, and no convergence-certified result; detailed iteration traces were not saved. The illustrative noise model is not calibrated to a real backend and does not establish hardware realism. Stage 3 mitigation is readout-focused, adds variance, and its uncertainty omits assignment-calibration uncertainty. Independence of customer outcomes is an explicit Stage 3 modeling assumption.", "Bodyx"),
              P("Reproducibility and provenance", "H2x"), P("Computational source commit: cfe9afcf72f05a730bdf9c15ce8c3556e3b27983. Independent final audit status: passed. It checked 100 Stage 1 prediction rows without loading labels; all 210 Stage 2 subsets and MILP agreement; four QAOA metric reconstructions; Stage 3 dynamic programming against explicit enumeration and statevector; 24 raw-rate/Wilson calculations; and selected-six agreement. No experiment command was run to produce this report. Only evidence post-processing and PDF typesetting were executed; no model fit, optimizer, Aer simulation, or QPU job was rerun for the publication refresh.", "Bodyx"),
              P("References and authoritative artifacts", "H2x"), P("[1] ATOMIQ Hackathon 2026, <i>Problem Statement: Hybrid Quantum-Classical Term Deposit Campaign Pipeline</i>, participant-supplied challenge document (not redistributed in the public repository).\n[2] Final actual-customer results and execution notes, reports/final_actual_challenge_results.md.\n[3] Independent final audit, results/final_actual_independent_audit.json.\n[4] Stage 1 repeated nested-CV predictions, metrics, and configuration, results/stage1_classical/.\n[5] Stage 2 exact enumeration/MILP, results/stage2_final/actual_stage2_result.json; actual-pool circuit results, results/stage2_final/actual_stage2_qaoa_results.json.\n[6] Stage 3 exact distribution, circuit, noise, and shot records, results/stage3_final/actual_stage3_results.json.\n[7] Qiskit Aer noise assumptions and methods, reports/qiskit_noise_evaluation.md; software versions are recorded in the run artifacts and docs/environment.md. [8] IBM hardware execution audit and raw run export, reports/ibm_hardware_execution_audit.md and results/ibm_hardware/.", "Smallx"), PageBreak(),
              P("Appendix A. Audit and computational classifications", "H1x"),
              table([["Component", "Classification", "Audit / status"],
                     ["Stage 1 model selection", "Classical repeated nested CV on Stage1_Train", "No test labels; fold-local processing; descriptive validation"],
                     ["Stage 2 enumeration + MILP", "Exact classical optimization", "210 subsets; objective difference 0"],
                     ["Stage 2 QAOA", "Actual parameterized Qiskit circuit; Aer simulation", "4 variants independently recomputed; all runs hit 10-call cap"],
                     ["Stage 2 noise/resource fixture", "Actual Qiskit/Aer on synthetic 4-item case", "Diagnostic only; not final candidate pool"],
                     ["Stage 3 Poisson-binomial", "Classical exact recurrence and enumeration", "DP/enumeration discrepancy 5.55e-17"],
                     ["Stage 3 statevector", "Qiskit ideal statevector", "P(K&gt;=3) equals exact reference"],
                     ["Stage 3 finite-shot/noisy", "Qiskit Aer finite-shot simulation", "24 raw rate and Wilson checks passed"],
                     ["Physical QPU", "Two completed IBM jobs", "Stage 3: Marrakesh; synthetic QAOA: Kingston; no advantage claim"]], [34*mm, 54*mm, 80*mm], font=7.1), Spacer(1,5*mm),
              P("Authoritative numbers are the machine-readable results cited above. This PDF's figures are visualization products; they do not supersede raw result files, manifests, per-fold records, or the independent audit. Historical/provisional and interrupted experiments are not pooled with the final actual-customer benchmark.", "Bodyx"),
              P("Appendix B. Saved seed and shot coverage", "H2x")]
    qseeds = sorted({r["seed"] for r in qaoa["results"]})
    story += [P(f"Actual-pool QAOA: seed(s) {', '.join(map(str,qseeds))}; four optimizer/objective variants; evaluation cap 10; 2,048 shots per finite-shot condition. Stage 3: seeds 11, 23, and 42 at each shot budget 512, 2,048, 8,192, and 32,768. Exact classical references have no sampling seed.", "Bodyx"), P("Appendix C. Reproduction boundaries", "H2x"), P("The completed computational milestone is preserved unchanged. To reproduce the research experiments, consult the original run scripts, environment documentation, run manifests, and result JSON/CSV files; those commands are intentionally not executed as part of this presentation pass. The supplemental figure script is scripts/build_presentation_figures.py. PDF typesetting is performed by scripts/build_research_report_pdf.py using the bundled ReportLab runtime, without installing or upgrading packages.", "Bodyx")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pages.build(story)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"PDF: {OUT}")
    print(f"Bytes: {OUT.stat().st_size}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    build()
