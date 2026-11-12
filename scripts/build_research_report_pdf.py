"""Build the concise, publication-ready project report from verified saved artifacts only.

This script is presentation code. It does not fit models, tune optimizers, simulate circuits,
read Stage1_Test labels, or submit IBM jobs.
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "atomiq_final_research_report.pdf"
FIG = ROOT / "reports" / "figures"
INK = colors.HexColor("#162B46")
BLUE = colors.HexColor("#245D83")
TEAL = colors.HexColor("#16877B")
GOLD = colors.HexColor("#D28A37")
PALE = colors.HexColor("#E8EEF3")
MIST = colors.HexColor("#F4F7F9")
MUTED = colors.HexColor("#566777")
WHITE = colors.white


def read_json(rel): return json.loads((ROOT / rel).read_text(encoding="utf-8"))
def read_csv(rel):
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
def fmt(v, n=3): return f"{float(v):.{n}f}"
def esc(v): return escape(str(v)).replace("\n", "<br/>")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverEyebrow", fontName="Helvetica-Bold", fontSize=9, leading=13,
                          textColor=TEAL, spaceAfter=10, tracking=1.4))
styles.add(ParagraphStyle(name="CoverTitle", fontName="Helvetica-Bold", fontSize=30, leading=35,
                          textColor=INK, spaceAfter=12))
styles.add(ParagraphStyle(name="CoverDeck", fontName="Helvetica", fontSize=13, leading=19,
                          textColor=MUTED, spaceAfter=8))
styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=21, leading=26,
                          textColor=INK, spaceAfter=9, keepWithNext=True))
styles.add(ParagraphStyle(name="Subhead", fontName="Helvetica-Bold", fontSize=12, leading=16,
                          textColor=BLUE, spaceBefore=6, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=10, leading=14.5,
                          textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name="BodySmall", fontName="Helvetica", fontSize=8.7, leading=12.3,
                          textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="Caption", fontName="Helvetica-Oblique", fontSize=8.2,
                          leading=11, textColor=MUTED, spaceBefore=4, spaceAfter=8))
styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8.3, leading=11,
                          textColor=INK))
styles.add(ParagraphStyle(name="CellHead", fontName="Helvetica-Bold", fontSize=8.1, leading=10,
                          textColor=WHITE))
styles.add(ParagraphStyle(name="Metric", fontName="Helvetica-Bold", fontSize=18, leading=21,
                          textColor=INK, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabel", fontName="Helvetica", fontSize=8.2, leading=11,
                          textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Note", fontName="Helvetica", fontSize=9, leading=13,
                          textColor=INK, backColor=MIST, borderColor=PALE, borderWidth=.7,
                          borderPadding=8, spaceBefore=4, spaceAfter=8))


def P(text, style="Body"):
    return Paragraph(text, styles[style])

def cell(text, head=False):
    if isinstance(text, Paragraph): return text
    return P(esc(text), "CellHead" if head else "Cell")

def data_table(rows, widths, font=8.3):
    data = [[cell(v, True) for v in rows[0]]]
    data += [[cell(v) for v in row] for row in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), INK),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, MIST]),
        ("GRID", (0,0), (-1,-1), .35, PALE),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    return t

def callout(text, accent=TEAL):
    t=Table([[P(text,"BodySmall")]], colWidths=[172*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),MIST),
                           ("BOX",(0,0),(-1,-1),.6,PALE),
                           ("LINEBEFORE",(0,0),(0,-1),3,accent),
                           ("LEFTPADDING",(0,0),(-1,-1),11),
                           ("RIGHTPADDING",(0,0),(-1,-1),11),
                           ("TOPPADDING",(0,0),(-1,-1),8),
                           ("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    return t

def figure(name, width, caption, max_height=112*mm):
    path=FIG/name
    with PILImage.open(path) as im: ratio=im.height/im.width
    height=width*ratio
    if height>max_height:
        height=max_height; width=height/ratio
    image=Image(str(path),width=width,height=height,kind="proportional",hAlign="CENTER")
    return [image,P(caption,"Caption")]

class Report(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,leftMargin=19*mm,rightMargin=19*mm,
                         topMargin=19*mm,bottomMargin=18*mm,
                         title="Atomiq: Hybrid Quantum-Classical Campaign Optimization",
                         author="Atomiq project team",
                         subject="Research report of validated classical, simulated, and IBM hardware results")
        frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id="main",
                    leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="paper",frames=frame,onPage=self.decorate)])
    def decorate(self,canvas,doc):
        canvas.saveState(); w,h=A4
        if doc.page>1:
            canvas.setStrokeColor(PALE); canvas.setLineWidth(.6)
            canvas.line(19*mm,h-12*mm,w-19*mm,h-12*mm)
            canvas.setFont("Helvetica-Bold",7.5); canvas.setFillColor(BLUE)
            canvas.drawString(19*mm,h-9*mm,"ATOMIQ  /  RESEARCH REPORT")
            canvas.setStrokeColor(PALE); canvas.line(19*mm,12*mm,w-19*mm,12*mm)
            canvas.setFont("Helvetica",7.5); canvas.setFillColor(MUTED)
            canvas.drawString(19*mm,8*mm,"Validated results • no quantum-advantage claim")
            canvas.drawRightString(w-19*mm,8*mm,str(doc.page))
        canvas.restoreState()

def validate_sources():
    audit=read_json("results/final_actual_independent_audit.json")
    if audit.get("status")!="passed_independent_actual_result_audit":
        raise RuntimeError("Final actual-results audit did not pass; report generation stopped")
    if audit.get("stage1_test_labels_requested_or_used") is not False:
        raise RuntimeError("Test-label boundary was not certified")
    for f in ["presentation_stage1_model_metrics.png","presentation_stage1_calibration.png",
              "presentation_stage2_top_ten.png","presentation_stage2_qaoa_comparison.png",
              "presentation_stage3_probabilities_distribution.png",
              "presentation_stage3_qubit_validation.png","presentation_stage3_shot_convergence.png",
              "ibm_hardware/fig7_optimisation_quality.png"]:
        if not (FIG/f).is_file(): raise FileNotFoundError(FIG/f)
    return audit

def build():
    audit=validate_sources()
    stage1=read_csv("results/stage1_classical/model_summary.csv")
    stage2=read_json("results/stage2_final/actual_stage2_result.json")
    qaoa=read_json("results/stage2_final/actual_stage2_qaoa_results.json")
    stage3=read_json("results/stage3_final/actual_stage3_results.json")
    hw3=read_csv("results/ibm_hardware/source_export_contents/processed/stage3_results.csv")
    hwq=read_csv("results/ibm_hardware/source_export_contents/processed/qaoa_hardware_depth_comparison.csv")
    usage=read_csv("results/ibm_hardware/source_export_contents/processed/qpu_usage.csv")
    rawcat=next(x for x in stage1 if x["model"]=="catboost_native" and x["prediction_type"]=="raw")
    calcat=next(x for x in stage1 if x["model"]=="catboost_native" and x["prediction_type"]=="sigmoid_calibrated")
    optimum=stage2["exact_enumeration_optimum"]
    ids=optimum.get("customer_ids",optimum.get("ids",[]))
    selected=read_csv("results/stage2_final/actual_selected_six.csv")
    exact3=float(stage3["exact_probability_at_least_three"])
    h3=next(x for x in hw3 if x["source"]=="hardware" and x["mitigation"]=="none")
    h3m=next(x for x in hw3 if x["source"]=="hardware" and x["mitigation"]=="readout")
    h3sec=float(next(x["actual_seconds"] for x in usage if x["job_label"]=="job1_stage3_best_layout"))
    hqsec=float(next(x["actual_seconds"] for x in usage if x["job_label"]=="job2_stage2_synthetic_best_chain"))
    story=[]; doc=Report(OUT)

    # 1 — cover
    story += [Spacer(1,22*mm),P("AT O M I Q   /   FINAL RESEARCH SUBMISSION","CoverEyebrow"),
              HRFlowable(width="100%",thickness=2,color=TEAL,spaceAfter=13*mm),
              P("Hybrid Quantum–Classical<br/>Campaign Optimization","CoverTitle"),
              P("A leakage-controlled study of response prediction, constrained customer selection, and quantum-circuit evaluation","CoverDeck"),
              Spacer(1,11*mm),callout("<b>Final challenge answer</b><br/>Selected six: C0028 · C0031 · C0079 · C0081 · C0327 · C0338<br/>Exact independent-response reference: <b>P(K ≥ 3) = 0.2078515973394121</b>",TEAL),
              Spacer(1,7*mm)]
    metrics=Table([[P("0.2417","Metric"),P("16.1710","Metric"),P("0.207852","Metric"),P("31 s","Metric")],
                   [P("Stage 1 mean PR-AUC","MetricLabel"),P("Stage 2 exact objective","MetricLabel"),P("Stage 3 exact probability","MetricLabel"),P("IBM QPU time recorded","MetricLabel")]],
                  colWidths=[43*mm]*4,rowHeights=[15*mm,13*mm])
    metrics.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),MIST),("BOX",(0,0),(-1,-1),.7,PALE),
                                 ("INNERGRID",(0,0),(-1,-1),.5,PALE),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                 ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [metrics,Spacer(1,12*mm),P("Three stages, one connected pipeline","Subhead"),
              P("A classical model ranks customers using training data. An exact classical oracle selects six from the frozen top ten. A six-qubit circuit encodes their independent success probabilities. Separately, a synthetic QAOA instance was sampled on IBM hardware to compare circuit depths. This is an empirical implementation study; it does not demonstrate quantum advantage.","Body"),
              Spacer(1,17*mm),P("Final publication • 12 November 2026 • Computational baseline cfe9afc","BodySmall"),PageBreak()]

    # 2 — Stage 1
    story += [P("1  /  Response prediction","Section"),
              P("Data and protocol","Subhead"),
              P("The model comparison used 300 labeled Stage1_Train rows (35 positive, 265 negative), with three stratified folds repeated three times. Two-fold inner selection used average precision. Imputation, encoding, scaling, model fitting, and positive-slope sigmoid calibration were fit within each training partition. Customer ID and duration were excluded as predictors; Stage1_Test labels were not loaded or used.","Body"),
              P("CatBoost was the provisional reference by mean outer-fold average precision. Fold standard deviations are descriptive: repeated folds overlap and are not independent confidence intervals.","Body"),
              data_table([["CatBoost validation metric","Raw","Calibrated"],
                          ["ROC-AUC",fmt(rawcat["roc_auc_mean"],4),fmt(calcat["roc_auc_mean"],4)],
                          ["Average precision",fmt(rawcat["pr_auc_mean"],4),fmt(calcat["pr_auc_mean"],4)],
                          ["Precision@10 / Recall@10",f"{fmt(rawcat['precision_at_10_mean'])} / {fmt(rawcat['recall_at_10_mean'])}",f"{fmt(calcat['precision_at_10_mean'])} / {fmt(calcat['recall_at_10_mean'])}"],
                          ["Brier score / Log loss",f"{fmt(rawcat['brier_score_mean'],4)} / {fmt(rawcat['log_loss_mean'],4)}",f"{fmt(calcat['brier_score_mean'],4)} / {fmt(calcat['log_loss_mean'],4)}"],
                          ["10-bin ECE",fmt(rawcat["ece_10_bins_mean"],4),fmt(calcat["ece_10_bins_mean"],4)]],
                         [69*mm,51*mm,51*mm]),Spacer(1,4*mm)]
    story += figure("presentation_stage1_model_metrics.png",169*mm,
                    "Figure 1. Ten classical candidates across nine outer folds. Whiskers show fold standard deviation, not confidence intervals.",max_height=96*mm)
    story += [PageBreak(),P("Stage 1  /  Ranking and reliability","Section"),
              P("Average precision emphasizes the minority positive class; ROC-AUC provides a prevalence-insensitive ranking view. These curves aggregate repeated out-of-fold predictions and should be interpreted alongside fold-level variation.","Body")]
    story += figure("presentation_stage1_roc_pr.png",165*mm,
                    "Figure 2. CatBoost ROC and precision-recall performance across repeated out-of-fold predictions.",max_height=89*mm)
    story += figure("presentation_stage1_calibration.png",120*mm,
                    "Figure 3. Raw and calibrated reliability bins. Calibration reduces Brier score from 0.1523 to 0.1012 and log loss from 0.4767 to 0.3526; generalization under distribution shift is not established.",max_height=61*mm)
    story += [PageBreak()]

    # 3 — Stage 2 exact
    story += [P("2  /  Exact customer selection","Section"),
              P("The frozen inference probabilities defined the ten-customer candidate pool. All 210 six-customer subsets were enumerated and cross-checked with an independently encoded SciPy MILP. Both methods select the same portfolio. All 210 subsets in this pool satisfy budget and capacity, so those constraints are nonbinding for the final shortlist.","Body"),
              callout(f"<b>Exact objective:</b> {float(optimum['objective']):.10f}<br/><b>Budget:</b> 16 / 25 contact-cost units &nbsp;&nbsp; <b>Capacity:</b> 15 / 22 units<br/><b>Enumeration–MILP gap:</b> 0",BLUE),
              Spacer(1,4*mm)]
    story += figure("presentation_stage2_top_ten.png",163*mm,
                    "Figure 4. Frozen top ten ranked by Stage 1 probability; highlighted customers are the exact six selected using the Stage 2 objective.",max_height=83*mm)
    rows=[["Customer","p(response)","Expected value","Contact cost","Capacity"]]
    for r in selected:
        rows.append([r["customer_id"],fmt(r["stage1_probability"],6),f"{float(r['expected_deposit_value']):.0f}",f"{float(r['contact_cost']):.0f}",f"{float(r['capacity_units']):.0f}"])
    story += [P("Selected portfolio","Subhead"),data_table(rows,[30*mm,35*mm,38*mm,34*mm,34*mm]),
              Spacer(1,4*mm),P("The objective is exact conditional on the frozen Stage 1 probabilities and the supplied operational values. It does not remove uncertainty in the predicted probabilities.","BodySmall"),PageBreak()]

    # 4 — Stage 2 QAOA simulation
    story += [P("3  /  QAOA diagnostic on the actual shortlist","Section"),
              P("The actual ten-candidate pool was also evaluated with an exact-six XY-QAOA circuit. The additive cost phase is exact on this fixed-six sector because all 210 subsets are feasible. Four single-seed fits compare COBYLA/SPSA and expectation/CVaR objectives; each was capped at ten evaluations. None is convergence-certified.","Body"),
              callout("This is a real Qiskit circuit and Aer diagnostic, but parameter optimization was severely budget-limited. It is not the later IBM hardware QAOA experiment, which uses a separate synthetic instance.",GOLD),
              Spacer(1,5*mm)]
    story += figure("presentation_stage2_qaoa_comparison.png",168*mm,
                    "Figure 5. Actual-pool QAOA objective and feasibility comparisons from saved statevector and Aer results. Exact classical optimization is the reference.",max_height=105*mm)
    story += figure("presentation_stage2_optimizer_status.png",145*mm,
                    "Figure 6. Saved optimizer evaluation counts and termination status. No convergence trajectory was saved or reconstructed.",max_height=70*mm)
    story += [P("The exact optimum objective is 16.1710. The small-budget QAOA endpoints remain below that reference; sampled circuits are diagnostic and do not replace exact selection.","BodySmall"),PageBreak()]

    # 5 — Stage 3 exact and mapping
    customers=stage3["selected_customers_in_qubit_order"]
    story += [P("4  /  Stage 3 probability circuit","Section"),
              P("Assuming independent customer responses Xᵢ ~ Bernoulli(pᵢ), the exact target is P(K ≥ 3), where K = ΣXᵢ. A Poisson-binomial dynamic program and an explicit enumeration of all 64 bitstrings agree at <b>0.2078515973394121</b>. The ideal Qiskit statevector agrees to numerical precision.","Body"),
              P("For each customer, the circuit prepares RY(2 asin √pᵢ)|0⟩ and measures qᵢ into cᵢ. Qiskit prints bitstrings as c₅…c₀; therefore qᵢ is at character position 5−i. The tests explicitly validate this mapping.","BodySmall")]
    story += figure("presentation_stage3_probabilities_distribution.png",163*mm,
                    "Figure 7. The six frozen Stage 1 probabilities and their exact Poisson-binomial distribution; the target event is K ≥ 3.",max_height=82*mm)
    rows=[["Qubit","Customer","pᵢ","RY angle","Statevector pᵢ"]]
    for r in customers: rows.append([str(r["qubit"]),r["customer_id"],fmt(r["stage1_success_probability"],6),fmt(r["ry_radians"],6),fmt(r["statevector_marginal"],6)])
    story += [P("Input and individual-qubit validation","Subhead"),data_table(rows,[20*mm,32*mm,34*mm,39*mm,46*mm]),Spacer(1,4*mm)]
    story += figure("presentation_stage3_qubit_validation.png",120*mm,
                    "Figure 8. Statevector marginals match their customer-specific target probabilities.",max_height=58*mm)
    story += [P("This six-qubit product-state circuit has no entangling gates. It validates probability preparation and bit ordering; it does not model dependence between customer outcomes.","BodySmall"),PageBreak()]

    # 6 — Stage 3 finite shots and QPU
    story += [P("5  /  Sampling, noise, and Stage 3 hardware","Section"),
              P("Local simulation covered 512, 2,048, 8,192, and 32,768 shots over three seeds. Readout mitigation uses estimated assignment matrices and a projected quasi-distribution; its bootstrap intervals condition on the calibration estimate. Mitigation did not consistently reduce error and increased estimated variance.","Body"),]
    story += figure("presentation_stage3_shot_convergence.png",165*mm,
                    "Figure 9. Saved Stage 3 estimates across shot budgets. Whiskers show seed variation, not confidence intervals.",max_height=78*mm)
    story += [P("A later Colab run executed the same six-qubit Stage 3 circuit on IBM hardware. This is actual physical QPU execution, separate from the local Aer studies.","Subhead")]
    story += [data_table([["Stage 3 condition","Estimate","95% interval","Shots"],
                          ["Exact Poisson-binomial",fmt(exact3,6),"Exact","—"],
                          ["IBM Marrakesh, raw",fmt(h3["estimate"],6),f"{fmt(h3['ci_low'],6)}–{fmt(h3['ci_high'],6)}",f"{int(h3['shots']):,}"],
                          ["IBM Marrakesh, readout mitigated",fmt(h3m["estimate"],6),f"{fmt(h3m['ci_low'],6)}–{fmt(h3m['ci_high'],6)}",f"{int(h3m['shots']):,}"]],
                         [58*mm,31*mm,58*mm,24*mm]),Spacer(1,5*mm),
              callout(f"Job dbqfi99abdes73fo6vig • backend ibm_marrakesh • qubits [87, 6, 10, 139, 120, 107] • {h3sec:.1f} QPU seconds. The paired mitigation effect is inconclusive; its uncertainty interval includes zero.",TEAL),
              P("The source export's resumed registry incorrectly assigned Kingston metadata to this earlier Marrakesh job. The job ID and counts are unchanged. Canonical records use the original submission backend/layout; the immutable export and full correction rationale are retained in the repository.","BodySmall"),PageBreak()]

    # 7 — QAOA hardware
    story += [P("6  /  Synthetic QAOA hardware comparison","Section"),
              P("One IBM job evaluated fixed, noise-free-optimized exact-cardinality XY-QAOA parameters at p=1 and p=2. Each depth received 30,000 hardware shots on the same six-qubit chain, plus readout calibration circuits. The objective is a synthetic six-candidate choose-three instance, not the bank-customer selection. There was no hardware-in-loop optimization.","Body"),]
    qrows=[["Depth / condition","Feasibility","P(exact optimum)","Approx. ratio","95% ratio interval"]]
    for d in (1,2):
        for mitigation,label in [("none","Hardware raw"),("readout","Readout mitigated")]:
            r=next(x for x in hwq if int(x["depth"])==d and x["source"]=="hardware" and x["mitigation"]==mitigation)
            qrows.append([f"p={d} {label}",fmt(r["feasibility"],3),fmt(r["optimal_probability"],4),fmt(r["approximation_ratio"],4),f"{fmt(r['approximation_ratio_low'],4)}–{fmt(r['approximation_ratio_high'],4)}"])
    story += [data_table(qrows,[49*mm,26*mm,34*mm,29*mm,33*mm]),Spacer(1,3*mm)]
    story += figure("ibm_hardware/fig7_optimisation_quality.png",169*mm,
                    "Figure 9. Hardware and simulation quality for the synthetic QAOA instance. Values do not describe customer selection.",max_height=88*mm)
    story += [P("The p=2 approximation ratio is higher than p=1, while its exact-optimum sampling probability is lower. Readout correction increases measured feasibility; ratio changes are small and intervals overlap. The exact classical optimum remains authoritative. No quantum advantage or general mitigation gain is established.","Body"),
              callout(f"Job dbqg5d1abdes73fo7o10 • ibm_kingston • qubits [147, 148, 149, 150, 151, 152] • 30,000 shots per depth • {hqsec:.1f} QPU seconds. Combined usage across the two hardware jobs: {h3sec+hqsec:.1f} seconds.",BLUE),PageBreak()]

    # 8 — limitations and reproducibility
    story += [P("7  /  Reproduction and limitations","Section"),
              P("What these results support","Subhead"),
              P("The Stage 2 result is an exact optimum conditional on the frozen top ten, probabilities, objective, and constraints. The Stage 3 exact probability is valid under independent Bernoulli responses. The IBM records show that the declared circuits were submitted and returned hardware counts. The comparisons do not show a quantum speedup or quantum advantage.","Body"),
              P("Known limits","Subhead"),
              P("Stage 1 has only 300 labeled rows and 35 positive cases; repeated-fold estimates and calibration are noisy. All final-pool Stage 2 subsets are feasible, so the challenge instance does not test binding constraints. The actual-pool QAOA optimizer stopped at ten evaluations and is not convergence-certified. The hardware QAOA objective is synthetic. Stage 3 inherits uncertainty in Stage 1 probabilities and assumes independent outcomes. The original challenge PDF/workbook are not redistributed because permission is unverified.","Body"),
              P("Reproducibility","Subhead"),
              P("The local locked environment is CPython 3.12 with Qiskit 2.6.2 and Aer 0.18.2. The hardware Colab run recorded Python 3.13.18, Qiskit 2.2.3, Aer 0.17.2, and Runtime 0.43.1 (seed 2026); these are distinct environments. The final local test run passed 61 tests. The public notebook is disarmed by default and contains no execution output or API key. Hardware evidence, raw counts, calibration exports, and provenance correction are under results/ibm_hardware/.","Body"),
              data_table([["Task","Command / authoritative record"],
                          ["Install", "py -3.12 -m venv .venv; install requirements-lock.txt"],
                          ["Tests", ".venv/Scripts/python.exe -m pytest -q"],
                          ["Stage 1", "scripts/run_phase1_classical.py; see results/stage1_classical/"],
                          ["Final result audit", "scripts/audit_actual_results.py; results/final_actual_independent_audit.json"],
                          ["Hardware evidence", "reports/ibm_hardware_execution_audit.md; results/ibm_hardware/"]],
                         [43*mm,128*mm]),Spacer(1,5*mm),
              P("Report values are taken from saved artifacts and the passed independent audit. Report generation is deterministic post-processing only; it did not rerun models, optimizers, Aer, or IBM jobs.","BodySmall"),
              P("Project repository: github.com/Athleity/Atomiq-Hybrid-Quantum-Optimization • Computational baseline cfe9afc • Public sanitized submission commit 4d9c9a5","Caption")]
    OUT.parent.mkdir(parents=True,exist_ok=True); doc.build(story)
    print(f"PDF: {OUT}\nPages: {doc.page}\nBytes: {OUT.stat().st_size}\nSHA-256: {hashlib.sha256(OUT.read_bytes()).hexdigest()}")

if __name__=="__main__": build()
