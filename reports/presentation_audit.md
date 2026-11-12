# Final presentation audit

**Scope:** presentation-only review of the completed computational milestone at `cfe9afcf72f05a730bdf9c15ce8c3556e3b27983`. No model fitting, optimization, Aer execution, test-label access, environment modification, or computational-result regeneration was performed.

## Existing deliverables reviewed

- `reports/final_actual_challenge_results.md`: authoritative narrative for final actual-customer results.
- `results/final_actual_independent_audit.json`: independent oracle, prediction, distribution, and interval audit; status passed.
- Stage 1 model summary, fold metrics, OOF predictions, calibration bins, frozen-model record, and inference manifest.
- Stage 2 exact enumeration/MILP result, selected portfolio, top ten, and actual-pool QAOA result JSON.
- Stage 3 exact Poisson-binomial result, circuit/resource/noise profile, per-run estimates, intervals, and shot-count summaries.
- Existing Stage 3 figures `actual_stage3_probability_estimates.*` and `actual_stage3_absolute_error.*`; both were preserved without edits.
- Supplied challenge PDF, visually reviewed across its four pages as a source specification. It is not a final research report.

## Coverage gaps and presentation changes

There was no final research PDF. The only existing figures were the two Stage 3 threshold-estimate/error plots. Stage 1 ROC, precision-recall, calibration, and model comparison plots were absent. Stage 2 exact-vs-QAOA, CVaR, feasibility, evaluation-budget status, and resource/noise comparisons were absent. Stage 3 probability/distribution and individual-qubit validation plots were absent.

Ten new figures were created under `reports/figures/presentation_*.png`, exclusively from saved CSV/JSON outputs:

1. Stage 1 repeated-fold model metrics, with fold SD identified as descriptive.
2. CatBoost repeat-wise OOF ROC and precision-recall curves, with repeat ranges distinguished from confidence intervals.
3. CatBoost raw/calibrated reliability diagram.
4. Stage 2 frozen top ten and exact selected six.
5. Actual-pool QAOA expected objective, feasibility, and lower-quartile CVaR by simulation condition.
6. Optimizer evaluation-budget endpoint status.
7. Synthetic four-item connectivity resource and noise-scale sensitivity comparison.
8. Stage 3 selected probabilities and exact Poisson-binomial distribution.
9. Stage 3 target-vs-statevector individual-qubit validation.
10. Stage 3 shot-budget estimates and absolute error, summarized over saved seeds.

The two existing Stage 3 figures are included in the report in their original form. No missing curve was inferred from final values. In particular, actual-pool Stage 2 optimizer iteration histories were not saved. The report therefore shows the 10/10 evaluation caps and lack of convergence certification, not a fabricated convergence trajectory.

## Independent value checks

The report generator requires the passed independent audit and checks the selected-six agreement before typesetting. Values were checked against the machine-readable files:

- Stage 1 CatBoost: mean ROC-AUC 0.655538, average precision 0.241709, Precision@10 0.188889, Recall@10 0.161616; raw/calibrated Brier 0.152333/0.101247; log loss 0.476721/0.352610; ten-bin ECE 0.183319/0.018338.
- Stage 2: all 210 exact-six subsets feasible; exact enumeration and MILP objectives agree at 16.1710314160765; selected IDs C0028, C0031, C0079, C0081, C0327, C0338.
- Actual-pool QAOA: four variants, one seed, 10 evaluations each, none convergence-certified; result metrics independently recomputed with maximum absolute difference 0. The stated circuit resource report is from the saved QAOA resource object; measured-circuit depth additionally includes measurement operations.
- Stage 3: exact `P(K >= 3)` is 0.2078515973394121; DP and 64-outcome enumeration differ by at most 5.55e-17; statevector probability agrees with the exact value; the independent audit checked 24 raw rates and Wilson intervals.
- Stage 3 mitigation: the saved three-seed means and absolute errors were carried through as recorded; the report explicitly says mitigation did not consistently improve error and reports the 1.326 variance ratio.

All training validation scores remain validation estimates. Test-label fields were not used. The local Aer noise profiles are illustrative, not backend calibration. The computational audit did not include QPU execution; the later physical runs are documented separately in the hardware audit. No quantum advantage is claimed.

## PDF inspection and hardware-evidence refresh

The report at `output/pdf/atomiq_final_research_report.pdf` was refreshed after the later Colab hardware execution. The earlier 14-page PDF claimed no physical hardware execution, which was accurate at that earlier audit but became outdated. The refreshed A4 report is 15 pages and includes a hardware-results section, Stage 3 and p=1/p=2 synthetic QAOA tables, provenance correction, and an existing hardware QAOA figure. The contents review confirmed the old no-QPU statement is removed. Pages 13–15 were rendered and visually inspected; the hardware results and correction fit without clipping or an orphan page. The refreshed PDF is 3,631,934 bytes; SHA-256: `e883ec37d1ff670919ee867cd5d7862b887a065a91ebda085bf05e98d89a50e`.

The original two Stage 3 figures and computational artifacts are preserved. The report generator uses only saved audited evidence; it did not run models, optimizers, Aer, or QPU jobs. The supplied hardware export's Stage 3 backend/layout attribution is corrected in the canonical results and explicitly flagged in the report; its raw archive remains immutable.

## Reproduction boundary

`scripts/build_presentation_figures.py` only loads the saved verified summaries/predictions and creates report figures. `scripts/build_research_report_pdf.py` only reads the result artifacts and figures and typesets the PDF. Neither script runs a model, optimizer, or quantum simulator. The environment and computational artifacts at `cfe9afc` are preserved; these presentation files are intended for a separate commit.
