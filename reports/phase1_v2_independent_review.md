# Claude Stage 1 V2 review and expanded tuning

Date: 2026-11-11. All experiments in this report use the canonical `Stage1_Train` sheet only (300 rows, 35 positives). No test predictions were produced.

## Claude V2 package review

The archive contains a feature-engineering/model portfolio prototype, repeated-fold prediction pickles, ensemble and paired-comparison scripts, leakage demonstrations, a final-fit script, a report, and eight prototype tests. Its stored 9 outer validation index sets match the canonical `results/stage1_classical/fold_assignments.csv` exactly. The source's 17-model equal-weight soft vote reports fold-mean AP 0.2725, Precision@10 0.2222, ROC-AUC 0.6421, Brier 0.1007, and log loss 0.3509. Compared with the stored CatBoost baseline, its reported paired customer-bootstrap AP difference is +0.070 with 95% percentile interval [-0.009, 0.155]. This interval includes zero and the bootstrap omits fitting and model-selection uncertainty; it does not establish an improvement.

Those V2 numbers are **not independently reproduced model fits**. The archive's scripts hardcode `/home/claude/proj` data and result paths, its README is empty, and its archived artifacts omit the fold-assignment file. The fold match above was independently recovered from the archived prediction pickle and checked by customer ID. The V2 report also acknowledges that the candidate pool was assembled after seeing base-model results, increasing selection optimism. Its ensemble `stack_hgb_meta` uses in-sample meta-model outputs on the training side to fit its calibrator, so that method is not a reliable nested calibration estimate. The reported equal-weight soft vote does not use that meta-model, but remains exploratory because of pool adaptation. The bundled tests include a repeated-fit assertion that does not actually perturb validation labels; it is not evidence of validation-label independence. The canonical runner's fold-local preprocessing and isolated outer-validation interface remain authoritative.

The V2 report's final-fit single cross-fit AP and nested ensemble AP are different estimands. The final fit uses a single 5-fold, 2-repeat cross-fit over all 300 rows; nested outer evaluation trains each base model on roughly 200 rows, fits ensemble decisions within those outer-training rows, and evaluates on a separate 100-row fold. Training size, random partitions, ensemble fitting, and aggregation differ. The nested outer estimate is the appropriate quantity for model selection; the single cross-fit is a final-fit diagnostic, not a competing unbiased estimate.

## Expanded parameter experiment

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase1_classical.py --config work\expanded_tuning.yaml --output results\stage1_classical_expanded_tuning_20261111 --models catboost_native lightgbm
.\.venv\Scripts\python.exe scripts\audit_phase1_results.py --results results\stage1_classical_expanded_tuning_20261111 --output results\stage1_classical_expanded_tuning_20261111\independent_audit.json
```

This used the existing seed 42, exact existing 3-fold x 3-repeat outer assignments, two inner folds, average-precision tuning, and eight sampled candidates per model per outer fold (up from two). That produced 72 inner candidate evaluations for each model across nine outer folds (144 total), plus final-fit tuning. All preprocessing and calibration remained fold-local. The complete configuration, seeds, trial scores, fold metrics, raw and calibrated OOF predictions, calibration diagnostics, model artifact, and logs are in `results/stage1_classical_expanded_tuning_20261111/`.

| Model/configuration | AP mean (fold SD) | ROC-AUC mean | Precision@10 | Recall@10 | Brier | Log loss | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original CatBoost (2 candidates) | 0.2417 (0.0574) | 0.6555 | 0.1889 | 0.1616 | 0.1012 | 0.3526 | 0.0183 |
| Expanded CatBoost (8 candidates) | 0.2432 (0.0687) | 0.6145 | 0.2444 | 0.2104 | 0.1022 | 0.3566 | 0.0212 |
| Original LightGBM (2 candidates) | 0.2205 (0.0759) | 0.5897 | 0.1778 | 0.1532 | 0.1025 | 0.3581 | 0.0237 |
| Expanded LightGBM (8 candidates) | 0.2208 (0.0642) | 0.5986 | 0.1889 | 0.1625 | 0.1022 | 0.3562 | 0.0193 |

Expanded CatBoost increases top-10 precision by about 5.6 percentage points across fold means and improves AP by only 0.0015, while ROC-AUC falls by 0.041 and Brier/log loss worsen slightly. Fold-wise AP changes are mixed (four of nine positive); this is not a stable all-metric improvement. Expanded LightGBM changes AP by only 0.0003 and has no meaningful top-10 gain. The added search therefore did not establish a new winner. Keep the original CatBoost as provisional pending broader independent comparisons; retain the expanded run as a diagnostic candidate. Do not select a winner from the apparent top-10 gain alone.

Independent audit passed: 18 model-fold prediction groups, 36 metric rows, maximum absolute metric recomputation error `2.78e-17`, zero top-10 membership disagreements. Audit read training data only. Existing pytest suite passed: 11 tests.

## Limitations and next work

- The expanded candidate budget is still modest and the validation set is small. Fold SDs are descriptive; the nine folds overlap through repeated CV.
- The V2 soft-vote implementation has not been ported as a canonical estimator or re-fit independently. It should not replace the baseline based on archived metrics.
- TabPFN, TabICL, XGBoost, EBM/GAM, and feature-engineering ablations remain unrun. The active environment has CatBoost, LightGBM, and Qiskit/Aer; `xgboost`, `tabpfn`, `tabicl`, and `interpret` are absent. No package installs were attempted.
- Quantum Stage 1 and Stages 2–3 remain unimplemented. No Stage1_Test labels or probabilities were used or generated.
