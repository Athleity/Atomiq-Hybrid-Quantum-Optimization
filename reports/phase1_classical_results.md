# Phase 1 — Classical Stage 1 baseline

**Status:** Complete for the classical baseline tournament. No Stage1_Test rows were read and no final test probabilities were generated. This remains the classical reference for later quantum Stage 1 comparisons.

## Reproduction

Run from the repository root with the pinned environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_phase1_classical.py
.\.venv\Scripts\python.exe scripts\audit_phase1_results.py --output results\stage1_classical\independent_audit.json
```

The experiment config is `configs/phase1_classical.yaml`. It fixes seed 42, three outer folds repeated three times, two inner stratified folds, two sampled configurations per tuned model, average precision as the inner selection score, and single-threaded fitting. All ten model families use the same materialized outer splits. Each outer validation fold has 100 rows, which allows exact Precision@10 calculation.

## Data and leakage controls

Only `Stage1_Train` was loaded. It has 300 rows, 35 positive and 265 negative outcomes. The loader rejects schema drift, excludes `customer_id` and `y` from predictors, and does not access the workbook README or `Stage1_Test`. The workbook README’s 400-row target total remains prohibited. Imputation, scaling, and one-hot encoding are inside estimator pipelines and are refit within each CV training partition. CatBoost’s missing-value substitutions are estimated inside each `.fit()` call. Validation and calibration labels never fit preprocessing.

## Model portfolio and result

Scores below are means over nine outer validation folds; parenthesized values are the fold standard deviations, not confidence intervals. Repeated CV folds overlap in their training data, so these standard deviations are descriptive and do not support a formal significance claim. PR-AUC is average precision. Ranking metrics are identical for raw and calibrated output except a small PR/ROC tie effect in Extra Trees; top-ten customer membership is identical for every model/fold.

| Model | ROC-AUC raw | PR-AUC raw | Precision@10 | Recall@10 | Brier raw → calibrated | Log loss raw → calibrated | ECE raw → calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|
| catboost_native | 0.656 (0.064) | 0.242 (0.057) | 0.189 | 0.162 | 0.152 → 0.101 | 0.477 → 0.353 | 0.183 → 0.018 |
| svm_linear | 0.594 (0.058) | 0.236 (0.078) | 0.211 | 0.181 | 0.159 → 0.104 | 0.529 → 0.362 | 0.181 → 0.017 |
| svm_rbf | 0.599 (0.095) | 0.233 (0.068) | 0.211 | 0.182 | 0.147 → 0.104 | 0.474 → 0.364 | 0.187 → 0.021 |
| random_forest | 0.608 (0.089) | 0.228 (0.068) | 0.211 | 0.181 | 0.136 → 0.103 | 0.444 → 0.359 | 0.135 → 0.022 |
| lightgbm | 0.590 (0.083) | 0.220 (0.076) | 0.178 | 0.153 | 0.139 → 0.103 | 0.500 → 0.358 | 0.143 → 0.024 |
| logistic_regularized | 0.564 (0.049) | 0.219 (0.063) | 0.211 | 0.181 | 0.155 → 0.103 | 0.580 → 0.359 | 0.163 → 0.025 |
| extra_trees | 0.579 (0.067) | 0.218 (0.084) | 0.189 | 0.162 | 0.114 → 0.104 | 0.555 → 0.364 | 0.093 → 0.023 |
| hist_gradient_boosting | 0.591 (0.065) | 0.216 (0.074) | 0.178 | 0.152 | 0.145 → 0.102 | 0.504 → 0.356 | 0.145 → 0.021 |
| logistic_unregularized | 0.556 (0.037) | 0.202 (0.052) | 0.189 | 0.162 | 0.148 → 0.104 | 1.224 → 0.364 | 0.158 → 0.028 |
| prior_baseline | 0.500 (0.000) | 0.117 (0.005) | 0.122 | 0.102 | 0.103 → 0.103 | 0.360 → 0.360 | 0.007 → 0.007 |

The prior baseline average precision is 0.117, equal to the training prevalence. CatBoost has the highest nested outer-fold mean average precision (0.242, fold SD 0.057) and ROC-AUC (0.656, SD 0.064). Its mean Precision@10 is 0.189 and Recall@10 is 0.162. Linear SVM, RBF SVM, Random Forest, and regularized logistic each have higher mean Precision@10 (0.211) but lower mean average precision. The difference between CatBoost (0.242) and the next model (linear SVM, 0.236) is small relative to fold-to-fold variation; the current CatBoost selection is provisional, not a statistically established winner.

## Calibration findings

Calibration is a cross-fitted sigmoid with a strictly positive slope. It estimates a monotone mapping from inner-fold training-only scores, then maps predictions from the selected model. The positive slope prevents class-direction reversal and preserves top-ten membership. For the CatBoost candidate, mean Brier score decreased from 0.152 to 0.101; mean log loss from 0.477 to 0.353; and 10-bin ECE from 0.183 to 0.018. Across models, calibrated Brier and log loss improved on most of the nine outer folds. Ranking is unchanged for the selected CatBoost model. Calibration remains uncertain because each outer fold has only about 11–12 positives; the reliability-bin counts and fold-level calibration intercept/slope should be inspected before interpreting fine probability differences.

The fitted outer-fold calibration diagnostics remain far from ideal for several estimators (intercept/slope in `model_summary.csv`), despite better held-out Brier/log loss/ECE. This is evidence of calibration instability/domain shift between inner cross-fit scores and the outer model refit, and a reason to retain both raw and calibrated OOF predictions. Do not assume the calibrated model is conclusively reliable for downstream expected-value optimization without later Stage 1/2 sensitivity analysis.

## Final selected classical candidate

The best nested-CV candidate was CatBoost with the final training-only inner search choosing `iterations=140`, `depth=5`, `learning_rate=0.03`, `l2_leaf_reg=5`, and no automatic class weighting. The saved object is `results/stage1_classical/models/selected_model_calibrated.joblib`; its metadata is `results/stage1_classical/selected_model.json`. This fitted estimator has only been smoke-tested on 20 Stage1_Train rows. It has not predicted the challenge test set. The model selection rule chose the highest mean calibrated outer-fold average precision, with Brier score and log loss as tie-breakers.

## Artifacts

- `results/stage1_classical/fold_metrics.csv`: 180 per-model, fold, and raw/calibrated metric rows.
- `results/stage1_classical/oof_predictions.csv`: 18,000 row-level OOF records, with train labels, probabilities, model, fold, and score source.
- `results/stage1_classical/fold_assignments.csv`: frozen repeated validation assignments.
- `results/stage1_classical/tuning_trials.csv` and `final_fit_tuning_trials.csv`: all sampled candidates and inner AP scores.
- `results/stage1_classical/model_summary.csv`: mean and standard deviation by model/output type.
- `results/stage1_classical/calibration_comparison.csv` and `calibration_bins.csv`: paired calibration effects and reliability bins.
- `results/stage1_classical/run_manifest.json`: run ID, versions, seeds/config, training fingerprint, prohibited-input flags, and status.
- `results/stage1_classical/independent_audit.json`: independent metric and top-ten-membership audit.
- `results/stage1_classical/experiment.log`: timestamped per-fold execution log.

## Independent verification and tests

The independent audit recomputed ROC-AUC, average precision, Precision@10, Recall@10, Brier score, log loss, and 10-bin ECE directly from the saved OOF rows using separate metric expressions. It checked training IDs/labels, 90 model-fold prediction groups, 180 stored metric rows, and raw-versus-calibrated top-ten membership. Maximum absolute metric discrepancy was 5.6e-17; there were zero top-ten membership disagreements.

Actual pytest result: **11 passed**. Tests cover strict input schema and target counts, exclusion of identifier/target/duration, fold-local preprocessing and unseen categories, deterministic tie-breaking, search treatment of balanced/unweighted cases, monotone calibration including saturated logistic scores, shared folds, and an end-to-end small nested run that writes training-only OOF outputs and a reloadable model.

## Failed/superseded runs

- The initial 12-fold / 3-candidate protocol was stopped before the first fold finished because CatBoost exceeded the runtime budget. Its partial outputs are retained under `results/stage1_classical_interrupted_20261111/` and are not used for results.
- A completed run using scikit-learn’s unconstrained sigmoid calibration was superseded after audit showed several folds reversed ranking direction. It is retained under `results/stage1_classical_unconstrained_calibration_superseded_20261111/` and is not used for selection.
- A positive-slope run with overly tight probability clipping was also superseded after it merged distinct near-zero scores and changed small-fold ranking metrics. It is retained under `results/stage1_classical_positive_slope_superseded_20261111/`.

## Quantum Stage 1 interface

`run_nested_repeated_cv()` accepts a sequence of `ModelSpec` objects or sklearn-compatible estimators through the `model_specs` argument. It supplies the same repeated stratified outer splits, inner tuning, raw score adapter (`predict_proba` or binary `decision_function`), positive-slope calibration, metrics, manifest, and artifacts. Quantum estimators can be implemented separately and passed to a separate results directory, keeping QML scores out of the classical tournament. For kernel methods, an adapter should expose sklearn-style `fit`/`predict_proba` (or `decision_function`) and store any train-fold-fitted feature scaling internally; do not precompute transforms using all 300 rows.

## Remaining work

This completes the classical baseline phase only. No external test predictions, QML models, Stage 2 optimizer, or Stage 3 circuit were run. Next: Claude may review this implementation; then Phase 2/3 work can add advanced tabular and quantum candidates under the same leakage controls. Before final Stage 1 freeze, rerun the full comparison if the evaluation protocol or candidate set changes.
