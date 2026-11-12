# Project state — ATOMIQ Hackathon 2026

**Current status (2026-11-12):** The complete three-stage challenge pipeline and independent audits are complete. The latest Colab export adds two actual IBM hardware jobs: Stage 3 on ibm_marrakesh and a synthetic Stage 2 QAOA depth comparison on ibm_kingston. A resume-metadata defect in the supplied export mislabeled the Stage 3 backend/layout; the immutable source archive is preserved, while canonical extracted records and notebook resume metadata are corrected and the discrepancy is documented in reports/ibm_hardware_execution_audit.md. No Stage1_Test labels were loaded or used.

## Objective and leakage boundaries

Build a reproducible, scientifically defensible hybrid quantum-classical pipeline for the supplied three-stage challenge. The challenge PDF/workbook defines challenge constraints; the user's pasted brief defines the requested workflow.

- Training: only Stage1_Train labels (300 rows, 35 yes / 265 no).
- Do not use the workbook README's leaked aggregate 46/354 label totals.
- Do not use customer_id as a predictor or use duration.
- Stage1_Test labels are never used. A final inference-only feature read was authorized after Stage 1 freeze; the result is recorded in the final execution section.
- Every cross-validation transformation, estimator, hyperparameter search, and calibrator is trained only on the respective outer-training partition.
- Stage2_Data contains 400 operational rows; the final run joins on exact customer_id and restricts candidates to the frozen Stage1_Test top ten.
- Stage 2: exact-six selection from top ten, maximize p * expected_deposit_value - contact_cost, budget <=25, capacity <=22. Exact enumeration is the oracle (at most 210 subsets).
- Stage 3: six independent Bernoulli customers; estimate P(K>=3), compare with exact Poisson-binomial reference.
- The same frozen Stage 1 probabilities feed the final Stages 2 and 3.

## Completed

### Phase 0 and environment

- Source audit in docs/phase0_audit.md.
- Isolated .venv with CPython 3.12.14 on Windows 11 x86-64.
- Dependency dry run had no conflicts, dependency lockfiles retained, and pip check passed.
- Environment smoke checks passed for scikit-learn, CatBoost, LightGBM, Qiskit, Aer, Qiskit ML and Qiskit Optimization.
- Runtime limits: 8 logical CPUs, 7.8 GiB RAM. No GPU or IBM hardware backend confirmed.
- Environment setup and limits: docs/environment.md.

### Phase 1 implementation and experiments

- Strict training-sheet-only loader: src/atomiq/schema.py.
- Fold-local mixed-type preprocessing: src/atomiq/preprocessing.py.
- Ten candidates: prior baseline; unregularized logistic; elastic-net/L2/L1 logistic; linear and RBF SVM; Random Forest; Extra Trees; HistGradientBoosting; CatBoost native categories; LightGBM.
- Imbalance evaluated inside the inner CV for eligible models as no class weighting vs balanced class weighting. No synthetic resampling.
- Repeated stratified outer CV: 3 folds x 3 repeats (9 folds, each 100 validation rows); 2 inner folds; 2 seeded candidate configurations per tuned family; average precision inner selection; one CPU thread.
- Raw and fold-local positive-slope sigmoid calibrated probabilities retained.
- Metrics: ROC-AUC, average precision (PR-AUC), Precision@10, Recall@10, Brier, log loss, 10-bin ECE, calibration intercept/slope.
- Saved the final nested-CV selected estimator on training rows only; final fit remains provisional.
- Independent audit recomputed 7 predictive/reliability metrics with separate expressions: maximum absolute stored-metric difference 5.6e-17; checked 90 model-fold groups / 180 metric rows; zero raw/calibrated top-ten membership disagreements.
- Actual pytest: 11 passed. Full benchmark completed without an unhandled estimator failure. A prior regularized-logistic convergence warning was addressed by raising max_iter to 20,000; final run had no convergence warnings.
- Claude's Stage 1 V2 archive was reviewed independently. Its saved 9 outer-fold validation assignments exactly match the canonical fold assignments. Its archived soft-vote metrics remain exploratory: candidate-pool adaptation and a secondary in-sample meta-calibration path limit confirmatory interpretation; its code hardcodes Claude-specific paths and its tests do not establish validation-label independence. Full review: reports/phase1_v2_independent_review.md.
- Expanded CatBoost/LightGBM parameter search ran on the same 3x3 outer folds with 8 candidates per model/fold (72 per model total, versus 18 originally). Trial-level and final-fit tuning are preserved separately in results/stage1_classical_expanded_tuning_20261111/. Independent metrics audit passed with maximum absolute recomputation error 2.8e-17 and zero top-10 membership disagreements. Screening did not establish a new overall winner.

## Current benchmark result

Winning mean outer-fold average precision is CatBoost (0.242, fold SD 0.057); mean ROC-AUC 0.656 (SD 0.064); mean Precision@10 0.189 and Recall@10 0.162. Top-10 precision was 0.211 for linear SVM, RBF SVM, Random Forest, and regularized logistic, with lower mean AP. The next AP candidate after CatBoost is linear SVM at 0.236 (SD 0.078), so the observed difference is small relative to fold variation and does not establish statistical superiority.

For CatBoost, positive-slope sigmoid calibration reduced mean Brier from 0.152 to 0.101, log loss from 0.477 to 0.353, and ECE from 0.183 to 0.018. The positive slope prevents ranking inversion. Validation diagnostics still show residual calibration instability for several estimators; probability quality must be reviewed with downstream sensitivity before freezing.

Selected candidate, provisional:
- CatBoost native categoricals.
- Final nested inner-search parameters: iterations=140, depth=5, learning_rate=0.03, l2_leaf_reg=5.0, auto_class_weights=None.
- Reloadable artifact: results/stage1_classical/models/selected_model_calibrated.joblib.
- Metadata: results/stage1_classical/selected_model.json.
- No Stage1_Test predictions generated.

Expanded-search result: CatBoost AP 0.243 (fold SD 0.069), ROC-AUC 0.614, Precision@10 0.244, Brier 0.102, log loss 0.357. Versus original CatBoost, AP changes +0.0015 and Precision@10 +0.0556, but ROC-AUC falls by 0.0410 and Brier/log loss are slightly worse. Fold-level AP improves in only 4/9 folds. This trade-off is unstable; original CatBoost remains the provisional reference. Expanded LightGBM AP remains 0.221 with no meaningful gain.

Full results and caveats: reports/phase1_classical_results.md and reports/phase1_v2_independent_review.md.

## Reproduction and artifacts

From repository root:

- `.\\.venv\\Scripts\\python.exe -m pytest -q`
- `.\\.venv\\Scripts\\python.exe scripts\\run_phase1_classical.py`
- `.\\.venv\\Scripts\\python.exe scripts\\audit_phase1_results.py --output results\\stage1_classical\\independent_audit.json`

Primary result folder: results/stage1_classical/:
- fold_metrics.csv, oof_predictions.csv, fold_assignments.csv
- tuning_trials.csv, final_fit_tuning_trials.csv
- model_summary.csv, calibration_comparison.csv, calibration_bins.csv
- run_config.yaml, run_manifest.json, selected_model.json
- models/selected_model_calibrated.joblib
- independent_audit.json, experiment.log

Superseded runs are retained and explicitly marked:
- results/stage1_classical_interrupted_20261111/ — initial slower search stopped before a full fold; partial diagnostics only.
- results/stage1_classical_unconstrained_calibration_superseded_20261111/ — scikit-learn sigmoid could reverse ranking direction on small folds; invalidated for selection.
- results/stage1_classical_positive_slope_superseded_20261111/ — clipping merged distinct tiny raw scores; superseded by machine-bound clipping.

## Quantum Stage 1 integration interface

run_nested_repeated_cv accepts a sequence of ModelSpec/sklearn-compatible estimators through model_specs. Classical estimators accept the shared pandas feature DataFrame; generic adapters can implement predict_proba or binary decision_function. Quantum kernels/QVCs should be separate adapters and write to a separate output directory, while reusing the same folds, metrics, seeds, and calibration rules. Any feature scaling or encoding must be fit inside each fold. Do not precompute transformations over all 300 labeled rows.

## Environment details

- Use .venv or .venv\Scripts\python.exe; default WindowsApps python alias is unavailable.
- requirements.in and requirements-lock.txt are preserved and unchanged.
- Qiskit Aer is in reduced-maintenance mode; no GPU package or hardware credentials are configured.
- qiskit-algorithms excluded due upstream unsupported status.

## Known limitations

- Nine validation folds on 300 rows with 35 positives are noisy; fold SDs are descriptive and folds overlap. No formal significance tests or confidence intervals have been asserted.
- Inner search budget is deliberately small for host speed (two parameter candidates per tuned model/fold); it is suitable for screening, not exhaustive hyperparameter optimization.
- Calibration diagnostics are unstable with small positive counts and show some residual model-score domain shift. Retain raw/calibrated results; avoid treating calibrated probabilities as precise truth.
- The top-ten Precision@10 metric is low-count: each fold has 10 selected rows and typically 11-12 positives in the 100-row validation set.
- At the time of the original Phase 1 status, Stage2 exact-oracle and Qiskit diagnostics used provisional/synthetic inputs, Stage3 used provisional OOF probabilities, and final challenge predictions had not been executed. The later final execution is recorded at the end of this file.
- Claude's V2 soft-vote has not been independently re-fit; its archived result is not confirmatory because the model pool was adapted after observing base-model comparisons. V2 model scripts are not integrated into the canonical runner.
- TabPFN, TabICL, XGBoost, and InterpretML/EBM are absent from the active environment and have not been installed or tested. Compatibility with optional dependency versions remains unassessed.

## Last successful actions

Full classical benchmark with final positive-slope calibrator completed on 2026-11-11. Independent metric audit passed; serialized selected estimator loaded and produced valid probabilities on 20 training rows. Expanded CatBoost/LightGBM screen and independent audit completed. The latest test suite covers the Stage 2/3 circuit additions and passes 54 tests.

## Next exact action (superseded by final execution below)

Freeze the Stage 1 candidate/calibration decision before creating final Stage1_Test probabilities. Then rebuild the top-ten Stage 2 pool from those frozen probabilities, compare the classical oracle with Qiskit circuits and classical optimizer baselines, and run Stage 3 on the resulting six probabilities. Preserve the current provisional training-only circuit diagnostics as development evidence; do not promote them to final challenge results.


## Stage 2 initial development update (2026-11-11)

Stage 2 development has started without changing Stage 1 benchmark artifacts or loading Stage1_Test. `src/atomiq/stage2.py` loads only `Stage2_Data` and `Stage2_Constraints`, evaluates the six-of-ten objective and constraints, exhaustively enumerates all 210 subsets, and cross-checks the optimum against an independently encoded SciPy MILP (including exact integer slack encoding in the QUBO representation). The training-only candidate pool derives from repeated Stage1_Train OOF calibrated scores; its oracle result is provisional and must not be presented as the final challenge selection.

Current development oracle selects C0073, C0094, C0130, C0159, C0217, C0329, with objective 9.54561244282308, contact cost 12/25, and capacity 17/22. Exhaustive enumeration and MILP agree. In this particular pool all 210 fixed-six subsets are feasible, so it cannot validate binding-inequality handling.

`src/atomiq/stage2_quantum.py` adds a standard shared-angle cycle XY-QAOA control with exact Dicke initialization, COBYLA tuning against exact statevector expectation, Aer finite-shot sampling, explicitly illustrative noise stress testing, and circuit resource reporting. This is not IWS-XY, DLA warm-start QAOA, IF-QAOA, or QTG-QAOA. Their primary-source mechanisms and exact adaptation requirements are documented in `reports/stage2_methodology_research.md`; none is being misrepresented as implemented. The current run is a provisional small-instance development experiment, not final benchmark evidence.

Validation: `.\\.venv\\Scripts\\python.exe -m pytest -q` passed (20 tests). `scripts/run_stage2_oracle_dev.py` completed and independently agreed across enumeration and MILP. `scripts/run_stage2_quantum_dev.py` completed for seeds 11/23/42 with COBYLA stopping after 58/56/40 objective evaluations, exact expected objectives 6.9269/6.1813/5.8861, exact optimum probabilities 0.0259/0.0156/0.0106, and illustrative noisy feasible-shot rates 0.253/0.224/0.241. Circuit transpilation reported 6264 depth and 1053 CX gates in the stated basis. Results and caveats are in `reports/stage2_methodology_research.md` and `results/stage2_quantum_development/xy_qaoa_development_results.json`. No physical QPU was used. Full Stage 2 report, executable commands, and limitations: `reports/stage2_methodology_research.md`.

## Stage 2 advanced-method development update (2026-11-11)

Implemented and exercised four research adaptations on a shared synthetic binding-constraint fixture, without changing Stage 1 outputs or lockfiles, and without loading Stage1_Test:

- IWS-XY exact-K weighted Johnson graph and Boltzmann marginal updates (`stage2_iws_xy.py`). The four-item, choose-two fixture has 3 feasible of 6 states and exact optimum 39.5. Three seeds with four rounds of 1,024 ideal samples produced final feasible rates around 0.995–0.997 and optimum sample rate 0.992. COBYLA hit its 100-evaluation cap for all seeds. This dense compressed-sector generalization does not retain the paper's shallow one-hot circuit guarantee.
- DLA-informed warm start (`stage2_dla_warm_start.py`): numerical restricted/full DLA dimensions are 15/36 in the six-dimensional sector. 100+100 Adam steps transfer into an expanded ansatz; matched random-init ablation shows no consistent warm-start advantage. Multi-angle expected feasible objectives by seed were 38.176/35.222/38.370 versus random initialization 38.364/38.370/38.369. Exact statevector only.
- IF-QAOA signed QPE slack indicators, controlled objective phase and uncomputation (`stage2_if_qaoa.py`). Direct arithmetic, phase and uncomputation checks pass. Seed 11 reached its 100-call cap; expected feasible objective 19.6724, feasible probability 0.6023 and optimum probability 0.2831. Diagnostic only; no finite-shot or noise run.
- QTG/AAM-QAOA high-level feasible-subspace simulator (`stage2_qtg_qaoa.py`): all supported states are feasible. Its suffix-feasibility predicate is classical dynamic programming, not a reversible quantum circuit. All six seed/bias optimizations hit the 80-call cap. Bias 0 expected objectives 37.642–37.845 and optimum probabilities 0.764–0.768; bias 1 expected objectives 36.007–36.230 and optimum probabilities 0.515–0.519. No gate decomposition/noisy run.
- Standard XY binding control: exact feasible probabilities 0.810–0.871; illustrative, non-calibrated noisy Aer shot feasibility 0.558–0.589. It preserves exact cardinality but not both resource constraints.

Shared inputs/configs/results are in `src/atomiq/stage2_development_fixtures.py`, `configs/stage2_*_development.yaml`, and dedicated `results/stage2_*_development/` folders. The training-only ten-candidate pool result remains separate and all 210 subsets are feasible, so it cannot assess constraint handling. No advanced result is a challenge-scale benchmark, hardware result, or quantum advantage claim. See `reports/stage2_methodology_research.md` for method mappings, limitations, source citations and exact commands.

Verification for that batch: pytest passed 38 tests in 22.40 seconds and pip check was clean. No multi-depth or common-budget convergence study was part of that batch.

## Qiskit circuit and noise evaluation (2026-11-12)

Inspected and reused the installed CPython 3.12.14 environment with Qiskit 2.6.2 and Qiskit Aer 0.18.2. No package install/upgrade occurred; `requirements.in` and `requirements-lock.txt` are unchanged. No physical backend calibration was available or retrieved.

- Added `src/atomiq/stage2_qiskit_circuits.py`: symbolic standard QAOA and exact-K XY-QAOA circuits, a scalable additive phase shortcut when every fixed-K state is feasible, exact penalty phase when inequalities bind, COBYLA/SPSA under comparable statevector-evaluation budgets, expectation/CVaR objectives and depth interpolation.
- Added `src/atomiq/qiskit_noise.py`: hypothetical 7-qubit linear backend NoiseModel with per-qubit T1/T2, gate durations, single/two-qubit depolarization, measurement-window relaxation, asymmetric readout, fixed-layout routing/resource reports, readout-only Aer assignment calibration, matrix inversion/simplex projection and bootstrap variance overhead. The 10-qubit profile repeats the declared assumptions for qubits 7–9. Neither profile is hardware-calibrated.
- Four-item binding fixture QAOA sweep (seed 11, 2,048 evaluation shots): noisy XY feasibility 0.179–0.193 versus ideal statevector feasibility 0.526–0.707; readout projection 0.184–0.199. Mitigation variance ratio 1.35–1.39; uncertainty increased. COBYLA runs were capped at 12 calls and SPSA at 10 calls; none is certified converged. A noisy SPSA run completed 25 evaluations of 128 shots each in 40.9 seconds.
- Depth-one exact-K XY mixer resource comparison on the seven-qubit line: path/cycle/complete had 167/168/170 routing swaps, 1,091/1,106/1,124 transpiled depth, and 691/701/712 CX gates. The exact truth-table phase-oracle synthesis dominates these small fixture costs.
- Executed a 10-qubit, exact-six XY circuit using seed-11 angles from the existing exact-statevector development fit. It used the training-only OOF candidate pool and did not load Stage1_Test. Resource report: depth 6,010, 7,089 gates, 3,081 CX and 981 routing swaps. At 512 noisy shots feasibility was 0.1230, exact-optimum sample probability 0, and best feasible measured objective 9.0826 versus exact optimum 9.5456. Readout projection moved feasibility to 0.1267 and increased bootstrap variance 43.7%.
- Added a 6-qubit Stage 3 RY-Bernoulli circuit, per-qubit probability checks, bitstring-order tests and independent Poisson-binomial dynamic program. For the provisional six OOF probabilities, exact P(K>=3)=0.05426956001511194 and Qiskit statevector differs by 2.1e-17. At 2,048 noisy shots across seeds 11/23/42, estimates were 0.07031/0.07080/0.07129; readout-projected 0.05974/0.05994/0.06033. Projection reduced point bias but increased bootstrap variance 33–37%. At seed 11, 512- and 8,192-shot sweeps are also saved.
- Prior IWS/DLA results remain classical compressed-sector simulations; QTG remains a high-level simulator with classical feasibility DP; IF-QAOA has an actual Qiskit circuit but no finite-shot/noisy experiment. This work did not relabel or duplicate those implementations as hardware-capable methods.

Evidence and reproduction: `reports/qiskit_noise_evaluation.md`, `configs/qiskit_circuit_noise.yaml`, `results/stage2_qiskit_noise/qiskit_stage2_noise_results.json`, `results/stage2_qiskit_ten_candidate/ten_candidate_exact_six_qiskit_results.json`, and `results/stage3_qiskit_noise/qiskit_stage3_noise_results.json`. Commands are documented in README and `docs/reproducibility.md`.

Verification: `.\\.venv\\Scripts\\python.exe -m pytest -q` passed 54 tests in 71.33 seconds. `.\\.venv\\Scripts\\python.exe -m pip check` reported no broken requirements. `scripts/audit_qiskit_noise_results.py` independently re-decoded persisted counts and matched all Stage 2/3 feasibility, optimum and threshold metrics; it saved `results/qiskit_noise_independent_audit.json` and full mitigated distributions. Stage 2/3 circuit runs and readout calibration completed under Aer. No physical QPU run occurred.

At the end of the Qiskit noise-evaluation batch, the remaining work was final Stage 1 freeze, feature-only inference, and actual-customer Stage 2/3 execution. That work was completed in the final execution section below. The illustrative-noise and small-budget optimizer limitations still apply.

## Final challenge execution (2026-11-12)

The selected Stage 1 CatBoost model was frozen from Stage1_Train before Stage1_Test inference. Freeze provenance is `results/stage1_classical/frozen_model.json`; `results/stage1_classical/inference_manifest.json` records the later feature-only scoring action and prediction digest. `src/atomiq/schema.py::load_stage1_inference_data` uses an explicit `usecols` list for the ID and 15 allowed predictors; it never requests a target column. The persisted inference output has 100 probabilities and deterministic tie handling (probability descending, customer ID ascending): `results/stage1_classical/stage1_test_predictions.csv`. No test labels were loaded or used for selection, tuning, or evaluation.

Stage 1 frozen selection: native-categorical CatBoost, iterations 140, depth 5, learning rate 0.03, L2 5, calibrated with the training-only OOF positive-slope sigmoid. Nested outer validation estimates: PR-AUC 0.241709, ROC-AUC 0.655538, Precision@10 0.188889, Recall@10 0.161616, Brier 0.101247, log loss 0.352610. These remain noisy internal validation estimates on 300 rows / 35 positives, not held-out test metrics.

The top ten frozen inference candidates were joined to Stage2_Data by exact customer ID. Exhaustive enumeration evaluated all 210 exact-six subsets; all were feasible. The independent SciPy MILP selected the same optimum: C0028, C0031, C0079, C0081, C0327, C0338; expected net objective 16.1710314161; contact cost 16/25; capacity 15/22. Artifacts: `results/stage2_final/actual_top_ten.csv`, `actual_selected_six.csv`, and `actual_stage2_result.json`.

The actual Stage 3 probabilities from those six customers were evaluated by independent Poisson-binomial dynamic programming, explicit 64-outcome enumeration, a six-qubit Qiskit circuit and Aer. Both exact classical methods return P(K>=3)=0.2078515973394121; ideal Qiskit statevector agrees with absolute difference zero. Statevector marginals equal each selected frozen Stage 1 probability. Qubit q_i maps to classical bit c_i; count strings display c5...c0. The transpiled circuit uses 6 logical qubits, depth 4, 24 one-qubit gates, zero two-qubit gates and zero swaps. Noise assumptions are explicitly illustrative, not hardware calibration. Twelve paired finite-shot runs cover 512, 2,048, 8,192 and 32,768 shots, three seeds per level, with ideal Aer, noisy Aer, asymmetric readout calibration and readout-mitigated estimates. Exact results and per-run CIs/timing: `results/stage3_final/actual_stage3_results.json`; report and plots: `reports/final_actual_challenge_results.md` and `reports/figures/actual_stage3_*`.

In these Stage 3 runs, mitigation did not consistently reduce absolute error. Across three seeds, the mean raw noisy estimates at 512/2,048/8,192/32,768 shots were 0.180339/0.207520/0.213053/0.214742; projected mitigated means were 0.163128/0.195841/0.198778/0.200688. Estimated mitigation variance was 1.326 times raw on average. Readout mitigation corrects only the readout channel; uncertainty intervals do not include finite-shot assignment-calibration uncertainty. The illustrative product-state circuit has no two-qubit gates, so Stage 3 did not apply an artificial two-qubit error channel.

Added scripts: `scripts/freeze_and_predict_stage1.py`, `scripts/run_actual_stage2.py`, `scripts/run_actual_stage3.py`, and `scripts/run_actual_stage2_qaoa.py`. Stage 3 unit coverage explicitly checks six-qubit/measurement count, customer-to-qubit/classical-bit mapping, RY angles, unique asymmetric marginals, full 64-state independent distribution, endpoint probabilities and Qiskit display ordering.

The actual-pool QAOA diagnostic completed: four one-seed COBYLA/SPSA × expectation/CVaR fits, each capped at 10 evaluations, with ideal Statevector and equal 2,048-shot ideal/noisy Aer sampling plus readout mitigation. None certified convergence. Ideal statevector expected rewards ranged 10.420–11.400 versus exact 16.171; noisy measured feasible-shot rates were about 0.11–0.12, and the exact optimum was not sampled at the noisy shot budget. Transpiled cycle XY resources were depth 6,010, 7,089 gates, 3,081 CX, and 981 routing swaps. Full results: `results/stage2_final/actual_stage2_qaoa_results.json`. This circuit uses the exact additive objective phase because all 210 fixed-six states are feasible; it does not demonstrate binding-constraint handling.

Verification: the first full pytest pass exposed four incorrect new endpoint-test assertions that compared a 64-outcome vector to a seven-bin weight histogram. The tests were corrected to aggregate by Hamming weight; focused Stage 3/schema tests passed (15 tests), then the corrected full suite passed (61 tests in the final run, 29.12 seconds). `scripts/audit_actual_results.py` passed independently: zero Stage 2 enumeration/MILP objective gap; zero QAOA statevector/sample metric recomputation error; Poisson-binomial DP vs 64-outcome enumeration maximum absolute difference 5.55e-17; statevector weight-histogram difference 5.55e-17; all 24 raw shot-rate/Wilson checks passed. No test labels were read.

## Final presentation audit (presentation-only)

Accepted computational milestone: `cfe9afcf72f05a730bdf9c15ce8c3556e3b27983`. A presentation-only audit generated missing Stage 1-3 figures from existing audited results and typeset `output/pdf/atomiq_final_research_report.pdf`. No experiments were rerun, algorithms or environment changed, or test labels read. Two pre-existing Stage 3 figures and all computational outputs remain unchanged. The actual-pool Stage 2 optimizer histories were not saved, so the report presents evaluation-cap/convergence-status evidence rather than an invented trajectory. Full figure provenance and audit details: `reports/presentation_audit.md`.


## Final GitHub submission and actual IBM hardware evidence (2026-11-12)

The latest supplied Colab evidence was inspected, packaged under `results/ibm_hardware/`, and cross-checked against its raw counts, job IDs, processed results, and contemporaneous original submission information. The original ZIP is preserved byte-identically as `source_export_unmodified.zip` (SHA-256 recorded in `provenance_correction.json`).

- Stage 3 job `dbqfi99abdes73fo6vig` completed on `ibm_marrakesh`, qubits `[87, 6, 10, 139, 120, 107]`, with 8,192 experiment shots plus two 2,048-shot readout calibrations and 5.0 reconciled QPU seconds. Raw estimate 0.201538 (95% Wilson interval 0.192992–0.210364); readout-mitigated 0.202275. The paired mitigation absolute-error reduction CI includes zero, so mitigation is inconclusive.
- Synthetic choose-three QAOA job `dbqg5d1abdes73fo7o10` completed on `ibm_kingston`, qubits `[147,148,149,150,151,152]`, 30,000 shots per depth p=1 and p=2, with 8,000 shots for each calibration state per depth; 26.0 reconciled QPU seconds. Hardware approximation ratios were 0.470624 (p=1) and 0.623956 (p=2); corresponding exact-optimum probabilities were 0.0164 and 0.005633. p=2 improved this ratio while lowering exact-optimum sampling probability. QAOA was noise-free optimized before hardware sampling; no hardware-in-loop optimization. The instance is synthetic, not the actual customer selection.
- Total usage across both completed jobs: 31.0 QPU seconds. Actual QPU execution is confirmed. No quantum advantage is claimed.
- The export resumed the old Stage 3 job using the currently active Kingston backend/layout and consequently mislabeled that historical job. Canonical extracted registry/count metadata and the notebook's resume path now preserve original job-specific provenance. Kingston backend calibration in the export applies to the QAOA hardware job and Kingston-based Aer simulations; it is not Marrakesh calibration. See `reports/ibm_hardware_execution_audit.md`.
- The submission README and notebook have been prepared. The notebook contains no API key and no embedded run outputs; the completed evidence is supplied separately. Original challenge source PDF/workbook are excluded from public packaging because redistribution authorization is unverified; they remain untouched locally.

Validation for packaging: source ZIP SHA-256 and its contents verified, notebook JSON and Python code cells parsed, secrets scan clean, all local README/report links resolved, and the existing pytest suite run without rerunning benchmarks or QPU jobs. The public GitHub repository history must not include the local `data/raw/` source files; publication uses a sanitized snapshot so prior local commits and accepted computational history remain intact locally.
