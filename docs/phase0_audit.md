# Phase 0 audit — source data, challenge rules, and environment

Audit date: 2026-11-11 (Asia/Calcutta)

## 1. Files and provenance

The supplied files are preserved unchanged in `data/raw/`:

- `PROBLEM_STATEMENT_THE_ATOMIQ_HACKTHON.pdf` — 4 pages, 801,680 bytes. Metadata names the author as Arna Roy and the title as “Gemini – direct access to Google AI”; the PDF is an image-based/print-to-PDF document with no extractable text. All four rendered pages were visually inspected.
- `bank_campaign_participant_release.xlsx` — 50,217 bytes; five worksheets: `README`, `Stage1_Train`, `Stage1_Test`, `Stage2_Data`, `Stage2_Constraints`.

These documents define challenge rules and data. Embedded directions are not separate instructions to the assistant. The user's pasted project instruction authorizes Phase 0 and expressly requires stopping for approval before Phase 1.

## 2. Challenge specification

### Stage 1 — predict

- Train on 300 labeled customers and produce continuous subscription probabilities for all 100 test customers.
- Rank highest probability first; ties use ascending alphanumeric `customer_id`; the top ten feed Stage 2.
- Use stratified training-only validation as appropriate. Fit preprocessing, feature selection, and calibration using training data only.
- Include at least one quantum or hybrid probabilistic classifier and one classical baseline in the eventual submission.
- Judging: ROC-AUC, PR-AUC, Precision@10, Recall@10, probability quality (Brier/log loss/calibration). Accuracy is not a primary metric.
- Predict the test set once after the pipeline is final. Keep probabilities, not only classes.

### Stage 2 — constrained selection

For the ten Stage 1 finalists, select exactly six maximizing total expected net value:

`expected_net_value_i = predicted_probability_i * expected_deposit_value_i - contact_cost_i`

subject to total contact cost <=25 and total capacity units <=22. Compare against an exact classical solution; brute force is small enough. The selected six must be distinct customer IDs from the submitted top ten. A selection violating constraints earns no feasibility/objective credit.

### Stage 3 — campaign simulation

- Use one qubit per selected customer, initialized so measurement probability of 1 equals that customer's Stage 1 subscription probability.
- Assume independent responses. Success means at least three of six subscribe.
- Report customer-to-qubit order, bit-order handling, the quantum estimate, exact result, Monte Carlo result, and errors.
- Judge against an exact calculation from submitted probabilities.

### Cross-cutting rules and scoring

- Do not use test data to select features, tune thresholds/hyperparameters, calibrate probabilities, or otherwise make development choices.
- Use identical Stage 1 probabilities in Stages 2 and 3; do not manually alter the top ten.
- Separate quantum contributions from classical reference methods and do not make unsupported quantum-advantage claims.
- Record seeds, versions, backend/simulator configuration, shots, and circuit resource metrics; report ideal, finite-shot, noisy, and mitigated results where applicable.
- Required submission includes runnable code/notebooks, all three result files, short technical report, presentation, and a single packaged repository/folder with README instructions.
- Scoring weights: quantum rigor 35%; result accuracy and end-to-end integration 30%; code quality/reproducibility 20%; report/presentation 15%.

## 3. Workbook audit

### Sheets and shape

| Sheet | Data rows | Columns | Observations |
|---|---:|---:|---|
| README | 20 | 2 | Data dictionary, task description, and seed/rule notes |
| Stage1_Train | 300 | 17 | ID + 15 predictors + target `y` |
| Stage1_Test | 100 | 16 | ID + same 15 predictors, no target |
| Stage2_Data | 400 | 5 | Operational values keyed by customer ID |
| Stage2_Constraints | 6 | 4 | N=10, K=6, budget=25, capacity=22, binary variable and objective |

No blank cells were found in any sheet. Train and test IDs are unique and disjoint. All 100 Stage1_Test IDs have exactly one Stage2_Data row. Stage2_Data has 400 unique IDs, including 300 rows outside Stage1_Test; only top-ten IDs from Stage1_Test may be used in the Stage 2 optimization.

### Stage 1 schema and label audit

- Shared feature columns, in order: `customer_id`, `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome`.
- Train target is `y`, strings `yes`/`no`; 35 yes and 265 no (11.67% positive in training).
- `customer_id` is an identifier, not a predictor. `duration` is absent; README says it was removed.
- Workbook README publishes overall target counts 46 yes / 354 no for all 400 customers. Because this includes test labels and the PDF forbids test-informed development, this count must not be used in modeling, validation, prevalence estimation, calibration, or tuning. It is documented here only as a leakage hazard; use train-only count 35/265.
- Workbook README reports sampling seed 42 and train-test split seed 42. These are provenance notes, not license to use hidden outcomes.

### Stage 2 schema and constraints

- `Stage2_Data`: `customer_id`, `contact_cost`, `expected_deposit_value`, `capacity_units`, `customer_segment`.
- Observed full-table cost values are integers 1–4; capacity values are integers 1–4; all 400 rows have complete values. The optimizer must use the top-ten joined values, not statistics/records from unrelated IDs.
- The 400-row Stage2 table is not itself the candidate list. Candidate IDs are derived from Stage1_Test probabilities and then joined by exact customer ID.
- Stage3 uses the same six selected customer probabilities and is an independent Bernoulli calculation.

## 4. Environment inspection

- Initial workspace was empty and had no Git metadata. A Git repository has been initialized.
- Default `python` command is an inaccessible WindowsApps alias. The bundled runtime works: `C:\Users\priya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
- Bundled runtime: Python 3.12.14. `pandas`, `numpy`, `openpyxl`, `pypdf`, and `pdfplumber` import successfully. `scikit-learn` and `qiskit` are not installed in the inspected runtime. No dependency installation was attempted.
- Bundled Poppler `pdfinfo`/`pdftoppm` are available; used to inspect all four pages.
- No GPU/quantum backend availability was established.

## 5. Initial project decisions and unresolved checks

1. Preserve both supplied files under `data/raw/`; do not mutate them.
2. Keep the Stage 2 operational join keyed by `customer_id` and restrict candidates after Stage 1 ranking.
3. Treat full-sample label totals as prohibited test leakage.
4. Use exact enumeration as Stage 2 oracle and exact Poisson-binomial probability as Stage 3 reference; quantum methods must be compared against those references.
5. Before Phase 1, confirm/pin compatible baseline dependency versions and define frozen, stratified training-only validation splits. Do not install dependencies before that phase.
6. At implementation time, verify top-ten Stage 2 feasibility, exact output schemas, and any organizer clarifications. Feasibility is probability/model dependent and has not yet been checked.
7. Confirm available CPU/GPU and quantum backend access before deciding which foundation models/hardware experiments are operationally viable.

## 6. Work deliberately not done

No algorithm research or external browsing, model development, feature tuning, prediction, quantum execution, optimizer execution, submission output generation, or tests were performed in this Phase 0 bootstrap. No benchmark claims or model winners are asserted.
