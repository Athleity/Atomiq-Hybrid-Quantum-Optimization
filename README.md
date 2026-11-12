# Atomiq: Hybrid Quantum–Classical Campaign Optimization

A reproducible, three-stage solution to the Atomiq term-deposit campaign challenge. It predicts subscription probabilities, selects six of the top ten customers under operational constraints, and estimates the probability that at least three of the selected customers subscribe. The repository separates classical references, quantum circuit experiments, simulation, and actual IBM hardware evidence so readers can see exactly what was measured.

> **Scientific scope.** One Stage 3 job and one synthetic Stage 2 QAOA job were executed on IBM Quantum hardware. The QAOA instance is synthetic; it is not the bank-customer optimization. No quantum advantage is claimed. The exact classical Stage 2 oracle and Stage 3 Poisson-binomial calculation remain authoritative references.

## Results at a glance

| Stage / experiment | Result | Interpretation |
|---|---:|---|
| Stage 1 nested repeated CV | CatBoost PR-AUC 0.2417 (SD 0.057), ROC-AUC 0.6555 (SD 0.064), Brier 0.1012 | Training-only validation on 300 customers / 35 positives; noisy estimates, not test metrics |
| Stage 2 actual challenge pool | Six IDs: C0028, C0031, C0079, C0081, C0327, C0338; objective 16.1710314161 | Exact enumeration of 210 subsets agrees with independent SciPy MILP; all subsets in this pool are feasible |
| Stage 3 exact reference | P(K ≥ 3) = 0.2078515973394121 | Poisson-binomial dynamic programming agrees with explicit enumeration of 64 outcomes |
| Stage 3 IBM hardware | 0.201538; 8,192 shots; 95% Wilson interval [0.192992, 0.210364] | Job `dbqfi99abdes73fo6vig`, correctly attributed to `ibm_marrakesh`; 5.0 QPU seconds |
| Synthetic QAOA p=1, IBM hardware | Feasibility 0.8147; approximation ratio 0.470624 | 30,000 shots; job `dbqg5d1abdes73fo7o10` on `ibm_kingston` |
| Synthetic QAOA p=2, IBM hardware | Feasibility 0.7615; approximation ratio 0.623956 | 30,000 shots; same job/backend; higher ratio than p=1 but lower exact-optimum sampling probability |

Stage 3 readout-mitigated estimate was 0.202275 and the paired absolute-error reduction was inconclusive (95% interval −0.001289 to 0.001840). For synthetic QAOA, mitigation raised feasibility, while approximation-ratio changes were small and intervals overlap. See the [hardware execution audit](reports/ibm_hardware_execution_audit.md) for full detail and a material metadata correction.

## Project structure

```text
configs/                     Seeds, model/search and simulation settings
src/atomiq/                  Modular preprocessing, models, optimization and circuits
scripts/                     Reproducible experiment runners and independent audits
tests/                       Unit and integration tests
notebooks/                   IBM hardware experiment notebook
results/stage1_classical/    CV predictions, fold metrics, searches, calibration and audits
results/stage2_final/        Actual-pool oracle, selected six and QAOA diagnostic
results/stage3_final/        Actual six-customer probability and circuit results
results/ibm_hardware/        Raw Colab export, extracted counts, calibration and QAOA results
reports/                     Research reports and provenance audits
reports/figures/             Publication-ready figures
output/pdf/                  Typeset research report
PROJECT_STATE.md             Detailed project history, exact commands and limitations
```

The original challenge PDF and participant workbook are not included in this public repository because redistribution permission has not been established. The source files remain in the local working repository. To reproduce data-dependent runs, obtain the files through the authorized competition channel and place them under `data/raw/` as documented in `data/raw/README.md`. The software and result artifacts do not include Stage1_Test labels or label-derived aggregate totals.

## Leakage and evaluation boundaries

- Stage 1 fitting, model selection and calibration use `Stage1_Train` labels only. Customer IDs and `duration` are excluded from predictors.
- Preprocessing, estimator fitting, tuning and positive-slope sigmoid calibration are fold-local.
- After model freeze, inference reads only the test customer ID and allowed feature columns. Test labels are neither loaded nor used for feature/model/threshold selection.
- Stage 2 joins on customer ID, evaluates all 210 exact-six choices and cross-checks the optimum with an independently encoded MILP.
- Stage 3 reuses the frozen six probabilities and assumes independent Bernoulli outcomes, as specified by the challenge.
- Synthetic development fixtures, interrupted searches and superseded calibration runs are preserved separately and labeled in `PROJECT_STATE.md` and reports.

## Reproduce locally

Validated local environment: Windows x86-64, CPython 3.12.14, Qiskit 2.6.2 and Qiskit Aer 0.18.2. The requirements and lockfiles are retained. From PowerShell at the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

To run the training-only Stage 1 benchmark and independent audit:

```powershell
.\.venv\Scripts\python.exe scripts\run_phase1_classical.py
.\.venv\Scripts\python.exe scripts\audit_phase1_results.py --output results\stage1_classical\independent_audit.json
```

The complete command catalog, including development-only quantum simulation and final-result audit commands, is in [`docs/reproducibility.md`](docs/reproducibility.md). Final artifacts are already checked in; the final hardware jobs are historical evidence, not a request to submit new QPU jobs.

## IBM hardware notebook and evidence

Open [`notebooks/atomiq_ibm_hardware_experiments_QAOA_HARDWARE_EXPANDED.ipynb`](notebooks/atomiq_ibm_hardware_experiments_QAOA_HARDWARE_EXPANDED.ipynb) in Google Colab. Add `IBM_QUANTUM_API_KEY` to Colab Secrets; `IBM_QUANTUM_INSTANCE` is optional when IBM resolves the desired instance. The notebook is disarmed for new hardware submissions by default; enabling jobs requires changing the explicit hardware, confirmation, and job allowlist gates. Its outputs have been cleared to avoid presenting stale run output as a new execution; completed run data, raw counts and figures are in [`results/ibm_hardware/`](results/ibm_hardware/).

The supplied Colab export had one provenance error: while resuming the completed Stage 3 job, it copied the currently selected Kingston backend and layout onto a job previously submitted to Marrakesh. The job ID and raw counts are intact. Canonical extracted records and the notebook's resume metadata now identify Stage 3 as `ibm_marrakesh` on `[87, 6, 10, 139, 120, 107]`; the original export is retained unchanged for audit. The synthetic QAOA job is correctly attributed to `ibm_kingston` on `[147, 148, 149, 150, 151, 152]`. See [`reports/ibm_hardware_execution_audit.md`](reports/ibm_hardware_execution_audit.md) and [`results/ibm_hardware/provenance_correction.json`](results/ibm_hardware/provenance_correction.json).

## Research report and authoritative outputs

- [Final research report (PDF)](output/pdf/atomiq_final_research_report.pdf)
- [Final actual challenge results](reports/final_actual_challenge_results.md)
- [Stage 1 benchmark and calibration report](reports/phase1_classical_results.md)
- [Qiskit noise evaluation](reports/qiskit_noise_evaluation.md)
- [Stage 2 methods and development limits](reports/stage2_methodology_research.md)
- [Hardware execution audit](reports/ibm_hardware_execution_audit.md)
- [Presentation audit and figure provenance](reports/presentation_audit.md)

For current result authority, use the machine-readable artifacts in `results/stage1_classical/`, `results/stage2_final/`, `results/stage3_final/`, and `results/ibm_hardware/source_export_contents/processed/`, together with their independent audits and reports. For hardware provenance specifically, use the corrected canonical extraction and correction JSON, not the backend/layout fields in the immutable raw export.

## Limitations

- Stage 1 has only 300 labeled rows and 35 positive cases. Nine overlapping validation folds give noisy descriptive estimates; no held-out test metric or statistical superiority claim is made.
- All 210 actual-pool Stage 2 subsets satisfy the given constraints, so the final pool does not test binding-constraint handling.
- Actual-pool QAOA parameter searches were small and capped; they did not converge reliably and did not beat the exact oracle. The hardware QAOA experiment uses a synthetic objective.
- Stage 3 relies on independent customer responses and estimated Stage 1 probabilities. Probability uncertainty carries into the final tail estimate.
- Readout mitigation effects were inconclusive on Stage 3; QAOA mitigation primarily raised feasibility, with overlapping approximation-ratio intervals.
- Simulator noise profiles are not interchangeable with hardware execution. The Stage 3 exported backend-calibration simulation used Kingston calibration and is not a Marrakesh calibration result.
- No quantum advantage is demonstrated.

See [`PROJECT_STATE.md`](PROJECT_STATE.md) for the chronological work log and exact experiment history.
