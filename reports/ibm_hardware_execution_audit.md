# IBM Quantum hardware execution audit (12 November 2026)

## Scope and evidence hierarchy

This audit examines the latest supplied Colab output archive, `results/ibm_hardware/source_export_unmodified.zip`, and its extracted copy. The archive is retained byte-for-byte; its SHA-256 is in `results/ibm_hardware/provenance_correction.json`. Raw counts are unchanged. Canonical extracted records correct one metadata attribution defect described below. No experiment was rerun to prepare this submission.

Evidence order: raw per-circuit counts and IBM job IDs establish that circuits returned results; the contemporaneous submitted-job record establishes backend/layout; processed tables are derived summaries. The archive's resumed registry metadata is not authoritative where it conflicts with the original submission record.

## What actually ran

Two completed jobs are recorded:

- **Stage 3 hardware validation** — job `dbqfi99abdes73fo6vig`, backend `ibm_marrakesh`, physical qubits `[87, 6, 10, 139, 120, 107]`, 8,192 Stage 3 shots plus two 2,048-shot readout-calibration circuits; reconciled usage 5.0 QPU seconds. It prepared independent Bernoulli qubits from the six actual Stage 1 probabilities and estimated the threshold event.
- **Stage 2 QAOA hardware comparison** — job `dbqg5d1abdes73fo7o10`, backend `ibm_kingston`, physical qubits `[147, 148, 149, 150, 151, 152]`. It evaluated p=1 and p=2 with 30,000 shots each, and 8,000 shots for each prepared-zero/prepared-one calibration circuit at each depth; reconciled usage 26.0 QPU seconds.

Total usage reconciled in this export is 31.0 seconds. IBM hardware was actually used. The QAOA instance is a synthetic six-item choose-three objective, not the bank's customer selection. Parameters were optimized noise-free in simulation and evaluated on hardware; hardware-in-loop optimization was not run.

The environment recorded in the run metadata is Python 3.13.18, Qiskit 2.2.3, Qiskit Aer 0.17.2 and Qiskit IBM Runtime 0.43.1, seed 2026. This is distinct from the local project lock environment (Python 3.12/Qiskit 2.6.2/Aer 0.18.2). The notebook's setup cell contains its Colab pins. Do not infer that the local lock was used for the hardware run.

## Stage 3 result

The validated independent classical reference is `P(K >= 3) = 0.2078515973394121`. The Stage 3 hardware estimate is 0.201538 (8,192 shots; Wilson 95% interval 0.192992–0.210364; absolute error 0.006314). Readout mitigation gives 0.202275 (bootstrap interval 0.192653–0.211392; absolute error 0.005577). The paired estimate of absolute-error reduction is 0.000737 with interval −0.001289 to 0.001840, so the mitigation effect is inconclusive.

**Provenance correction:** the exported resume registry and processed Stage 3 table label this old job `ibm_kingston` and assign Kingston's current layout `[56,149,64,118,68,17]`. The original job submission evidence identifies the same job ID as `ibm_marrakesh` on `[87,6,10,139,120,107]`. The resume path populated backend/layout from the notebook's current selected backend, which is invalid for an already-submitted job. Canonical extracted records and notebook resume metadata now use the original backend/layout. The immutable source ZIP still exposes its original incorrect fields so reviewers can inspect exactly what was supplied. Stage 3 counts and estimator are preserved; Kingston-based Aer estimates and Kingston calibration tables are simulator/calibration evidence and must not be described as Marrakesh calibration.

## Stage 2 QAOA results

Hardware p=1: feasibility 0.8147 (95% interval 0.8110–0.8192), exact-optimum sampling probability 0.0164, approximation ratio 0.470624 (0.468091–0.473198). Readout-mitigated: feasibility 0.915758 (0.908133–0.923443), exact-optimum probability 0.017122, approximation ratio 0.472897 (0.470817–0.475306).

Hardware p=2: feasibility 0.7615 (0.756897–0.766284), exact-optimum sampling probability 0.005633, approximation ratio 0.623956 (0.622351–0.625588). Readout-mitigated: feasibility 0.882229 (0.873185–0.890208), exact-optimum probability 0.005379, approximation ratio 0.624478 (0.622821–0.626053).

The ideal statevector approximation ratios are 0.476255 for p=1 and 0.633196 for p=2. Thus p=2 has a higher approximation ratio but lower exact-optimum sampling probability; hardware noise reduces both ideal ratios. Readout mitigation raises measured feasibility, while the reported approximation-ratio changes are small and their intervals overlap. These results do not establish a reliable mitigation gain or quantum advantage. The exact classical optimum remains the reference.

The QAOA benchmark's state preparation is exact-cardinality, but the displayed feasibility metric refers to the additional synthetic resource constraints. Transpiled circuit counts and backend calibration evidence are in `source_export_contents/processed/transpiled_circuit_metrics.csv`, `transpiled_circuit_metrics_full.json`, and the `backend_*_calibration.csv` tables. These are IBM backend-derived values captured during that Colab session, not estimates invented for this report.

## Reproduction and interpretation

Open `notebooks/atomiq_ibm_hardware_experiments_QAOA_HARDWARE_EXPANDED.ipynb` in Colab. Add Colab Secrets named `IBM_QUANTUM_API_KEY`; `IBM_QUANTUM_INSTANCE` is optional if the service can identify the desired instance. The delivered notebook has outputs cleared and includes an explicit resume metadata map, so opening it does not itself claim a fresh run. Hardware execution remains guarded by confirmation and budget gates. The checked-in evidence is the completed historical run, not a rerun.

For the full classical and simulated pipeline, follow [`docs/reproducibility.md`](../docs/reproducibility.md) and the local locked environment. The original challenge PDF and workbook are deliberately absent from the public package because their redistribution terms are unverified; obtain them through the authorized competition channel and place them in `data/raw/` for reproduction. They are neither changed nor deleted from the local repository.

## Files to inspect

- `results/ibm_hardware/provenance_correction.json` — exact correction and source archive digest.
- `results/ibm_hardware/source_export_unmodified.zip` — byte-identical source export, with source metadata defect preserved.
- `results/ibm_hardware/source_export_contents/` — extracted evidence; canonical Stage 3 hardware attribution corrected.
- `reports/figures/ibm_hardware/` — figures supplied with the Colab results archive.
- `notebooks/atomiq_ibm_hardware_experiments_QAOA_HARDWARE_EXPANDED.ipynb` — source notebook with corrected resume metadata; outputs cleared to prevent stale output from contradicting the canonical audit.
