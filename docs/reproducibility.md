# Reproducibility

## Runtime

The lock was resolved and validated for CPython 3.12.14, Windows x86-64. Use the instructions in `docs/environment.md`; the committed Python pin is `>=3.12,<3.13`. GitHub Actions installs the full `requirements-lock.txt` and runs `python -m pytest -q` on Windows.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_phase1_classical.py
.\.venv\Scripts\python.exe scripts\audit_phase1_results.py --output results\stage1_classical\independent_audit.json
.\.venv\Scripts\python.exe scripts\run_stage2_oracle_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_quantum_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_if_qaoa_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_iws_binding_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_dla_warm_start_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_qtg_qaoa_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_xy_binding_dev.py
.\.venv\Scripts\python.exe scripts\run_qiskit_stage2_noise.py
.\.venv\Scripts\python.exe scripts\complete_qiskit_resource_reports.py
.\.venv\Scripts\python.exe scripts\run_qiskit_stage2_ten_candidate.py
.\.venv\Scripts\python.exe scripts\run_qiskit_stage3_noise.py
.\.venv\Scripts\python.exe scripts\audit_qiskit_noise_results.py
```

Stage 1 seeds and search budgets are stored in `configs/phase1_classical.yaml`. Stage 2/3 settings are in corresponding YAML files under `configs/`. The Qiskit noise runs require Qiskit 2.6.2/Aer 0.18.2 from the unchanged lockfile. `qiskit_circuit_noise.yaml` defines the assumed noise profile, shot counts, seeds and optimizer budgets. Every result records optimizer termination and the profile assumptions. The local Aer noise profile is hypothetical, not hardware-calibrated; its outputs are simulation-only. A later, separate Colab run did execute Stage 3 and a synthetic QAOA comparison on IBM hardware; use `reports/ibm_hardware_execution_audit.md` for that evidence and its provenance correction. Capped fits must not be described as converged.

## Result interpretation

Use `PROJECT_STATE.md` and the relevant report to determine whether an artifact is a final benchmark, a provisional development experiment, a superseded run, or an interrupted diagnostic. Development data and synthetic constraints are never challenge test results. Random seeds support repeatability but do not remove sampling or small-sample uncertainty.
# Reproducibility additions for the completed IBM hardware batch

The hardware notebook is `notebooks/atomiq_ibm_hardware_experiments_QAOA_HARDWARE_EXPANDED.ipynb`. It is Colab-oriented and requests the `IBM_QUANTUM_API_KEY` secret; `IBM_QUANTUM_INSTANCE` is optional. Outputs are cleared in the delivered notebook. The completed hardware evidence is saved under `results/ibm_hardware/` and was not rerun during final publication.

The saved Colab run used Python 3.13.18, Qiskit 2.2.3, Aer 0.17.2, and IBM Runtime 0.43.1 (seed 2026). These differ from the repository's locked local test/simulation environment. The hardware run's source export has a corrected Stage 3 resume-provenance issue; read `reports/ibm_hardware_execution_audit.md` before interpreting its backend/qubit metadata. In particular, do not use its Kingston backend calibration as Stage 3 Marrakesh calibration.

The public repository omits the original challenge PDF/workbook pending redistribution authorization. For data-dependent reproduction, obtain those inputs through the authorized competition channel and place them in `data/raw/` following its README. The tests do not require rerunning the long benchmark or submitting hardware jobs.
