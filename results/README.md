# Results guide

Use these folders as the source of record. Reports interpret the files; the immutable IBM export is retained only for raw-evidence inspection and must be read with its correction note.

## Final actual-customer artifacts

- stage1_classical/: frozen model provenance, feature-only predictions, training OOF predictions, folds, metrics, calibration, tuning trials, and audits. stage1_test_predictions.csv contains model scores, not test outcomes.
- stage2_final/: frozen top ten, selected six, exact enumeration/MILP result, and actual-pool QAOA simulation diagnostic.
- stage3_final/: exact Poisson-binomial distribution, Qiskit statevector, Aer/noise/mitigation runs, and shot-count summaries.
- final_actual_independent_audit.json: independent final pipeline checks; test labels were not requested or used.

## Actual IBM hardware evidence

- ibm_hardware/provenance_correction.json: canonical Stage 3 backend/layout correction and immutable source archive digest.
- ibm_hardware/source_export_contents/: extracted Colab outputs. Stage 3 hardware run metadata is corrected to Marrakesh; QAOA job and Kingston Aer calibration remain associated with Kingston.
- ibm_hardware/source_export_unmodified.zip: exact supplied ZIP. It contains the original resumed Stage 3 attribution error; never treat that entry as canonical.

## Development-only and superseded evidence

Folders with development, interrupted, superseded, expanded_tuning, or qiskit_noise in their names are diagnostic, provisional, or superseded as explained in PROJECT_STATE.md and the relevant methods report. They are not pooled with final actual-customer benchmark metrics.
