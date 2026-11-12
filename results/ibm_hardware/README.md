# IBM hardware evidence (latest Colab batch)

`source_export_unmodified.zip` is the exact supplied Colab export; its SHA-256 is recorded in `provenance_correction.json`. `source_export_contents/` is its unpacked copy for direct inspection. The Stage 3 attribution defect in that source is preserved and explained; do not use its original `job1_stage3_best_layout` backend/qubit fields as canonical. The canonical extracted copy has those metadata fields corrected, while the raw measurement counts, job ID and derived Stage 3 estimate remain unchanged.

## Verified execution

- Stage 3 job `dbqfi99abdes73fo6vig`: completed on `ibm_marrakesh`, physical qubits `[87, 6, 10, 139, 120, 107]`, 8,192 measured experiment shots, 5.0 reconciled QPU seconds. The export's resumed record incorrectly relabelled this job with Kingston's currently selected backend/layout.
- Synthetic six-candidate exact-cardinality XY-QAOA job `dbqg5d1abdes73fo7o10`: completed on `ibm_kingston`, qubits `[147, 148, 149, 150, 151, 152]`, 30,000 shots per depth (p=1 and p=2) plus 8,000 shots per calibration circuit, 26.0 QPU seconds.
- Batch QPU usage: 31.0 seconds reconciled across both jobs.
- The QAOA objective/parameters were optimized noise-free in simulation and then evaluated on hardware; no hardware-in-loop optimization was performed. The QAOA instance is synthetic and is not the actual bank-customer selection instance.

Canonical tables, counts, calibration, transpiled resource data and figures are in `source_export_contents/processed`, `raw` and `figures`. Read `../reports/ibm_hardware_execution_audit.md` for the independent interpretation and exact metrics.
