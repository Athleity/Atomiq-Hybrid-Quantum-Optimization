# Architecture and stage boundaries

The repository retains its established flat `src/atomiq` package paths; no source moves are required to understand the pipeline.

## Stage 1

- `schema.py` defines the allowed training data source and feature boundary.
- `preprocessing.py`, `estimators.py`, `calibration.py`, `metrics.py`, and `evaluation.py` implement fold-local modeling and evaluation.
- `scripts/run_phase1_classical.py` writes cross-validation evidence and the provisional model under `results/stage1_classical/`.
- `scripts/audit_phase1_results.py` recomputes persisted metrics independently.

## Stage 2

- `stage2.py` contains candidate/problem data structures, operational-data joining, objective/feasibility checks, exhaustive oracle, QUBO audit representation, and independent MILP.
- `stage2_quantum.py` implements the standard Dicke-state, fixed-cardinality XY-QAOA control and Aer helpers.
- `stage2_if_qaoa.py` implements integer-slack QPE indicators and controlled objective phase as an IF-QAOA adaptation.
- `stage2_iws_xy.py` implements an exact-K weighted Johnson-graph adaptation with Boltzmann marginal updates in compressed-sector simulation.
- `stage2_dla_warm_start.py` computes numerical DLA closures for restricted/expanded exact-K ansatz generators, then pretrains, transfers and fine-tunes parameters.
- `stage2_qtg_qaoa.py` simulates biased branching and the AAM rank-one mixer over feasible exact-K states. Its suffix predicate remains classical and is not a gate-level QTG circuit.
- `stage2_development_fixtures.py` provides a shared four-item synthetic instance with both resource constraints binding.
- `stage2_qiskit_circuits.py` builds symbolic Qiskit standard QAOA and exact-K XY-QAOA ansätze and fits angles with COBYLA/SPSA and expectation/CVaR objectives.
- `qiskit_noise.py` builds the explicit illustrative Aer NoiseModel, physical line transpilation, measurement-window relaxation, assignment calibration, readout correction, bootstrap uncertainty, and resource reports.
- `stage3_quantum.py` prepares six independent Bernoulli qubits and provides an independent Poisson-binomial reference and Qiskit bitstring decoder.
- Scripts and YAML configs are explicit experiment entry points; each result folder records whether its candidate pool is synthetic or development-only.

The specialized Stage 2 modules are research adaptations, not challenge-scale benchmarks. IWS/DLA remain compressed-sector mathematical simulations; QTG's feasibility predicate is classical; IF has a Qiskit circuit but no Aer finite-shot/noise run. Stage 3 has a runnable circuit but currently uses provisional training-only OOF probabilities. Keep all results separate from Stage 1 evidence.

## Data flow and trust boundaries

Stage 1 creates out-of-fold predictions from training rows only. Stage 2 callers explicitly supply a ten-row candidate/probability table, which joins to operational data by exact `customer_id`; IDs are not model features. Challenge test labels and leaked workbook aggregate target counts are outside the permitted data flow. Exact enumeration and MILP are independent decision references for each named ten-candidate pool.
