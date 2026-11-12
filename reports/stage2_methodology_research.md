# Stage 2 methodology, implementation, and development evidence

**Status: methods implemented as research adaptations; no Stage 2 challenge-scale benchmark.** Stage 1 remains provisional. No Stage1_Test sheet or label, leaked aggregate label count, or Stage1_Test-derived artifact was read in these experiments. Advanced-method experiments use a synthetic four-item fixture only; the earlier ten-candidate training-only pool was used solely for the original exact-oracle/XY development run.

## Problem and oracle

For candidate indicators `x_i`, the challenge objective is `sum_i x_i (p_i * expected_deposit_value_i - contact_cost_i)`, with exact cardinality six, contact budget at most 25, and capacity at most 22. `src/atomiq/stage2.py` independently enumerates all `C(10,6)=210` fixed-cardinality selections and solves a SciPy MILP with integer slack encoding. For the named provisional pool, enumeration and MILP agree at objective `9.54561244282308` for IDs C0073, C0094, C0130, C0159, C0217, C0329; every one of its 210 selections is feasible. It therefore cannot test binding-constraint algorithms and is not the challenge top ten.

Advanced methods share `four_item_binding_problem()` in `src/atomiq/stage2_development_fixtures.py`: four synthetic candidates, choose exactly two, budget <=3 and capacity <=3. Three of six exact-cardinality selections are feasible. Exhaustive enumeration gives objective 39.5 at SYN0/SYN1. This is deliberately small and not the challenge candidate pool.

## Method mappings and implementation limits

### IWS-XY exact-K adaptation

The source paper uses a one-hot weighted graph state/mixer and iterative warm start ([IWS-XY](https://arxiv.org/html/2604.02083)). `src/atomiq/stage2_iws_xy.py` generalizes the state space to the Johnson graph of exact-K subsets. Positive edge weights produce a connected weighted Laplacian whose unique ground vector has amplitudes proportional to square roots of product weights. The objective oracle uses feasibility-conditioned semantics: infeasible states receive zero cost; feasible states retain objective ordering. The update estimates marginals by ideal multinomial draws and applies a Boltzmann update with capped-simplex projection to preserve expected cardinality.

This changes the paper's one-hot graph and does not retain its shallow circuit guarantee; this is a dense compressed-sector simulation. On the shared six-state fixture, three seeds with four updates of 1,024 shots each produced final feasible sample rates around 0.995–0.997 and optimum sample rate around 0.992. COBYLA reached its 100-evaluation cap for all seeds (`optimizer_success=false`); the fitted angle search is truncated. The ten-item run has 210 sector states and is also compressed-sector simulation, not a compiled circuit. Evidence: `results/stage2_iws_binding_development/iws_xy_binding_development_results.json` and `results/stage2_iws_xy_development/iws_xy_development_results.json`.

### DLA-informed warm-start QAOA

The cited method restricts an ansatz, pretrains it, adds expressive generators and transfers parameters ([paper and supplement](https://arxiv.org/html/2505.18396), [published article](https://doi.org/10.1038/s41534-026-01192-4)). `src/atomiq/stage2_dla_warm_start.py` uses exact-K sector matrices for cycle XY/local Z restricted generators and all-pairs ZZ in the expanded ansatz. It numerically computes Lie closure with reorthogonalized Gram-Schmidt; on the four-item K=2 sector the restricted algebra dimension is 15 and expanded dimension is 36, equal to `dim(u(6))`. It runs 100 restricted and 100 full-ansatz Adam steps, transfers restricted parameters and initializes added ZZ parameters at zero. A random full-ansatz initialization receives the same full-stage budget as an ablation.

Across seeds 11/23/42, multi-angle warm-start expected feasible objectives were 38.176/35.222/38.370; matched random-initialized values were 38.364/38.370/38.369. Warm start does not consistently win. Shared-angle results cluster near 18.37 because the shared ansatz has only about 0.5 feasibility mass on this fixture. This is a synthetic exact-statevector experiment; no shot/noise circuit evaluation is included. DLA closure is numerical within the recorded tolerance and finite sector. Evidence: `results/stage2_dla_warm_start_development/dla_warm_start_development_results.json`.

### IF-QAOA adaptation

The method uses phase estimation to encode integer constraint indicators and condition objective phase on feasibility ([IF-QAOA](https://arxiv.org/html/2504.08663), [PRA publication](https://doi.org/10.1103/fb5m-cl9m)). `src/atomiq/stage2_if_qaoa.py` computes signed two's-complement registers for `limit - sum(a_i x_i)`, controls profit phase on both nonnegative-slack signs, and uncomputes both registers. Width is derived so all possible slacks fit without wraparound. Exact cardinality comes from Dicke preparation and an XY mixer. Tests verify positive/negative integer slack, selected phase versus infeasible identity phase, uncomputation and exact-K support.

On the common fixture, the 12-qubit circuit (4 selectors plus 8 indicator qubits) transpiled to depth 334, 253 CX, 492 RZ and 94 SX in the recorded basis. Seed 11 reached its 100-evaluation cap with expected feasible objective 19.6724, feasible probability 0.6023 and optimum probability 0.2831. This is an incomplete optimizer diagnostic, not performance evidence. No finite-shot, noise or physical hardware experiment was run. Evidence: `results/stage2_if_qaoa_development/if_qaoa_development_results.json`.

### QTG / AAM-QAOA adaptation

The QTG/AAM formulation uses biased-Hadamard branching, a feasibility-preserving tree transition, and a conjugated rank-one mixer ([QTG/AAM-QAOA](https://arxiv.org/html/2411.00518); [original QTG](https://arxiv.org/html/2310.06623)). `src/atomiq/stage2_qtg_qaoa.py` builds a state over exact-K assignments with a feasible suffix under both resources, then applies the AAM rank-one reflection/phase and objective phase in the feasible subspace. Tests validate support, normalization, unitarity and reproducibility. All sampled states on the common fixture are feasible.

Critical limitation: suffix feasibility is a memoized **classical dynamic program** in the simulator. This is a high-level exact-amplitude adaptation, not a reversible circuit. Reversible lookahead/resource accumulators and gate synthesis are unimplemented; reported logical-qubit count is a lower-bound estimate, not a compiled count. At depth one, three seeds x biases 0 and 1, 4,096 ideal shots; all six COBYLA runs hit the 80-evaluation cap. Bias 0 gave expected objectives 37.642–37.845 and optimum probabilities 0.764–0.768; bias 1 gave 36.007–36.230 and 0.515–0.519. These are truncated synthetic simulation outcomes. Evidence: `results/stage2_qtg_qaoa_development/qtg_qaoa_development_results.json`.

### Standard XY-QAOA control

`src/atomiq/stage2_quantum.py` remains a standard cycle-XY control, not one of the methods above. On the binding fixture it preserves exact cardinality but its penalty oracle does not preserve resource constraints. Across seeds, exact feasible probability is 0.810–0.871 and optimum probability 0.129–0.374. Ideal Aer draws agree approximately with exact probabilities. Under the explicitly illustrative, uncalibrated depolarizing/readout model, sampled feasibility falls to 0.558–0.589. This illustrates sensitivity for this tiny synthesized circuit and is not hardware evidence. Evidence: `results/stage2_xy_binding_development/xy_qaoa_binding_development_results.json`.

The original ten-candidate pool XY result is in `results/stage2_quantum_development/xy_qaoa_development_results.json`. Its all-feasible pool makes feasibility handling uninformative; do not compare numerically with the synthetic four-item results.

## Reproduction

From PowerShell at repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_stage2_oracle_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_quantum_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_xy_binding_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_iws_binding_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_dla_warm_start_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_if_qaoa_dev.py
.\.venv\Scripts\python.exe scripts\run_stage2_qtg_qaoa_dev.py
```

YAML files under `configs/` record seeds, depth, shots and optimizer limits. Each JSON captures the configuration and optimizer termination. Do not omit `optimizer_success=false` or describe max-evaluation exits as convergence. Runs use exact statevector/compressed-sector simulation; ideal finite shots are not physical sampling. Only standard XY includes an illustrative noise stress test.

## Authoritative boundaries and outstanding work

- Challenge objective/constraints and exact oracle: `src/atomiq/stage2.py`. Candidate inputs remain provisional.
- Shared synthetic binding fixture: `src/atomiq/stage2_development_fixtures.py`.
- Source modules, configs and raw JSONs above are authoritative for exact method variants and outcomes.
- Stage 1 artifacts remain in `results/stage1_classical/`; Stage 2 diagnostics are separate.
- No Stage1_Test or test-derived artifact was used; no final challenge top-ten selection was made.
- Optimizer budgets need controlled expansion and multi-depth comparison; methods need common objective-evaluation and shot budgets.
- QTG needs reversible two-resource/exact-K construction before gate-resource or noise claims. IWS needs circuit synthesis/resource analysis. IF and DLA need finite-shot/noise evaluations. Challenge-pool evaluations should wait until Stage 1 is frozen.
- No quantum advantage, asymptotic scalability or hardware suitability claim is supported.
