# Qiskit circuit and noise evaluation

**Run status:** completed local Qiskit/Aer diagnostics, not hardware experiments. **Data status:** Stage 2 uses a synthetic binding fixture plus one provisional ten-candidate pool built only from Stage1_Train calibrated OOF predictions. Stage 3 uses the six probabilities in that same provisional selection artifact. Stage1_Test, its labels, and test-derived artifacts were not read.

## Runtime and simulation model

The existing environment was reused without installing or upgrading packages: CPython 3.12.14, Qiskit 2.6.2, Qiskit Aer 0.18.2. `requirements.in` and `requirements-lock.txt` are unchanged. Aer `AerSimulator(method="statevector")` is used directly so each experiment controls the explicit noise model, coupling-aware transpilation, simulator seed, and raw counts. A primitive `SamplerV2` abstraction was not needed for this backend-specific run; direct Aer execution exposes the hardware-style noise and physical compilation settings being measured.

No real backend calibration was available or retrieved. `IllustrativeBackendProfile` is a declared, hypothetical seven-qubit superconducting line profile; it is **not hardware-calibrated** and has no backend name or calibration timestamp. Its assumptions are:

| Quantity | Assumption |
|---|---:|
| T1 per qubit | 57–89 microseconds; explicit vector in each JSON |
| T2 per qubit | 43–66 microseconds; explicit vector in each JSON, each <= 2*T1 |
| 1-qubit depolarization | 0.0004 per `sx` and `x` |
| 2-qubit depolarization | 0.008 per `cx` |
| Gate times | `sx` 35 ns, `x` 70 ns, `cx` 300 ns |
| Measurement window | 800 ns, represented by a pre-measurement `id` relaxation exposure |
| Readout assignment | Per-qubit asymmetric 0->1 and 1->0 vectors, each recorded in JSON |
| Connectivity | Bidirectional linear chain; fixed initial layout maps logical qubit i to physical qubit i |

Thermal relaxation is composed after the assumed noisy gate channels using the declared gate duration; the measurement exposure receives relaxation before the classical assignment error. `rz` is treated as virtual and has no assigned duration or gate error. Idle periods between gates are not separately scheduled or modelled. Ten-qubit Stage 2 uses a 10-qubit line profile that repeats the seven assumed qubit values cyclically for qubits 7–9. This extension is another explicit synthetic assumption, not calibration.

The Aer assignment calibration runs prepared-0 and prepared-1 circuits with readout errors enabled and gate errors disabled, so the estimated per-qubit matrix isolates the configured assignment channel. Mitigation uses tensor-product assignment-matrix inversion, reports the raw quasi-distribution, and projects it onto the probability simplex for physical probability metrics. This corrects local readout assignment only. Parametric bootstrap compares the variance of raw and mitigated feasibility estimates; it does not correct gate errors or model uncertainty.

## Parameterized circuit implementation

`src/atomiq/stage2_qiskit_circuits.py` builds symbolic `QuantumCircuit`s with `ParameterVector` angles:

- Standard QAOA starts in `|+>^n`, applies a basis-state-selective diagonal penalty/objective phase, and uses the all-qubit RX mixer. Its classical cost penalizes wrong cardinality and both resource inequalities.
- Exact-K XY-QAOA starts in an exact Dicke state and applies RXX/RYY gates on a path, cycle, or complete graph. It preserves the selected count exactly. If every state in the exact-K sector is feasible, the additive objective phase simplifies to one-qubit phase gates; when inequalities bind, the implementation applies the full feasibility-penalty diagonal oracle.
- The circuit angles are parameterized and evaluated with Qiskit `Statevector`. COBYLA and a seeded SPSA implementation are compared at a matched near-equal function-evaluation budget. Expectation and lower-tail CVaR cost objectives are both implemented.
- A depth-two XY run is seeded by linear interpolation of a depth-one fit. A noisy SPSA run also trains at finite shots; its endpoint is compared with ideal-trained angles.

The synthetic binding fixture has four variables, exact cardinality two, three feasible choices out of six, and exact optimum 39.5 at SYN0/SYN1. At depth one, one-seed, 2,048-shot results show substantial simulated circuit-noise effects: XY ideal feasibility is 0.515–0.701 for the fitted variants, while noisy Aer feasibility is 0.179–0.193. Readout projection moves noisy feasibility only to 0.184–0.199. The 100-replicate bootstrap variance ratio (mitigated/raw) is 1.35–1.39, so mitigation **increases variance** in this experiment.

The optimization endpoints are budget-truncated: COBYLA was limited to 12 objective calls; SPSA used 10 calls under the comparable cap. All `success` flags are false and must not be called converged. At this small budget, XY/CVaR COBYLA found training CVaR cost -39.50; XY/CVaR SPSA reached -38.02. XY/expectation training costs were 885.45 (COBYLA) and 856.28 (SPSA). The standard-QAOA expectation run ended at penalized cost 1,347.57. These in-sample optimizer values are diagnostic and not held-out performance estimates.

At the same depth-one parameter point, XY mixer connectivity resource reports on the seven-qubit line were:

| Mixer graph | Routed SWAPs before basis translation | Final depth | CX count |
|---|---:|---:|---:|
| Path | 167 | 1,091 | 691 |
| Cycle | 168 | 1,106 | 701 |
| Complete | 170 | 1,124 | 712 |

The differences are small because the generic exact diagonal cost oracle dominates routing. These counts are tied to the recorded Qiskit version, basis, line map, initial layout, optimization level, and transpiler seed. Complete connectivity gives a denser logical mixer at greater routed depth and two-qubit cost.

The depth-two interpolated run and noisy SPSA fit are recorded in the JSON, along with per-circuit depths/gate counts, shot counts, raw bitstrings, objective gap, optimal-solution sampling probability, calibration runtime, mitigation runtime, and bootstrap variance.

## Provisional ten-candidate exact-six circuit

`scripts/run_qiskit_stage2_ten_candidate.py` executes a 10-qubit, K=6 parameterized XY circuit on the existing training-only candidate pool. It reuses seed-11 angles from the earlier exact-statevector fit, whose phase/mixer convention is covered by circuit tests. The Qiskit circuit's ideal statevector expected objective is 6.92695 against the exact oracle value 9.54561; exact-optimum probability is 0.02592. At 2,048 ideal Aer shots, expected objective is 6.95435 and optimum sampling is 0.02490.

The same circuit, transpiled to the illustrative 10-qubit line, has depth 6,010, 7,089 total gates, 3,081 CX gates, and 981 routing SWAPs before basis translation. Noisy Aer required 27.9 seconds for 512 shots in the recorded run. It measured feasibility 0.1230, conditional-on-feasible expected objective 4.9425, and sampled the exact optimum zero times; the best feasible observed objective was 9.08264. Readout projection changed feasibility to 0.1267 and conditional objective to 4.9406, still sampled the optimum zero times. The 200-replicate feasibility variance ratio was 1.437 (mitigated/raw): variance increased. All 210 exact-six subsets in this particular pool are feasible, so this run tests circuit compilation/noise but cannot test binding-constraint behavior. The candidate pool is provisional, not the challenge's final top ten.

## Stage 3 six-qubit circuit

`src/atomiq/stage3_quantum.py` prepares six independent Bernoulli qubits with `RY(2 asin(sqrt(p_i)))`, measures q_i into c_i, and decodes Qiskit strings in displayed `q5...q0` order. Its probabilities are read from the existing exact Stage 2 selection artifact and are explicitly marked as calibrated Stage1_Train OOF values, not a frozen model's challenge predictions. Marginal-probability and bit-order tests are in `tests/test_stage3_quantum.py`.

For the six provisional selected customers, the exact Poisson-binomial probability is `P(K >= 3) = 0.05426956001511194`. The Qiskit statevector result is `0.05426956001511196`, differing by `2.1e-17`. At 2,048 shots across seeds 11, 23, and 42:

| Seed | Ideal Aer | Noisy Aer | Readout-projected noisy estimate |
|---:|---:|---:|---:|
| 11 | 0.04492 | 0.07031 | 0.05974 |
| 23 | 0.05029 | 0.07080 | 0.05994 |
| 42 | 0.05127 | 0.07129 | 0.06033 |

At seed 11, the additional shot sweep gave ideal/noisy/mitigated estimates 0.04102/0.06250/0.05411 at 512 shots and 0.05591/0.06995/0.05792 at 8,192 shots. Readout projection removes part of the positive noise bias, but at 2,048 shots it remains above the exact value. The mitigated/raw bootstrap variance ratio ranges from 1.27 to 1.37 across the Stage 3 runs: variance rises even where point bias decreases. This is a negative mitigation result on variance and should not be hidden.

## Interpretation and limits

- **Actual Qiskit/Aer executions:** standard QAOA, exact-K XY-QAOA, 10-qubit exact-six XY, Stage 3 Bernoulli circuit, Aer ideal/noisy shots, and readout-only calibration circuits.
- **Ideal statevector:** parameter fitting, exact distribution checks, and Stage 3 cross-check. Statevector is a mathematical ideal simulation.
- **Noisy simulation:** Aer trajectories under the declared synthetic profile. This is not a retrieved backend, a real device calibration, or a physical QPU result.
- **Classical mathematical simulations retained from prior Stage 2:** IWS-XY and DLA are compressed-sector simulations; QTG/AAM uses a classical memoized feasibility predicate; IF-QAOA has a Qiskit circuit and arithmetic/unitary tests but has not yet had a finite-shot/noisy run. Those methods were not duplicated or relabeled as hardware circuits here.
- Only one optimizer seed was used for the four-item optimizer comparison, and all endpoints hit evaluation budgets. Noise and readout effects dominate the small optimizer differences. No performance ranking, quantum advantage, or convergence claim follows.
- Stage 3 probabilities remain uncertain because Stage 1 is not frozen and the six values are repeated-OOF estimates. Noise mitigation cannot repair that uncertainty.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_qiskit_stage2_noise.py
.\.venv\Scripts\python.exe scripts\complete_qiskit_resource_reports.py
.\.venv\Scripts\python.exe scripts\run_qiskit_stage2_ten_candidate.py
.\.venv\Scripts\python.exe scripts\run_qiskit_stage3_noise.py
.\.venv\Scripts\python.exe scripts\audit_qiskit_noise_results.py
```

Configs and authoritative outputs:

- `configs/qiskit_circuit_noise.yaml`
- `results/stage2_qiskit_noise/qiskit_stage2_noise_results.json`
- `results/stage2_qiskit_ten_candidate/ten_candidate_exact_six_qiskit_results.json`
- `results/stage3_qiskit_noise/qiskit_stage3_noise_results.json`
- `results/qiskit_noise_independent_audit.json`

The saved-count audit independently re-decodes every raw noisy Stage 2/Stage 3 bitstring, recomputes feasibility, objective and threshold metrics, and compares those calculations with the experiment JSON. It saves full mitigated quasi and projected distributions so results can be re-evaluated without re-running Aer.

## Qiskit API references

- Aer [custom noise-model tutorial](https://qiskit.github.io/qiskit-aer/tutorials/3_building_noise_models.html) and [backend-noise tutorial](https://qiskit.github.io/qiskit-aer/tutorials/2_device_noise_simulation.html) describe quantum errors, classical readout errors, and duration-based thermal relaxation composition.
- Aer [`thermal_relaxation_error`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.thermal_relaxation_error.html) requires physically valid `T2 <= 2*T1`; the profile validates that bound.
- Aer [`ReadoutError`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.ReadoutError.html) defines its matrix as recorded outcome conditioned on true outcome. The implementation stores a transposed internal `A[observed,true]` matrix and validates it against Aer calibration circuits.
- Aer [`AerSimulator`](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator) supports noisy `statevector` trajectory simulation; experiments use that method with finite shot counts and simulator seeds.
- The [Qiskit transpiler guide](https://quantum.cloud.ibm.com/docs/en/qiskit/transpile) describes layout, routing and SWAP insertion against a coupling map; the recorded circuit reports use fixed initial layouts and declared line connectivity.
