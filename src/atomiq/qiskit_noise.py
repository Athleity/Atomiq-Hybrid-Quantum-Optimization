"""Reproducible, explicitly illustrative Aer backend and readout mitigation.

This profile is a model assumption, not calibration from a real processor.
It describes a seven-qubit line with heterogeneous T1/T2, gate errors,
durations, and asymmetric readout assignment. All times are in seconds in the
profile and converted to seconds by Aer error constructors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Mapping

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


@dataclass(frozen=True)
class IllustrativeBackendProfile:
    """Hypothetical small superconducting backend; values are not calibration."""

    name: str = "illustrative_7q_line_v1"
    physical_qubits: int = 7
    t1_seconds: tuple[float, ...] = (72e-6, 61e-6, 83e-6, 68e-6, 76e-6, 57e-6, 89e-6)
    t2_seconds: tuple[float, ...] = (54e-6, 47e-6, 63e-6, 51e-6, 58e-6, 43e-6, 66e-6)
    readout_0_to_1: tuple[float, ...] = (0.018, 0.026, 0.021, 0.031, 0.017, 0.029, 0.022)
    readout_1_to_0: tuple[float, ...] = (0.034, 0.028, 0.041, 0.036, 0.025, 0.044, 0.032)
    one_qubit_depolarizing: float = 0.0004
    two_qubit_depolarizing: float = 0.008
    sx_duration_seconds: float = 35e-9
    x_duration_seconds: float = 70e-9
    cx_duration_seconds: float = 300e-9
    measurement_duration_seconds: float = 800e-9

    def __post_init__(self) -> None:
        n = self.physical_qubits
        if n < 2 or any(len(values) != n for values in (
            self.t1_seconds, self.t2_seconds, self.readout_0_to_1, self.readout_1_to_0
        )):
            raise ValueError("Per-qubit profile arrays must match physical_qubits (at least two).")
        if any(t1 <= 0 or t2 <= 0 or t2 > 2 * t1 for t1, t2 in zip(self.t1_seconds, self.t2_seconds)):
            raise ValueError("Each qubit must satisfy T1 > 0 and 0 < T2 <= 2*T1.")
        if any(not 0 <= p < 0.5 for p in (*self.readout_0_to_1, *self.readout_1_to_0)):
            raise ValueError("Readout assignment probabilities must be in [0, 0.5).")
        if not (0 <= self.one_qubit_depolarizing < 1 and 0 <= self.two_qubit_depolarizing < 1):
            raise ValueError("Depolarizing probabilities must be in [0, 1).")
        if min(self.sx_duration_seconds, self.x_duration_seconds,
               self.cx_duration_seconds, self.measurement_duration_seconds) <= 0:
            raise ValueError("All assumed gate and measurement durations must be positive.")

    @property
    def coupling_map(self) -> CouplingMap:
        return CouplingMap.from_line(self.physical_qubits, bidirectional=True)

    @property
    def basis_gates(self) -> list[str]:
        return ["id", "rz", "sx", "x", "cx"]

    def readout_assignment(self, qubit: int) -> np.ndarray:
        """Matrix A[observed, true], preserving directional asymmetry."""
        p01, p10 = self.readout_0_to_1[qubit], self.readout_1_to_0[qubit]
        return np.array([[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=float)

    def readout_only_noise_model(self) -> NoiseModel:
        """Readout-only model used to calibrate assignment matrices without X-gate bias."""
        model = NoiseModel(basis_gates=self.basis_gates)
        for q in range(self.physical_qubits):
            model.add_readout_error(ReadoutError(self.readout_assignment(q).T.tolist()), [q])
        return model

    def noise_model(self, *, gate_scale: float = 1.0) -> NoiseModel:
        """Add depolarizing and duration-dependent relaxation to the readout channel."""
        if gate_scale < 0:
            raise ValueError("gate_scale must be nonnegative.")
        model = NoiseModel(basis_gates=self.basis_gates)
        for q in range(self.physical_qubits):
            for gate, duration, p in (
                ("sx", self.sx_duration_seconds, self.one_qubit_depolarizing),
                ("x", self.x_duration_seconds, self.one_qubit_depolarizing),
                # An explicit id immediately before measurement models the
                # assumed acquisition window's relaxation before assignment error.
                ("id", self.measurement_duration_seconds, 0.0),
            ):
                relaxation = thermal_relaxation_error(
                    self.t1_seconds[q], self.t2_seconds[q], duration
                )
                error = depolarizing_error(min(p * gate_scale, 0.999999), 1).compose(relaxation)
                model.add_quantum_error(error, gate, [q])
            model.add_readout_error(ReadoutError(self.readout_assignment(q).T.tolist()), [q])
        for q in range(self.physical_qubits - 1):
            r0 = thermal_relaxation_error(self.t1_seconds[q], self.t2_seconds[q], self.cx_duration_seconds)
            r1 = thermal_relaxation_error(self.t1_seconds[q + 1], self.t2_seconds[q + 1], self.cx_duration_seconds)
            relaxation_2q = r0.tensor(r1)
            error_2q = depolarizing_error(
                min(self.two_qubit_depolarizing * gate_scale, 0.999999), 2
            ).compose(relaxation_2q)
            model.add_quantum_error(error_2q, "cx", [q, q + 1])
            model.add_quantum_error(error_2q, "cx", [q + 1, q])
        return model

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update({
            "calibration_source": None,
            "calibration_timestamp": None,
            "is_hardware_calibrated": False,
            "topology": "bidirectional linear chain",
            "time_unit": "seconds",
            "readout_convention": "P(observed | true), asymmetric 0->1 and 1->0",
            "measurement_relaxation_model": "thermal relaxation on an id gate lasting measurement_duration_seconds, followed by classical assignment error",
        })
        return result


def illustrative_profile_for_size(physical_qubits: int) -> IllustrativeBackendProfile:
    """Create an explicitly synthetic line profile of the requested width.

    Widths above seven repeat the seven assumed per-qubit values cyclically;
    this is a transparent simulator convenience, never a proxy for retrieved
    device calibration.
    """
    if physical_qubits < 2:
        raise ValueError("At least two physical qubits are required.")
    base = IllustrativeBackendProfile()
    def extend(values):
        return tuple(values[index % len(values)] for index in range(physical_qubits))
    return replace(
        base,
        name=f"illustrative_{physical_qubits}q_line_repeated_v1" if physical_qubits > 7 else base.name,
        physical_qubits=physical_qubits,
        t1_seconds=extend(base.t1_seconds),
        t2_seconds=extend(base.t2_seconds),
        readout_0_to_1=extend(base.readout_0_to_1),
        readout_1_to_0=extend(base.readout_1_to_0),
    )


def transpile_for_profile(
    circuit: QuantumCircuit,
    profile: IllustrativeBackendProfile,
    *,
    seed: int,
    optimization_level: int = 1,
) -> QuantumCircuit:
    if circuit.num_qubits > profile.physical_qubits:
        raise ValueError("Circuit exceeds the illustrative backend's physical-qubit count.")
    return transpile(
        circuit,
        basis_gates=profile.basis_gates,
        coupling_map=profile.coupling_map,
        initial_layout=list(range(circuit.num_qubits)),
        seed_transpiler=seed,
        optimization_level=optimization_level,
    )


def resource_report(circuit: QuantumCircuit, profile: IllustrativeBackendProfile, *, seed: int) -> dict[str, object]:
    """Compile to a physical line and include explicit routing-swap accounting."""
    started = perf_counter()
    routed = transpile(
        circuit,
        basis_gates=[*profile.basis_gates, "swap"],
        coupling_map=profile.coupling_map,
        initial_layout=list(range(circuit.num_qubits)),
        seed_transpiler=seed,
        optimization_level=0,
    )
    executable = transpile_for_profile(circuit, profile, seed=seed, optimization_level=1)
    counts = {str(gate): int(count) for gate, count in executable.count_ops().items()}
    one_q_names = {"id", "rz", "sx", "x", "h", "ry", "rx"}
    two_q_names = {"cx", "cz", "swap", "rxx", "ryy"}
    return {
        "logical_qubits": circuit.num_qubits,
        "physical_qubits_available": profile.physical_qubits,
        "physical_qubits_used": circuit.num_qubits,
        "initial_layout_logical_to_physical": list(range(circuit.num_qubits)),
        "coupling_map": [list(edge) for edge in profile.coupling_map.get_edges()],
        "basis_gates": profile.basis_gates,
        "transpiled_depth": int(executable.depth()),
        "transpiled_total_gates": int(sum(counts.values())),
        "transpiled_one_qubit_gates": int(sum(v for k, v in counts.items() if k in one_q_names)),
        "transpiled_two_qubit_gates": int(sum(v for k, v in counts.items() if k in two_q_names)),
        "transpiled_gate_counts": counts,
        "routing_swap_count_before_basis_translation": int(routed.count_ops().get("swap", 0)),
        "routing_transpiled_depth_before_basis_translation": int(routed.depth()),
        "compile_runtime_seconds": perf_counter() - started,
        "optimization_level": 1,
        "seed_transpiler": seed,
    }


def add_measurement_window(circuit: QuantumCircuit) -> QuantumCircuit:
    """Add explicit id relaxation exposure before measurement; preserve circuit in-place semantics."""
    measured = (
        circuit.remove_final_measurements(inplace=False)
        if any(instruction.operation.name == "measure" for instruction in circuit.data)
        else circuit.copy()
    )
    for q in range(measured.num_qubits):
        measured.id(q)
    measured.barrier()
    measured.measure_all()
    return measured


def run_aer_counts(
    circuit: QuantumCircuit,
    *,
    profile: IllustrativeBackendProfile,
    shots: int,
    seed: int,
    noise: bool,
    gate_scale: float = 1.0,
) -> tuple[dict[str, int], float, QuantumCircuit]:
    if shots < 1:
        raise ValueError("shots must be positive.")
    measured = add_measurement_window(circuit)
    noise_model = profile.noise_model(gate_scale=gate_scale) if noise else None
    # Level 0 preserves explicit pre-measurement `id` instructions; these carry
    # the declared acquisition-window T1/T2 exposure in the Aer model.
    compiled = transpile_for_profile(measured, profile, seed=seed, optimization_level=0)
    simulator = AerSimulator(method="statevector", noise_model=noise_model)
    started = perf_counter()
    counts = simulator.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
    runtime = perf_counter() - started
    return {str(k): int(v) for k, v in counts.items()}, runtime, compiled


def _bitstring_vector(counts: Mapping[str, int], n_qubits: int) -> np.ndarray:
    vector = np.zeros(1 << n_qubits, dtype=float)
    total = 0
    for bits, count in counts.items():
        cleaned = bits.replace(" ", "")
        if len(cleaned) != n_qubits or set(cleaned) - {"0", "1"}:
            raise ValueError(f"Invalid {n_qubits}-qubit count key: {bits!r}")
        vector[int(cleaned, 2)] += int(count)
        total += int(count)
    if total <= 0:
        raise ValueError("Cannot normalize empty counts.")
    return vector / total


def calibration_assignment_matrices(
    n_qubits: int,
    *,
    profile: IllustrativeBackendProfile,
    shots: int,
    seed: int,
) -> tuple[list[np.ndarray], float]:
    """Estimate each physical qubit's 2x2 readout matrix from Aer calibration circuits."""
    if not 1 <= n_qubits <= profile.physical_qubits:
        raise ValueError("Calibration qubit count must fit the profile.")
    simulator = AerSimulator(method="statevector", noise_model=profile.readout_only_noise_model())
    matrices: list[np.ndarray] = []
    started = perf_counter()
    for q in range(n_qubits):
        columns = []
        for prepared in (0, 1):
            circuit = QuantumCircuit(n_qubits, n_qubits)
            if prepared:
                circuit.x(q)
            circuit.measure(range(n_qubits), range(n_qubits))
            compiled = transpile_for_profile(circuit, profile, seed=seed, optimization_level=0)
            counts = simulator.run(compiled, shots=shots, seed_simulator=seed + q + prepared).result().get_counts()
            probs = _bitstring_vector(counts, n_qubits)
            observed_one = sum(probs[index] for index in range(1 << n_qubits) if (index >> q) & 1)
            columns.append([1.0 - observed_one, observed_one])
        matrices.append(np.asarray(columns, dtype=float).T)
    return matrices, perf_counter() - started


def assignment_matrix(matrices: list[np.ndarray]) -> np.ndarray:
    """Build A[observed integer, true integer] with Qiskit's little-endian qubits."""
    if not matrices:
        raise ValueError("At least one per-qubit assignment matrix is required.")
    result = matrices[-1]
    for matrix in reversed(matrices[:-1]):
        result = np.kron(result, matrix)
    return result


def project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection of a mitigation quasi-distribution onto probabilities."""
    vector = np.asarray(values, dtype=float)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(vector) + 1)
    valid = ordered - cumulative / indices > 0
    rho = int(np.flatnonzero(valid)[-1])
    threshold = cumulative[rho] / (rho + 1)
    return np.maximum(vector - threshold, 0.0)


def mitigate_readout(
    counts: Mapping[str, int],
    matrices: list[np.ndarray],
) -> dict[str, object]:
    n = len(matrices)
    observed = _bitstring_vector(counts, n)
    matrix = assignment_matrix(matrices)
    quasi = np.linalg.lstsq(matrix, observed, rcond=None)[0]
    projected = project_probability_simplex(quasi)
    return {
        "quasi_distribution": quasi,
        "projected_distribution": projected,
        "raw_distribution": observed,
        "condition_number": float(np.linalg.cond(matrix)),
        "negative_quasi_mass": float(-quasi[quasi < 0].sum()),
    }


def bootstrap_metric_variance(
    counts: Mapping[str, int],
    matrices: list[np.ndarray],
    metric,
    *,
    seed: int,
    replicates: int = 200,
) -> dict[str, object]:
    """Parametric bootstrap variance for raw and mitigated metric estimates."""
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    n = len(matrices)
    p_raw = _bitstring_vector(counts, n)
    total = sum(counts.values())
    a = assignment_matrix(matrices)
    # Factor the channel once. Re-solving this 2**n x 2**n system for every
    # bootstrap replicate changes the asymptotic cost from O(B d^2) to O(B d^3).
    inverse = np.linalg.inv(a)
    rng = np.random.default_rng(seed)
    started = perf_counter()
    draws = rng.multinomial(total, p_raw, size=replicates)
    raw = [float(metric(draw / total)) for draw in draws]
    estimates = (draws / total) @ inverse.T
    mitigated = [float(metric(project_probability_simplex(estimate))) for estimate in estimates]
    bootstrap_runtime = perf_counter() - started
    var_raw = float(np.var(raw, ddof=1))
    var_mitigated = float(np.var(mitigated, ddof=1))
    return {
        "bootstrap_replicates": replicates,
        "raw_metric_variance": var_raw,
        "mitigated_metric_variance": var_mitigated,
        "mitigation_variance_ratio": var_mitigated / var_raw if var_raw > 0 else None,
        "bootstrap_runtime_seconds": bootstrap_runtime,
    }
