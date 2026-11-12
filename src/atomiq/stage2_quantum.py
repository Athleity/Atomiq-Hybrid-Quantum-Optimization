"""Small-instance QAOA circuits and simulation helpers for Stage 2.

The natural cardinality-preserving baseline uses a Dicke initial state and an
XY mixer. Budget/capacity are evaluated by the exact diagonal cost oracle. If
they bind, this is explicitly a penalty-based XY-QAOA variant; it is not IF-QAOA.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from scipy.optimize import minimize

from atomiq.stage2 import Stage2Problem, enumerate_optimum, evaluate_selection


@dataclass(frozen=True)
class QAOAResult:
    method: str
    depth: int
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    exact_expected_objective: float
    exact_optimal_solution_probability: float
    exact_feasible_probability: float
    exact_best_supported_objective: float
    optimizer_success: bool
    optimizer_message: str
    evaluations: int


def _bit_count(value: int) -> int:
    return int(value).bit_count()


def _mask_to_selected(mask: int, problem: Stage2Problem) -> tuple[str, ...]:
    return tuple(
        candidate.customer_id
        for index, candidate in enumerate(problem.candidates)
        if (mask >> index) & 1
    )


def _state_and_costs(problem: Stage2Problem, penalty: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = len(problem.candidates)
    dimension = 1 << n
    amplitudes = np.zeros(dimension, dtype=complex)
    masks = [mask for mask in range(dimension) if _bit_count(mask) == problem.selected_count]
    if len(masks) != comb(n, problem.selected_count):
        raise AssertionError("Internal fixed-cardinality state-space size mismatch.")
    amplitudes[masks] = 1.0 / np.sqrt(len(masks))

    if penalty is None:
        net_range = sum(abs(candidate.expected_net_value) for candidate in problem.candidates)
        penalty = net_range + 1.0
    costs = np.zeros(dimension, dtype=float)
    for mask in range(dimension):
        selected = _mask_to_selected(mask, problem)
        evaluation = evaluate_selection(problem, selected)
        # Standard QAOA minimizes energy. The large penalty is only for the
        # inequalities; exact cardinality is enforced by the invariant sector.
        costs[mask] = -evaluation.objective
        if not evaluation.feasible and len(selected) == problem.selected_count:
            violation = max(0, evaluation.total_contact_cost - problem.budget_limit)
            violation += max(0, evaluation.total_capacity - problem.capacity_limit)
            costs[mask] += penalty * violation * violation
    return amplitudes, costs


def build_xy_qaoa_circuit(
    problem: Stage2Problem,
    gamma: Sequence[float],
    beta: Sequence[float],
    *,
    topology: str = "cycle",
) -> QuantumCircuit:
    """Build shared-angle XY-QAOA with exact Dicke preparation.

    The diagonal cost is represented exactly as a truth-table phase oracle. For
    this ten-qubit Stage 2 instance this is practical in Aer but expensive to
    synthesize to hardware gates; circuit resource reports retain both the
    logical oracle and decomposed circuit costs.
    """
    if len(gamma) != len(beta) or not gamma:
        raise ValueError("gamma and beta must have the same positive depth.")
    n = len(problem.candidates)
    if topology not in {"cycle", "path", "complete"}:
        raise ValueError("topology must be cycle, path, or complete.")
    amplitudes, costs = _state_and_costs(problem)
    fixed_sector_feasible = all(
        evaluate_selection(problem, _mask_to_selected(mask, problem)).feasible
        for mask in range(1 << n)
        if _bit_count(mask) == problem.selected_count
    )
    net_values = [candidate.expected_net_value for candidate in problem.candidates]
    qc = QuantumCircuit(n, name=f"stage2_xy_qaoa_{topology}_p{len(gamma)}")
    qc.append(StatePreparation(amplitudes), range(n))
    edges = [(i, i + 1) for i in range(n - 1)]
    if topology == "cycle":
        edges.append((n - 1, 0))
    elif topology == "complete":
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for layer_gamma, layer_beta in zip(gamma, beta, strict=True):
        if fixed_sector_feasible:
            # On the invariant K-sector all inequality indicators are one, so
            # the challenge objective is exactly additive. Apply its exact
            # one-qubit phase separator and omit only an irrelevant global phase.
            for index, net_value in enumerate(net_values):
                qc.p(float(layer_gamma) * net_value, index)
        else:
            phases = np.exp(-1.0j * layer_gamma * costs)
            qc.append(DiagonalGate(phases), range(n))
        # RXX(beta) RYY(beta) = exp[-i beta (XX+YY)/2], preserving Hamming weight.
        for first, second in edges:
            qc.rxx(float(layer_beta), first, second)
            qc.ryy(float(layer_beta), first, second)
    return qc


def exact_qaoa_distribution(problem: Stage2Problem, circuit: QuantumCircuit) -> np.ndarray:
    state = Statevector.from_instruction(circuit).data
    probabilities = np.abs(state) ** 2
    probabilities /= probabilities.sum()
    return probabilities


def optimize_xy_qaoa(
    problem: Stage2Problem,
    *,
    depth: int = 1,
    seed: int = 42,
    max_evaluations: int = 60,
    topology: str = "cycle",
) -> QAOAResult:
    """Optimize exact expected penalized objective using deterministic COBYLA."""
    if depth < 1 or max_evaluations < 1:
        raise ValueError("depth and max_evaluations must be positive.")
    rng = np.random.default_rng(seed)
    initial = rng.uniform(0.0, 2.0 * np.pi, 2 * depth)
    counts = 0
    optimum, _ = enumerate_optimum(problem)
    n = len(problem.candidates)
    objective_values = np.zeros(1 << n, dtype=float)
    feasible = np.zeros(1 << n, dtype=bool)
    for mask in range(1 << n):
        eval_result = evaluate_selection(problem, _mask_to_selected(mask, problem))
        objective_values[mask] = eval_result.objective
        feasible[mask] = eval_result.feasible

    def loss(parameters: np.ndarray) -> float:
        nonlocal counts
        counts += 1
        gamma = parameters[:depth]
        beta = parameters[depth:]
        circuit = build_xy_qaoa_circuit(problem, gamma, beta, topology=topology)
        probability = exact_qaoa_distribution(problem, circuit)
        return float(-np.dot(probability, objective_values) + 1e6 * np.dot(probability, ~feasible))

    result = minimize(
        loss,
        initial,
        method="COBYLA",
        bounds=[(0.0, 2.0 * np.pi)] * (2 * depth),
        options={"maxiter": max_evaluations, "rhobeg": 0.5, "tol": 1e-5, "catol": 1e-7},
    )
    gamma = tuple(float(value) for value in result.x[:depth])
    beta = tuple(float(value) for value in result.x[depth:])
    circuit = build_xy_qaoa_circuit(problem, gamma, beta, topology=topology)
    probabilities = exact_qaoa_distribution(problem, circuit)
    expected_objective = float(np.dot(probabilities, objective_values))
    optimal_masks = [
        mask for mask in range(1 << n)
        if feasible[mask]
        and abs(objective_values[mask] - optimum.objective) <= 1e-10
    ]
    supported = np.flatnonzero(probabilities > 1e-12)
    best_supported = max(objective_values[mask] for mask in supported if feasible[mask])
    return QAOAResult(
        method=f"xy_qaoa_{topology}_shared_angle",
        depth=depth,
        gamma=gamma,
        beta=beta,
        exact_expected_objective=expected_objective,
        exact_optimal_solution_probability=float(probabilities[optimal_masks].sum()),
        exact_feasible_probability=float(probabilities[feasible].sum()),
        exact_best_supported_objective=float(best_supported),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        evaluations=counts,
    )


def sample_aer(
    circuit: QuantumCircuit,
    *,
    shots: int,
    seed: int,
    noise_model: NoiseModel | None = None,
) -> dict[str, int]:
    """Sample a logical circuit with the installed local Qiskit Aer backend."""
    if shots < 1:
        raise ValueError("shots must be positive.")
    measured = circuit.copy()
    measured.measure_all()
    backend = AerSimulator(method="statevector", noise_model=noise_model)
    compiled = transpile(measured, backend, seed_transpiler=seed, optimization_level=1)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    return {str(key): int(value) for key, value in result.get_counts().items()}


def illustrative_noise_model() -> NoiseModel:
    """A transparent stress-test model, not calibrated to any real QPU."""
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["sx", "x"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
    model.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.02, 0.98]]))
    return model


def circuit_resource_summary(circuit: QuantumCircuit, *, seed: int = 42) -> dict[str, object]:
    """Return logical and decomposed resource counts on a documented basis."""
    basis = ["cx", "rz", "sx", "x"]
    compiled = transpile(circuit, basis_gates=basis, seed_transpiler=seed, optimization_level=0)
    return {
        "logical_qubits": circuit.num_qubits,
        "ancilla_qubits": 0,
        "logical_depth": circuit.depth(),
        "logical_operations": {str(k): int(v) for k, v in circuit.count_ops().items()},
        "transpiled_basis": basis,
        "transpiled_depth": compiled.depth(),
        "transpiled_operations": {str(k): int(v) for k, v in compiled.count_ops().items()},
    }
