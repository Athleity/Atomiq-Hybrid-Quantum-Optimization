"""Indicator Function QAOA (IF-QAOA) for Stage 2's integer inequalities.

The QPE registers encode the signed integer slacks ``limit - sum(a_i x_i)``
in two's-complement form. The objective phase is applied only when both sign
bits are zero, then both QPE computations are uncomputed. Exact cardinality is
enforced separately by a Dicke initial state and an excitation-preserving XY
mixer. This is an explicit Stage 2 adaptation of IF-QAOA, not a penalty QUBO.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, PhaseGate, QFTGate, StatePreparation
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from atomiq.stage2 import Stage2Problem, enumerate_optimum, evaluate_selection


@dataclass(frozen=True)
class IFQAOAResult:
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    expected_objective: float
    feasible_probability: float
    optimal_solution_probability: float
    optimizer_success: bool
    optimizer_message: str
    objective_evaluations: int
    logical_qubits: int


def signed_register_width(limit: int, weights: Sequence[int]) -> int:
    """Smallest two's-complement width representing every possible slack."""
    if int(limit) != limit or limit < 0 or any(int(w) != w or w < 0 for w in weights):
        raise ValueError("Constraint limits and coefficients must be nonnegative integers.")
    largest_magnitude = max(limit, abs(limit - sum(int(w) for w in weights)))
    width = 1
    while largest_magnitude >= (1 << (width - 1)):
        width += 1
    return width


def _qpe_terms(problem: Stage2Problem, weights: Sequence[int], limit: int):
    if len(weights) != len(problem.candidates):
        raise ValueError("Each candidate needs exactly one integer constraint coefficient.")
    if any(int(w) != w or w < 0 for w in weights):
        raise ValueError("IF-QAOA phase estimation requires nonnegative integer coefficients.")
    if int(limit) != limit or limit < 0:
        raise ValueError("IF-QAOA phase estimation requires a nonnegative integer limit.")
    width = signed_register_width(limit, weights)
    return width, tuple(int(w) for w in weights), int(limit)


def _append_slack_qpe(
    circuit: QuantumCircuit,
    problem: Stage2Problem,
    selector: Sequence[int],
    estimate: Sequence[int],
    *,
    weights: Sequence[int],
    limit: int,
    inverse: bool = False,
) -> None:
    """Compute/uncompute exact QPE for g(x)=limit-sum(weights_i*x_i).

    Since g is integer-valued and bounded within the signed range selected by
    ``signed_register_width``, QPE is exact and its most-significant bit is the
    two's-complement sign bit. The diagonal unitary is synthesized with
    candidate-controlled phase rotations; it performs no classical filtering.
    """
    width, integer_weights, integer_limit = _qpe_terms(problem, weights, limit)
    if len(estimate) != width:
        raise ValueError(f"Expected {width} phase-estimation qubits; got {len(estimate)}.")
    turns = -1.0 if inverse else 1.0
    if inverse:
        circuit.append(QFTGate(width), list(estimate))
    else:
        for qubit in estimate:
            circuit.h(qubit)

    order = range(width - 1, -1, -1) if inverse else range(width)
    denominator = float(1 << width)
    for bit in order:
        phase_scale = 2.0 * np.pi * (1 << bit) / denominator
        circuit.p(turns * phase_scale * integer_limit, estimate[bit])
        for index, weight in enumerate(integer_weights):
            angle = -turns * phase_scale * weight
            if angle:
                circuit.cp(angle, estimate[bit], selector[index])
    if inverse:
        for qubit in estimate:
            circuit.h(qubit)
    else:
        circuit.append(QFTGate(width).inverse(), list(estimate))


def append_if_cost_phase(
    circuit: QuantumCircuit,
    problem: Stage2Problem,
    selector: Sequence[int],
    budget_estimate: Sequence[int],
    capacity_estimate: Sequence[int],
    gamma: float,
) -> None:
    """Append the paper's QPE indicator oracle and uncompute both registers."""
    if len(selector) != len(problem.candidates):
        raise ValueError("Selector register width must equal the candidate count.")
    budget_weights = [candidate.contact_cost for candidate in problem.candidates]
    capacity_weights = [candidate.capacity_units for candidate in problem.candidates]
    _append_slack_qpe(
        circuit, problem, selector, budget_estimate, weights=budget_weights,
        limit=problem.budget_limit,
    )
    _append_slack_qpe(
        circuit, problem, selector, capacity_estimate, weights=capacity_weights,
        limit=problem.capacity_limit,
    )

    budget_sign = budget_estimate[-1]
    capacity_sign = capacity_estimate[-1]
    # QPE sign 0 means a satisfied inequality. Convert the two zero-controls
    # to one-controls around controlled-P and restore the indicator registers.
    circuit.x(budget_sign)
    circuit.x(capacity_sign)
    # The paper assumes a nonpositive minimization objective so infeasible
    # assignments receive zero cost. A constant lower shift makes -profit
    # nonpositive on every assignment without changing feasible ranking.
    cost_offset = sum(max(0.0, -candidate.expected_net_value) for candidate in problem.candidates)
    if cost_offset:
        circuit.append(
            DiagonalGate([1, 1, 1, np.exp(1j * float(gamma) * cost_offset)]),
            [budget_sign, capacity_sign],
        )
    for index, candidate in enumerate(problem.candidates):
        net_value = candidate.expected_net_value
        if net_value:
            controlled_phase = PhaseGate(float(gamma) * net_value).control(2)
            circuit.append(controlled_phase, [budget_sign, capacity_sign, selector[index]])
    circuit.x(capacity_sign)
    circuit.x(budget_sign)

    _append_slack_qpe(
        circuit, problem, selector, capacity_estimate, weights=capacity_weights,
        limit=problem.capacity_limit, inverse=True,
    )
    _append_slack_qpe(
        circuit, problem, selector, budget_estimate, weights=budget_weights,
        limit=problem.budget_limit, inverse=True,
    )


def build_if_qaoa_circuit(
    problem: Stage2Problem,
    gamma: Sequence[float],
    beta: Sequence[float],
    *,
    topology: str = "cycle",
) -> QuantumCircuit:
    """Build IF-QAOA with exact-K initialization, QPE indicators and XY mixer."""
    if not gamma or len(gamma) != len(beta):
        raise ValueError("gamma and beta must have the same positive depth.")
    if topology not in {"cycle", "path", "complete"}:
        raise ValueError("topology must be cycle, path, or complete.")
    n = len(problem.candidates)
    budget_width = signed_register_width(
        problem.budget_limit, [candidate.contact_cost for candidate in problem.candidates]
    )
    capacity_width = signed_register_width(
        problem.capacity_limit, [candidate.capacity_units for candidate in problem.candidates]
    )
    circuit = QuantumCircuit(n + budget_width + capacity_width)
    selector = list(range(n))
    budget_estimate = list(range(n, n + budget_width))
    capacity_estimate = list(range(n + budget_width, n + budget_width + capacity_width))
    initial = np.zeros(1 << n, dtype=complex)
    masks = [mask for mask in range(1 << n) if mask.bit_count() == problem.selected_count]
    if len(masks) != comb(n, problem.selected_count):
        raise AssertionError("Dicke support cardinality mismatch.")
    initial[masks] = 1.0 / np.sqrt(len(masks))
    circuit.append(StatePreparation(initial), selector)

    edges = [(i, i + 1) for i in range(n - 1)]
    if topology == "cycle":
        edges.append((n - 1, 0))
    elif topology == "complete":
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for layer_gamma, layer_beta in zip(gamma, beta, strict=True):
        append_if_cost_phase(
            circuit, problem, selector, budget_estimate, capacity_estimate,
            float(layer_gamma),
        )
        for first, second in edges:
            circuit.rxx(float(layer_beta), first, second)
            circuit.ryy(float(layer_beta), first, second)
    circuit.name = f"stage2_if_qaoa_{topology}_p{len(gamma)}"
    return circuit


def selector_distribution(problem: Stage2Problem, circuit: QuantumCircuit) -> np.ndarray:
    """Return exact probabilities marginalized over clean IF ancillas."""
    n = len(problem.candidates)
    amplitudes = Statevector.from_instruction(circuit).data
    probabilities = np.abs(amplitudes.reshape(-1, 1 << n)) ** 2
    marginal = probabilities.sum(axis=0)
    marginal /= marginal.sum()
    return marginal


def optimize_if_qaoa(
    problem: Stage2Problem,
    *,
    depth: int = 1,
    seed: int = 42,
    max_evaluations: int = 40,
    topology: str = "cycle",
) -> IFQAOAResult:
    """Optimize exact statevector expectation for the IF-QAOA construction."""
    if depth < 1 or max_evaluations < 1:
        raise ValueError("depth and max_evaluations must be positive.")
    n = len(problem.candidates)
    objective = np.zeros(1 << n, dtype=float)
    minimization_cost = np.zeros(1 << n, dtype=float)
    feasible = np.zeros(1 << n, dtype=bool)
    cost_offset = sum(max(0.0, -candidate.expected_net_value) for candidate in problem.candidates)
    for mask in range(1 << n):
        selected = [
            candidate.customer_id for index, candidate in enumerate(problem.candidates)
            if (mask >> index) & 1
        ]
        result = evaluate_selection(problem, selected)
        objective[mask], feasible[mask] = result.objective, result.feasible
        minimization_cost[mask] = -result.objective - cost_offset if result.feasible else 0.0
    optimum, _ = enumerate_optimum(problem)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(0.0, 2.0 * np.pi, 2 * depth)
    evaluations = 0

    def loss(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        circuit = build_if_qaoa_circuit(
            problem, parameters[:depth], parameters[depth:], topology=topology
        )
        probabilities = selector_distribution(problem, circuit)
        return float(np.dot(probabilities, minimization_cost))

    result = minimize(
        loss, initial, method="COBYLA",
        bounds=[(0.0, 2.0 * np.pi)] * (2 * depth),
        options={"maxiter": max_evaluations, "rhobeg": 0.5, "tol": 1e-5, "catol": 1e-7},
    )
    gamma = tuple(float(value) for value in result.x[:depth])
    beta = tuple(float(value) for value in result.x[depth:])
    final = selector_distribution(
        problem, build_if_qaoa_circuit(problem, gamma, beta, topology=topology)
    )
    optimal = np.array([
        abs(objective[mask] - optimum.objective) <= 1e-10 and feasible[mask]
        for mask in range(1 << n)
    ])
    ancilla_count = signed_register_width(
        problem.budget_limit, [c.contact_cost for c in problem.candidates]
    ) + signed_register_width(
        problem.capacity_limit, [c.capacity_units for c in problem.candidates]
    )
    return IFQAOAResult(
        gamma, beta, float(np.dot(final[feasible], objective[feasible])), float(final[feasible].sum()),
        float(final[optimal].sum()), bool(result.success), str(result.message),
        evaluations, n + ancilla_count,
    )
