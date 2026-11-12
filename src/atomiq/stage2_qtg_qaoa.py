"""Quantum Tree Generator AAM-QAOA adaptation for Stage 2.

This uses the original QTG biased-Hadamard branching rule, followed by its
amplitude-amplification mixer U_SP [I-(1-e^-iβ)|0><0|] U_SP†. Exact cardinality
and both resource limits are integrated into the branch predicate. A suffix
dynamic-programming feasibility oracle prevents dead branches; a reversible
implementation would require compiling this lookup/control plus resource
accumulators. This module is the paper-style high-level statevector simulator,
not a hardware gate decomposition or an exact reproduction of one-capacity
knapsack experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, log2
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from atomiq.stage2 import Stage2Problem, enumerate_optimum, evaluate_selection


@dataclass(frozen=True)
class QTGResult:
    seed: int
    bias_b: float
    depth: int
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    optimizer_success: bool
    optimizer_message: str
    objective_evaluations: int
    expected_objective: float
    feasible_probability: float
    optimal_solution_probability: float
    approximation_gap: float
    shots: int
    sampled_feasible_rate: float
    sampled_optimal_hits: int
    best_sampled_objective: float
    feasible_state_count: int
    positive_amplitude_state_count: int
    logical_data_qubits_lower_bound: int


def _completion_oracle(problem: Stage2Problem):
    """Memoized exact suffix feasibility oracle for (index,count,cost,capacity)."""
    candidates = problem.candidates
    n = len(candidates)

    @lru_cache(maxsize=None)
    def can_complete(index: int, selected: int, cost: int, capacity: int) -> bool:
        if cost > problem.budget_limit or capacity > problem.capacity_limit:
            return False
        if selected > problem.selected_count:
            return False
        remaining = n - index
        needed = problem.selected_count - selected
        if needed < 0 or needed > remaining:
            return False
        if index == n:
            return needed == 0
        if needed == 0:
            return True
        candidate = candidates[index]
        take = (
            selected < problem.selected_count
            and cost + candidate.contact_cost <= problem.budget_limit
            and capacity + candidate.capacity_units <= problem.capacity_limit
            and can_complete(
                index + 1, selected + 1,
                cost + candidate.contact_cost,
                capacity + candidate.capacity_units,
            )
        )
        skip = can_complete(index + 1, selected, cost, capacity)
        return bool(take or skip)

    return can_complete


def qtg_initial_state(problem: Stage2Problem, *, bias_b: float = 0.0) -> tuple[tuple[int, ...], np.ndarray]:
    """Generate feasible-state amplitudes via QTG's sequential biased branches.

    For a discretionary branch the original H_b^0 rule assigns inclusion
    probability 1/(2+b), exclusion probability (1+b)/(2+b). A branch with only
    one feasible continuation is deterministically forced, as required for an
    exact-K constraint. Each complete feasible bitstring has positive support.
    """
    if not np.isfinite(bias_b) or bias_b < 0:
        raise ValueError("QTG bias_b must be finite and nonnegative.")
    can_complete = _completion_oracle(problem)
    if not can_complete(0, 0, 0, 0):
        raise ValueError("No feasible exact-cardinality state exists for QTG.")
    n = len(problem.candidates)
    states: list[int] = []
    amplitudes: list[float] = []
    p_take = 1.0 / (2.0 + bias_b)
    p_skip = (1.0 + bias_b) / (2.0 + bias_b)

    def visit(index: int, selected: int, cost: int, capacity: int, mask: int, amplitude: float) -> None:
        if index == n:
            if selected == problem.selected_count:
                states.append(mask)
                amplitudes.append(amplitude)
            return
        candidate = problem.candidates[index]
        can_take = (
            selected < problem.selected_count
            and cost + candidate.contact_cost <= problem.budget_limit
            and capacity + candidate.capacity_units <= problem.capacity_limit
            and can_complete(
                index + 1, selected + 1,
                cost + candidate.contact_cost,
                capacity + candidate.capacity_units,
            )
        )
        can_skip = can_complete(index + 1, selected, cost, capacity)
        if can_take and can_skip:
            visit(index + 1, selected + 1, cost + candidate.contact_cost,
                  capacity + candidate.capacity_units, mask | (1 << index),
                  amplitude * np.sqrt(p_take))
            visit(index + 1, selected, cost, capacity, mask,
                  amplitude * np.sqrt(p_skip))
        elif can_take:
            visit(index + 1, selected + 1, cost + candidate.contact_cost,
                  capacity + candidate.capacity_units, mask | (1 << index), amplitude)
        elif can_skip:
            visit(index + 1, selected, cost, capacity, mask, amplitude)
        else:
            raise AssertionError("Suffix feasibility oracle exposed a dead-end branch.")

    visit(0, 0, 0, 0, 0, 1.0)
    order = np.argsort(states)
    sorted_states = tuple(states[index] for index in order)
    state = np.asarray([amplitudes[index] for index in order], dtype=complex)
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, atol=1e-12):
        raise ArithmeticError(f"QTG tree state is not normalized (norm={norm}).")
    if any(value <= 0 for value in state.real) or np.any(np.abs(state.imag) > 1e-14):
        raise ArithmeticError("QTG feasible-state amplitudes must be strictly positive and real.")
    return sorted_states, state


def qtg_amplitude_amplification_mixer(state: np.ndarray, beta: float, vector: np.ndarray) -> np.ndarray:
    """Apply U_SP exp(-i beta |0><0|) U_SP† = exp(-i beta |phi><phi|)."""
    overlap = np.vdot(state, vector)
    return vector - (1.0 - np.exp(-1j * beta)) * overlap * state


def run_qtg_qaoa(
    problem: Stage2Problem,
    *,
    seed: int = 11,
    bias_b: float = 0.0,
    depth: int = 1,
    max_evaluations: int = 80,
    shots: int = 4096,
) -> QTGResult:
    """Optimize paper-style QTG AAM-QAOA in its feasible-state simulator."""
    if depth < 1 or max_evaluations < 1 or shots < 1:
        raise ValueError("depth, max_evaluations and shots must be positive.")
    masks, initial = qtg_initial_state(problem, bias_b=bias_b)
    evaluations = [
        evaluate_selection(
            problem,
            [candidate.customer_id for i, candidate in enumerate(problem.candidates) if (mask >> i) & 1],
        )
        for mask in masks
    ]
    objectives = np.array([row.objective for row in evaluations], dtype=float)
    optimum, exact_feasible = enumerate_optimum(problem)
    optimal_mask = np.array([
        abs(value - optimum.objective) <= 1e-10 for value in objectives
    ])
    rng = np.random.default_rng(seed)
    initial_parameters = rng.uniform(0, 2 * np.pi, 2 * depth)
    objective_evaluations = 0

    def evolved(parameters: Sequence[float]) -> np.ndarray:
        gamma, beta = parameters[:depth], parameters[depth:]
        vector = initial.copy()
        for angle_g, angle_b in zip(gamma, beta, strict=True):
            vector *= np.exp(-1j * angle_g * objectives)
            vector = qtg_amplitude_amplification_mixer(initial, float(angle_b), vector)
        return vector

    def loss(parameters: np.ndarray) -> float:
        nonlocal objective_evaluations
        objective_evaluations += 1
        vector = evolved(parameters)
        return float(-np.dot(np.abs(vector) ** 2, objectives))

    optimized = minimize(
        loss, initial_parameters, method="COBYLA",
        bounds=[(0.0, 2 * np.pi)] * (2 * depth),
        options={"maxiter": max_evaluations, "rhobeg": 0.5, "tol": 1e-6},
    )
    gamma = tuple(float(v) for v in optimized.x[:depth])
    beta = tuple(float(v) for v in optimized.x[depth:])
    final = evolved((*gamma, *beta))
    distribution = np.abs(final) ** 2
    distribution /= distribution.sum()
    sample_indices = rng.choice(len(masks), size=shots, p=distribution)
    sampled_objectives = objectives[sample_indices]
    optimal_hits = int(optimal_mask[sample_indices].sum())
    candidate_count = len(problem.candidates)
    budget_register = max(1, ceil(log2(sum(c.contact_cost for c in problem.candidates) + 1)))
    capacity_register = max(1, ceil(log2(sum(c.capacity_units for c in problem.candidates) + 1)))
    count_register = max(1, ceil(log2(problem.selected_count + 1)))
    logical_qubits = candidate_count + budget_register + capacity_register + count_register
    expected = float(np.dot(distribution, objectives))
    return QTGResult(
        seed=seed,
        bias_b=bias_b,
        depth=depth,
        gamma=gamma,
        beta=beta,
        optimizer_success=bool(optimized.success),
        optimizer_message=str(optimized.message),
        objective_evaluations=objective_evaluations,
        expected_objective=expected,
        feasible_probability=float(distribution.sum()),
        optimal_solution_probability=float(distribution[optimal_mask].sum()),
        approximation_gap=float((optimum.objective - expected) / max(abs(optimum.objective), 1e-12)),
        shots=shots,
        sampled_feasible_rate=1.0,
        sampled_optimal_hits=optimal_hits,
        best_sampled_objective=float(sampled_objectives.max()),
        feasible_state_count=len(masks),
        positive_amplitude_state_count=int(np.count_nonzero(np.abs(initial) > 0)),
        logical_data_qubits_lower_bound=logical_qubits,
    )
