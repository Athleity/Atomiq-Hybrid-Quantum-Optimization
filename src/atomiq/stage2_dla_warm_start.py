"""Dynamical-Lie-algebra-informed warm-start QAOA in a fixed-K sector.

Implements the paper's restricted cycle-XY + local-RZ pretraining, zero-
initialized all-pairs RZZ activation, parameter transfer, and full-ansatz
fine-tuning. Both shared-angle and per-generator multi-angle parameterizations
are supported. The simulator uses exact cardinality-sector statevectors and
parameter-shift Adam updates; this is an exact-K adaptation, not a reproduction
of the paper's portfolio/graph instances or its PennyLane execution setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Sequence

import numpy as np

from atomiq.stage2 import Stage2Problem, enumerate_optimum, evaluate_selection


Parameterization = Literal["shared", "multi"]


@dataclass(frozen=True)
class DLAReport:
    n_qubits: int
    selected_count: int
    sector_dimension: int
    restricted_dimension: int
    full_dimension: int
    tolerance: float
    complete: bool
    restricted_theoretical_upper_bound: int
    full_sector_dimension_upper_bound: int


@dataclass(frozen=True)
class DLAWarmStartResult:
    seed: int
    parameterization: str
    depth: int
    restricted_initial_cost: float
    restricted_final_cost: float
    full_initial_cost: float
    full_final_cost: float
    random_full_initial_cost: float
    random_full_final_cost: float
    full_expected_feasible_objective: float
    random_full_expected_feasible_objective: float
    full_feasible_probability: float
    random_full_feasible_probability: float
    random_full_optimal_solution_probability: float
    exact_feasible_probability: float
    exact_optimal_solution_probability: float
    restricted_steps: int
    full_steps: int
    transferred_parameter_count: int
    activated_zz_parameter_count: int
    sector_leakage: float
    trace: tuple[dict[str, float], ...]


def fixed_k_masks(n: int, k: int) -> tuple[int, ...]:
    if n < 1 or not 0 <= k <= n:
        raise ValueError("Require n>=1 and 0<=k<=n.")
    return tuple(sum(1 << i for i in subset) for subset in combinations(range(n), k))


def _family(generator_name: str) -> str:
    return generator_name.split("_", maxsplit=1)[0]


def sector_generators(
    n: int,
    k: int,
    *,
    include_zz: bool,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Return RZ, cycle-XY and optionally all-pairs RZZ Hermitian generators."""
    masks = fixed_k_masks(n, k)
    index = {mask: i for i, mask in enumerate(masks)}
    generators: list[tuple[str, np.ndarray]] = []
    for qubit in range(n):
        z = np.array([0.5 * (1 - 2 * ((mask >> qubit) & 1)) for mask in masks])
        generators.append((f"z_{qubit}", np.diag(z)))
    edges = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    for first, second in edges:
        matrix = np.zeros((len(masks), len(masks)), dtype=float)
        for row, mask in enumerate(masks):
            first_bit, second_bit = (mask >> first) & 1, (mask >> second) & 1
            if first_bit != second_bit:
                partner = mask ^ (1 << first) ^ (1 << second)
                matrix[row, index[partner]] = 1.0
        generators.append((f"xy_{first}_{second}", matrix))
    if include_zz:
        for first in range(n):
            for second in range(first + 1, n):
                values = np.array([
                    0.5
                    * (1 - 2 * ((mask >> first) & 1))
                    * (1 - 2 * ((mask >> second) & 1))
                    for mask in masks
                ])
                generators.append((f"zz_{first}_{second}", np.diag(values)))
    return tuple(generators)


def dynamical_lie_closure_dimension(
    generators: Sequence[np.ndarray],
    *,
    tolerance: float = 1e-10,
    max_dimension: int | None = None,
) -> tuple[int, bool]:
    """Numerically close a matrix Lie algebra using twice-reorthogonalized MGS.

    The DLA is a real vector space of skew-Hermitian matrices. Two-pass modified
    Gram-Schmidt with a relative Frobenius threshold avoids counting near-zero
    commutators as independent directions. ``complete=False`` signals that a
    caller-supplied dimension cap stopped closure before convergence.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if not generators:
        return 0, True
    shape = generators[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or any(g.shape != shape for g in generators):
        raise ValueError("All Lie generators must be square matrices with equal dimensions.")
    if max_dimension is not None and max_dimension < 1:
        raise ValueError("max_dimension must be positive when supplied.")

    basis: list[np.ndarray] = []

    def add_direction(skew_matrix: np.ndarray) -> bool:
        candidate = np.asarray(skew_matrix, dtype=complex).copy()
        if not np.allclose(candidate.conj().T, -candidate, atol=1e-9, rtol=0):
            raise ValueError("Generators must be Hermitian up to numerical precision.")
        original_norm = float(np.linalg.norm(candidate))
        if original_norm == 0.0:
            return False
        for _ in range(2):
            for vector in basis:
                coefficient = float(np.vdot(vector, candidate).real)
                candidate -= coefficient * vector
        residual_norm = float(np.linalg.norm(candidate))
        if residual_norm <= tolerance * max(1.0, original_norm):
            return False
        basis.append(candidate / residual_norm)
        return True

    for generator in generators:
        if add_direction(1j * generator):
            if max_dimension is not None and len(basis) >= max_dimension:
                return len(basis), False
    cursor = 0
    while cursor < len(basis):
        current = basis[cursor]
        for other_index in range(cursor):
            commutator = current @ basis[other_index] - basis[other_index] @ current
            if add_direction(commutator):
                if max_dimension is not None and len(basis) >= max_dimension:
                    return len(basis), False
        cursor += 1
    return len(basis), True


def analyze_dla(n: int, k: int, *, tolerance: float = 1e-10, max_dimension: int = 64) -> DLAReport:
    masks = fixed_k_masks(n, k)
    restricted_gens = [g for _, g in sector_generators(n, k, include_zz=False)]
    full_gens = [g for _, g in sector_generators(n, k, include_zz=True)]
    restricted_dim, restricted_complete = dynamical_lie_closure_dimension(
        restricted_gens, tolerance=tolerance, max_dimension=max_dimension
    )
    full_dim, full_complete = dynamical_lie_closure_dimension(
        full_gens, tolerance=tolerance, max_dimension=max_dimension
    )
    # The cycle XY + Z restricted action is the exterior-power representation
    # of u(n); the full unitary algebra on the K sector is the upper bound.
    return DLAReport(
        n_qubits=n,
        selected_count=k,
        sector_dimension=len(masks),
        restricted_dimension=restricted_dim,
        full_dimension=full_dim,
        tolerance=tolerance,
        complete=restricted_complete and full_complete,
        restricted_theoretical_upper_bound=n * n,
        full_sector_dimension_upper_bound=len(masks) ** 2,
    )


def _apply_gate(state: np.ndarray, generator: np.ndarray, angle: float, kind: str) -> np.ndarray:
    if kind in {"z", "zz"}:
        return np.exp(-1j * angle * np.diag(generator)) * state
    # Every single-edge XY generator is a collection of disjoint exchanges.
    rows, cols = np.where(np.triu(generator, k=1) != 0)
    result = state.copy()
    cosine, sine = np.cos(angle), np.sin(angle)
    for left, right in zip(rows, cols, strict=True):
        a, b = state[left], state[right]
        result[left] = cosine * a - 1j * sine * b
        result[right] = cosine * b - 1j * sine * a
    return result


def _parameter_layout(
    generators: Sequence[tuple[str, np.ndarray]],
    depth: int,
    parameterization: Parameterization,
) -> tuple[tuple[str, int], ...]:
    groups = tuple((_family(name), index) for index, (name, _) in enumerate(generators))
    if parameterization == "multi":
        return tuple((kind, index) for _layer in range(depth) for kind, index in groups)
    group_order = tuple(dict.fromkeys(kind for kind, _ in groups))
    return tuple((kind, -1) for _layer in range(depth) for kind in group_order)


def _run_ansatz(
    initial: np.ndarray,
    generators: Sequence[tuple[str, np.ndarray]],
    params: np.ndarray,
    depth: int,
    parameterization: Parameterization,
) -> np.ndarray:
    state = initial.copy()
    groups = tuple(dict.fromkeys(_family(name) for name, _ in generators))
    cursor = 0
    for _ in range(depth):
        for kind in groups:
            matching = [i for i, (name, _) in enumerate(generators) if _family(name) == kind]
            if parameterization == "shared":
                angle = float(params[cursor])
                cursor += 1
                for index in matching:
                    state = _apply_gate(state, generators[index][1], angle, kind)
            else:
                for index in matching:
                    angle = float(params[cursor])
                    cursor += 1
                    state = _apply_gate(state, generators[index][1], angle, kind)
    if cursor != len(params):
        raise ValueError(f"Ansatz consumed {cursor} parameters, received {len(params)}.")
    return state


def _objective_cost(problem: Stage2Problem, masks: Sequence[int]) -> tuple[np.ndarray, np.ndarray, float]:
    evaluations = [
        evaluate_selection(
            problem,
            [candidate.customer_id for i, candidate in enumerate(problem.candidates) if (mask >> i) & 1],
        )
        for mask in masks
    ]
    profits = np.array([row.objective for row in evaluations], dtype=float)
    feasible = np.array([row.feasible for row in evaluations], dtype=bool)
    offset = sum(max(0.0, -candidate.expected_net_value) for candidate in problem.candidates)
    costs = np.where(feasible, -profits - offset, 0.0)
    return costs, feasible, offset


def _adam_parameter_shift(
    initial: np.ndarray,
    objective,
    parameter_kinds: Sequence[str],
    *,
    steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, float, tuple[dict[str, float], ...]]:
    """Adam using exact spectral parameter-shift rules (XY has frequencies 1,2)."""
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive.")
    parameters = np.asarray(initial, dtype=float).copy()
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    trace: list[dict[str, float]] = []
    loss = objective(parameters)
    for step in range(1, steps + 1):
        gradient = np.empty_like(parameters)
        for index, kind in enumerate(parameter_kinds):
            gradient[index] = parameter_shift_derivative(objective, parameters, index, kind)
        first = beta1 * first + (1 - beta1) * gradient
        second = beta2 * second + (1 - beta2) * gradient**2
        first_hat = first / (1 - beta1**step)
        second_hat = second / (1 - beta2**step)
        parameters -= learning_rate * first_hat / (np.sqrt(second_hat) + adam_eps)
        parameters = np.mod(parameters, 2 * np.pi)
        loss = objective(parameters)
        trace.append({"step": float(step), "loss": float(loss), "gradient_l2": float(np.linalg.norm(gradient))})
    return parameters, float(loss), tuple(trace)


def parameter_shift_derivative(objective, parameters: np.ndarray, index: int, kind: str) -> float:
    """Exact derivative for one local generator's known spectral gap."""
    if not 0 <= index < len(parameters):
        raise IndexError("Parameter index is outside the parameter vector.")
    if kind == "xy":
        # Generator spectrum {-1,0,+1}; the two distinct gaps require two shifts.
        plus = parameters.copy(); plus[index] += np.pi / 2
        minus = parameters.copy(); minus[index] -= np.pi / 2
        plus_small = parameters.copy(); plus_small[index] += np.pi / 4
        minus_small = parameters.copy(); minus_small[index] -= np.pi / 4
        return float(
            (0.5 - np.sqrt(2) / 2) * (objective(plus) - objective(minus))
            + objective(plus_small) - objective(minus_small)
        )
    if kind in {"z", "zz"}:
        # RZ and RZZ generators have spectral gap one (eigenvalues ±1/2).
        plus = parameters.copy(); plus[index] += np.pi / 2
        minus = parameters.copy(); minus[index] -= np.pi / 2
        return float(0.5 * (objective(plus) - objective(minus)))
    raise ValueError(f"Unknown gate-generator family: {kind}")


def run_dla_warm_start(
    problem: Stage2Problem,
    *,
    seed: int,
    parameterization: Parameterization,
    depth: int = 1,
    restricted_steps: int = 100,
    full_steps: int = 100,
    learning_rate: float = 0.04,
) -> DLAWarmStartResult:
    """Train restricted XY+Z, transfer angles, activate all-pairs ZZ, refine."""
    if parameterization not in {"shared", "multi"}:
        raise ValueError("parameterization must be 'shared' or 'multi'.")
    if depth < 1:
        raise ValueError("depth must be positive.")
    n, k = len(problem.candidates), problem.selected_count
    masks = fixed_k_masks(n, k)
    if len(masks) != len(set(masks)):
        raise AssertionError("Fixed-K bitstring enumeration is not unique.")
    costs, feasible, _offset = _objective_cost(problem, masks)
    optimum, _ = enumerate_optimum(problem)
    initial = np.full(len(masks), 1 / np.sqrt(len(masks)), dtype=complex)
    restricted = sector_generators(n, k, include_zz=False)
    full = sector_generators(n, k, include_zz=True)
    restricted_layout = _parameter_layout(restricted, depth, parameterization)
    full_layout = _parameter_layout(full, depth, parameterization)
    rng = np.random.default_rng(seed)
    restricted_initial = rng.uniform(0.0, 2.0 * np.pi, len(restricted_layout))

    def restricted_objective(params: np.ndarray) -> float:
        state = _run_ansatz(initial, restricted, params, depth, parameterization)
        return float(np.dot(np.abs(state) ** 2, costs))

    restricted_initial_cost = restricted_objective(restricted_initial)
    restricted_params, restricted_final_cost, trace_restricted = _adam_parameter_shift(
        restricted_initial, restricted_objective,
        [kind for kind, _ in restricted_layout],
        steps=restricted_steps, learning_rate=learning_rate,
    )

    # Transfer every RZ and cycle-XY parameter by its generator name/order.
    # The all-pairs RZZ parameters are initialized to the identity (zero).
    full_initial = np.zeros(len(full_layout), dtype=float)
    if parameterization == "shared":
        for layer in range(depth):
            full_initial[3 * layer] = restricted_params[2 * layer]
            full_initial[3 * layer + 1] = restricted_params[2 * layer + 1]
        transferred_count = len(restricted_params)
        activated_count = depth
    else:
        r_cursor = f_cursor = 0
        for _layer in range(depth):
            # The paper's restricted multi-angle layer has n local Z and n
            # cycle-XY angles; the full layer appends n(n-1)/2 zero RZZ angles.
            restricted_layer_width = len(restricted)
            full_layer_width = len(full)
            full_initial[f_cursor:f_cursor + restricted_layer_width] = restricted_params[
                r_cursor:r_cursor + restricted_layer_width
            ]
            r_cursor += restricted_layer_width
            f_cursor += full_layer_width
        transferred_count = len(restricted_params)
        activated_count = depth * (n * (n - 1) // 2)

    def full_objective(params: np.ndarray) -> float:
        state = _run_ansatz(initial, full, params, depth, parameterization)
        return float(np.dot(np.abs(state) ** 2, costs))

    full_initial_cost = full_objective(full_initial)
    full_params, full_final_cost, trace_full = _adam_parameter_shift(
        full_initial, full_objective,
        [kind for kind, _ in full_layout],
        steps=full_steps, learning_rate=learning_rate,
    )
    final_state = _run_ansatz(initial, full, full_params, depth, parameterization)
    distribution = np.abs(final_state) ** 2
    sector_leakage = float(max(0.0, 1.0 - distribution.sum()))
    exact_feasible = float(distribution[feasible].sum())
    optimal = np.array([
        feasible[i] and abs(-costs[i] - _offset - optimum.objective) <= 1e-10
        for i in range(len(masks))
    ])
    random_initial = np.random.default_rng(seed + 100_003).uniform(
        0.0, 2.0 * np.pi, len(full_layout)
    )
    random_full_initial_cost = full_objective(random_initial)
    random_params, random_full_final_cost, _trace_random = _adam_parameter_shift(
        random_initial,
        full_objective,
        [kind for kind, _ in full_layout],
        steps=full_steps,
        learning_rate=learning_rate,
    )
    random_state = _run_ansatz(initial, full, random_params, depth, parameterization)
    random_distribution = np.abs(random_state) ** 2
    profits = np.where(feasible, -costs - _offset, 0.0)
    trace = tuple(
        {"phase": 0.0, **row} for row in trace_restricted
    ) + tuple({"phase": 1.0, **row} for row in trace_full)
    return DLAWarmStartResult(
        seed=seed,
        parameterization=parameterization,
        depth=depth,
        restricted_initial_cost=restricted_initial_cost,
        restricted_final_cost=restricted_final_cost,
        full_initial_cost=full_initial_cost,
        full_final_cost=full_final_cost,
        random_full_initial_cost=random_full_initial_cost,
        random_full_final_cost=random_full_final_cost,
        full_expected_feasible_objective=float(np.dot(distribution, profits)),
        random_full_expected_feasible_objective=float(np.dot(random_distribution, profits)),
        full_feasible_probability=float(distribution[feasible].sum()),
        random_full_feasible_probability=float(random_distribution[feasible].sum()),
        random_full_optimal_solution_probability=float(random_distribution[optimal].sum()),
        exact_feasible_probability=exact_feasible,
        exact_optimal_solution_probability=float(distribution[optimal].sum()),
        restricted_steps=restricted_steps,
        full_steps=full_steps,
        transferred_parameter_count=transferred_count,
        activated_zz_parameter_count=activated_count,
        sector_leakage=sector_leakage,
        trace=trace,
    )
