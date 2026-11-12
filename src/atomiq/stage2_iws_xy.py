"""Exact-K adaptation of Iterative Warm-Start XY-QAOA (IWS-XY).

The source paper's construction is for one-hot blocks and has a shallow
two-qubit warm-start mixer. Here one constraint is ``sum(x)=K``. We construct
the corresponding weighted mixer on the connected Johnson graph J(n,K): each
edge is a feasible one-item swap and carries the paper's warm-started 2x2
submixer. This proves the biased fixed-K state is the unique ground state,
but is an exact compressed-sector simulator, not the paper's shallow circuit.
It is consequently useful for algorithmic study and unsuitable as a claimed
hardware-efficient reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from atomiq.stage2 import Stage2Problem, enumerate_optimum, evaluate_selection


@dataclass(frozen=True)
class IWSIteration:
    iteration: int
    shots: int
    expected_feasible_objective: float
    feasible_shot_rate: float
    optimal_shot_rate: float
    best_feasible_objective: float | None
    marginal_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class IWSResult:
    seed: int
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    optimizer_success: bool
    optimizer_message: str
    optimization_evaluations: int
    iterations: tuple[IWSIteration, ...]
    total_shots: int
    exact_optimum: float
    exact_optimal_ids: tuple[str, ...]
    mixer_ground_energy: float
    mixer_ground_overlap: float


def _sector(problem: Stage2Problem) -> tuple[tuple[int, ...], dict[int, int]]:
    n = len(problem.candidates)
    masks = tuple(mask for mask in range(1 << n) if mask.bit_count() == problem.selected_count)
    return masks, {mask: index for index, mask in enumerate(masks)}


def _joint_distribution(marginals: Sequence[float], masks: Sequence[int]) -> np.ndarray:
    q = np.asarray(marginals, dtype=float)
    if q.ndim != 1 or np.any(~np.isfinite(q)) or np.any((q <= 0) | (q >= 1)):
        raise ValueError("Warm-start marginals must be finite and strictly between zero and one.")
    log_odds = np.log(q) - np.log1p(-q)
    log_weights = np.array([
        sum(log_odds[i] for i in range(len(q)) if (mask >> i) & 1)
        for mask in masks
    ])
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / weights.sum()


def weighted_johnson_mixer(
    marginals: Sequence[float],
    masks: Sequence[int],
    index: dict[int, int],
) -> np.ndarray:
    """Build the paper-aligned weighted XY mixer on exact-K subset states.

    For every Johnson-graph edge (S,T), the 2x2 block is the source paper's
    warm-started XY block with q=P(S)/(P(S)+P(T)); all edge terms are divided
    by the regular graph degree K(n-K). The positive edge weights imply a
    connected irreducible stoquastic matrix, so Perron-Frobenius gives the
    unique ground vector sqrt(P(S)) with energy -1.
    """
    n = len(marginals)
    if len(masks) == 0 or not masks:
        raise ValueError("The exact-K sector must contain at least one state.")
    k = masks[0].bit_count()
    degree = k * (n - k)
    if degree == 0:
        raise ValueError("The Johnson graph requires 0 < K < n.")
    distribution = _joint_distribution(marginals, masks)
    hamiltonian = np.zeros((len(masks), len(masks)), dtype=float)
    for left, mask in enumerate(masks):
        unselected = [i for i in range(n) if not ((mask >> i) & 1)]
        selected = [i for i in range(n) if (mask >> i) & 1]
        for removed in selected:
            for added in unselected:
                neighbor_mask = mask ^ (1 << removed) ^ (1 << added)
                right = index[neighbor_mask]
                if left >= right:
                    continue
                p_left, p_right = distribution[left], distribution[right]
                q = p_left / (p_left + p_right)
                hamiltonian[left, left] += 1.0 - 2.0 * q
                hamiltonian[right, right] += 2.0 * q - 1.0
                coupling = -2.0 * np.sqrt(q * (1.0 - q))
                hamiltonian[left, right] += coupling
                hamiltonian[right, left] += coupling
    hamiltonian /= degree
    return hamiltonian


def _cost_vector(problem: Stage2Problem, masks: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evaluations = [
        evaluate_selection(
            problem,
            [candidate.customer_id for i, candidate in enumerate(problem.candidates) if (mask >> i) & 1],
        )
        for mask in masks
    ]
    values = np.array([row.objective for row in evaluations], dtype=float)
    feasible = np.array([row.feasible for row in evaluations], dtype=bool)
    offset = sum(max(0.0, -candidate.expected_net_value) for candidate in problem.candidates)
    # IF-QAOA convention: minimize a nonpositive cost on feasible assignments,
    # while infeasible assignments receive zero cost, without penalty terms.
    cost = np.where(feasible, -values - offset, 0.0)
    return values, feasible, cost


def _project_capped_simplex(values: np.ndarray, total: float, lower: float, upper: float) -> np.ndarray:
    """Euclidean projection onto box constraints and a fixed-sum simplex."""
    if lower * len(values) > total or upper * len(values) < total:
        raise ValueError("Probability clamp bounds are incompatible with the fixed-K sum.")
    lo, hi = float(np.min(values - upper)), float(np.max(values - lower))
    for _ in range(100):
        shift = (lo + hi) / 2.0
        projected = np.clip(values - shift, lower, upper)
        if projected.sum() > total:
            lo = shift
        else:
            hi = shift
    projected = np.clip(values - (lo + hi) / 2.0, lower, upper)
    # Correct only floating-point residual while respecting the box.
    residual = total - projected.sum()
    free = np.flatnonzero((projected > lower + 1e-12) & (projected < upper - 1e-12))
    if len(free):
        projected[free] += residual / len(free)
    return projected


def run_iws_xy(
    problem: Stage2Problem,
    *,
    seed: int = 11,
    depth: int = 1,
    shots_per_iteration: int = 128,
    total_shots: int = 512,
    inverse_temperature: float = 4.0,
    epsilon: float = 0.1,
    max_evaluations: int = 40,
) -> IWSResult:
    """Run iterative IWS with one initial parameter optimization and reused angles."""
    if not 0 < epsilon < 1 or inverse_temperature <= 0:
        raise ValueError("epsilon must be in (0,1) and inverse_temperature positive.")
    if min(depth, shots_per_iteration, total_shots, max_evaluations) < 1:
        raise ValueError("depth, shot counts, and optimization budget must be positive.")
    masks, index = _sector(problem)
    n = len(problem.candidates)
    k = problem.selected_count
    values, feasible, costs = _cost_vector(problem, masks)
    optimum, _ = enumerate_optimum(problem)
    q = np.full(n, k / n, dtype=float)
    if min(q) <= epsilon / n or max(q) >= 1 - epsilon / n:
        raise ValueError("Initial marginals conflict with the selected epsilon clamp.")
    initial_distribution = _joint_distribution(q, masks)
    initial_state = np.sqrt(initial_distribution).astype(complex)
    mixer = weighted_johnson_mixer(q, masks, index)
    eigenvalues, eigenvectors = eigh(mixer)
    initial_ground = eigenvectors[:, 0]
    ground_overlap = float(abs(np.vdot(initial_ground, initial_state)) ** 2)

    rng = np.random.default_rng(seed)
    evaluations = 0

    def apply_layers(parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
        gamma = parameters[:depth]
        beta = parameters[depth:]
        current = state.copy()
        for angle_g, angle_b in zip(gamma, beta, strict=True):
            current *= np.exp(-1j * angle_g * costs)
            current = eigenvectors @ (np.exp(-1j * angle_b * eigenvalues) * (eigenvectors.conj().T @ current))
        return current

    initial_angles = rng.uniform(0.0, 2.0 * np.pi, 2 * depth)

    def loss(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        final_state = apply_layers(parameters, initial_state)
        return float(np.dot(np.abs(final_state) ** 2, costs))

    optimized = minimize(
        loss, initial_angles, method="COBYLA",
        bounds=[(0.0, 2.0 * np.pi)] * (2 * depth),
        options={"maxiter": max_evaluations, "rhobeg": 0.5, "tol": 1e-5, "catol": 1e-7},
    )
    gamma = tuple(float(v) for v in optimized.x[:depth])
    beta = tuple(float(v) for v in optimized.x[depth:])
    updates: list[IWSIteration] = []
    remaining = total_shots
    iteration = 0
    while remaining:
        batch = min(shots_per_iteration, remaining)
        distribution = _joint_distribution(q, masks)
        psi0 = np.sqrt(distribution).astype(complex)
        mixer = weighted_johnson_mixer(q, masks, index)
        eigenvalues, eigenvectors = eigh(mixer)
        state = psi0
        for angle_g, angle_b in zip(gamma, beta, strict=True):
            state = state * np.exp(-1j * angle_g * costs)
            state = eigenvectors @ (np.exp(-1j * angle_b * eigenvalues) * (eigenvectors.conj().T @ state))
        output_distribution = np.abs(state) ** 2
        output_distribution /= output_distribution.sum()
        sampled = rng.choice(len(masks), size=batch, p=output_distribution)
        sample_costs = costs[sampled]
        spread = float(sample_costs.max() - sample_costs.min())
        if spread <= np.finfo(float).eps:
            weights = np.ones(batch, dtype=float)
        else:
            log_weights = -inverse_temperature * sample_costs / spread
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
        weighted_marginals = np.array([
            np.dot(weights, [((masks[sample] >> i) & 1) for sample in sampled]) / weights.sum()
            for i in range(n)
        ])
        lower, upper = epsilon / n, 1.0 - epsilon / n
        q = _project_capped_simplex(weighted_marginals, float(k), lower, upper)
        sampled_values = values[sampled]
        sampled_feasible = feasible[sampled]
        best = sampled_values[sampled_feasible].max() if sampled_feasible.any() else None
        updates.append(IWSIteration(
            iteration=iteration,
            shots=batch,
            expected_feasible_objective=float(np.dot(output_distribution[feasible], values[feasible])),
            feasible_shot_rate=float(sampled_feasible.mean()),
            optimal_shot_rate=float(np.mean(
                sampled_feasible & (np.abs(sampled_values - optimum.objective) <= 1e-10)
            )),
            best_feasible_objective=None if best is None else float(best),
            marginal_probabilities=tuple(float(v) for v in q),
        ))
        remaining -= batch
        iteration += 1
    return IWSResult(
        seed=seed,
        gamma=gamma,
        beta=beta,
        optimizer_success=bool(optimized.success),
        optimizer_message=str(optimized.message),
        optimization_evaluations=evaluations,
        iterations=tuple(updates),
        total_shots=total_shots,
        exact_optimum=optimum.objective,
        exact_optimal_ids=optimum.customer_ids,
        mixer_ground_energy=float(eigenvalues[0]),
        mixer_ground_overlap=ground_overlap,
    )
