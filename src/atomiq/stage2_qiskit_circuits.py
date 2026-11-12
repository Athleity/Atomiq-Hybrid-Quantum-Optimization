"""Parameterized, executable Qiskit circuits for standard QAOA and exact-K XY-QAOA."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Literal, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import MCPhaseGate, StatePreparation
from qiskit.quantum_info import Statevector
from scipy.optimize import OptimizeResult, minimize

from atomiq.stage2 import Stage2Problem, enumerate_optimum, evaluate_selection


Ansatz = Literal["standard", "xy"]
MixerTopology = Literal["path", "cycle", "complete"]
ObjectiveKind = Literal["expectation", "cvar"]


@dataclass(frozen=True)
class CircuitAnsatz:
    circuit: QuantumCircuit
    gamma: tuple
    beta: tuple
    costs: np.ndarray
    objective_values: np.ndarray
    feasible: np.ndarray
    optimal: np.ndarray
    penalty: float
    kind: Ansatz
    topology: MixerTopology


@dataclass(frozen=True)
class VariationalFit:
    parameters: tuple[float, ...]
    optimizer: str
    objective_kind: str
    objective_value: float
    evaluations: int
    success: bool
    message: str
    seed: int
    runtime_seconds: float


def _selection(problem: Stage2Problem, mask: int) -> list[str]:
    return [c.customer_id for i, c in enumerate(problem.candidates) if (mask >> i) & 1]


def cost_table(problem: Stage2Problem, *, penalty_multiplier: float = 2.0):
    """Build minimized cost with explicit K and both resource-violation penalties."""
    if penalty_multiplier <= 0:
        raise ValueError("penalty_multiplier must be positive.")
    n = len(problem.candidates)
    exact = np.zeros(1 << n, dtype=float)
    feasible = np.zeros(1 << n, dtype=bool)
    for mask in range(1 << n):
        result = evaluate_selection(problem, _selection(problem, mask))
        exact[mask] = result.objective
        feasible[mask] = result.feasible
    net_range = sum(abs(c.expected_net_value) for c in problem.candidates)
    penalty = penalty_multiplier * (net_range + 1.0)
    costs = np.empty(1 << n, dtype=float)
    for mask in range(1 << n):
        result = evaluate_selection(problem, _selection(problem, mask))
        cardinality_violation = abs(len(result.customer_ids) - problem.selected_count)
        resource_violation = max(0, result.total_contact_cost - problem.budget_limit)
        resource_violation += max(0, result.total_capacity - problem.capacity_limit)
        costs[mask] = (
            -result.objective
            + penalty * cardinality_violation**2
            + penalty * resource_violation**2
        )
    optimum, _ = enumerate_optimum(problem)
    optimal = np.array([
        feasible[mask] and abs(exact[mask] - optimum.objective) <= 1e-10
        for mask in range(1 << n)
    ])
    return costs, exact, feasible, optimal, penalty


def _append_basis_phase(circuit: QuantumCircuit, theta, mask: int, n: int) -> None:
    """Apply a parameterized phase to exactly one little-endian basis state."""
    target = n - 1
    controls = list(range(target))
    target_is_one = bool((mask >> target) & 1)
    if not target_is_one:
        circuit.x(target)
    control_state = mask & ((1 << target) - 1)
    circuit.append(
        MCPhaseGate(theta, num_ctrl_qubits=target, ctrl_state=control_state),
        [*controls, target],
    )
    if not target_is_one:
        circuit.x(target)


def build_parameterized_qaoa(
    problem: Stage2Problem,
    *,
    depth: int = 1,
    kind: Ansatz = "xy",
    topology: MixerTopology = "cycle",
    penalty_multiplier: float = 2.0,
) -> CircuitAnsatz:
    """Build a real symbolic Qiskit circuit; return its angle Parameters and score tables."""
    if depth < 1:
        raise ValueError("depth must be positive.")
    if kind not in {"standard", "xy"}:
        raise ValueError("kind must be 'standard' or 'xy'.")
    if topology not in {"path", "cycle", "complete"}:
        raise ValueError("topology must be path, cycle, or complete.")
    n = len(problem.candidates)
    costs, objective, feasible, optimal, penalty = cost_table(
        problem, penalty_multiplier=penalty_multiplier
    )
    angles = ParameterVector("theta", 2 * depth)
    gammas, betas = tuple(angles[:depth]), tuple(angles[depth:])
    circuit = QuantumCircuit(n, name=f"stage2_{kind}_qaoa_{topology}_p{depth}")
    fixed_sector_feasible = (
        kind == "xy"
        and all(
            feasible[mask]
            for mask in range(1 << n)
            if mask.bit_count() == problem.selected_count
        )
    )
    if kind == "standard":
        circuit.h(range(n))
    else:
        initial = np.zeros(1 << n, dtype=complex)
        masks = [m for m in range(1 << n) if m.bit_count() == problem.selected_count]
        if len(masks) != comb(n, problem.selected_count):
            raise AssertionError("Exact-K Dicke support cardinality mismatch.")
        initial[masks] = 1.0 / np.sqrt(len(masks))
        circuit.append(StatePreparation(initial), range(n))

    for gamma, beta in zip(gammas, betas, strict=True):
        # Global phase is retained for direct basis-by-basis phase-oracle tests.
        if fixed_sector_feasible:
            # Within a fixed-K sector, -objective is additive and every
            # inequality indicator is one. Drop only the irrelevant global
            # phase; this avoids an exponential truth-table oracle.
            for q, candidate in enumerate(problem.candidates):
                if candidate.expected_net_value:
                    circuit.p(gamma * candidate.expected_net_value, q)
        else:
            for mask, value in enumerate(costs):
                if abs(value) > 1e-14:
                    _append_basis_phase(circuit, -gamma * float(value), mask, n)
        if kind == "standard":
            for q in range(n):
                circuit.rx(2.0 * beta, q)
        else:
            edges = [(q, q + 1) for q in range(n - 1)]
            if topology == "cycle":
                edges.append((n - 1, 0))
            elif topology == "complete":
                edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for first, second in edges:
                circuit.rxx(beta, first, second)
                circuit.ryy(beta, first, second)
    circuit.metadata = {
        "kind": kind, "topology": topology, "depth": depth,
        "fixed_sector_feasible_additive_cost": fixed_sector_feasible,
    }
    return CircuitAnsatz(
        circuit, gammas, betas, costs, objective, feasible, optimal, penalty, kind, topology
    )


def bind_angles(ansatz: CircuitAnsatz, parameters: Sequence[float]) -> QuantumCircuit:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (len(ansatz.gamma) + len(ansatz.beta),) or not np.isfinite(values).all():
        raise ValueError("Angle vector has invalid shape or non-finite values.")
    bindings = dict(zip((*ansatz.gamma, *ansatz.beta), values, strict=True))
    return ansatz.circuit.assign_parameters(bindings, inplace=False)


def cvar_lower_tail(costs: np.ndarray, probabilities: np.ndarray, alpha: float = 0.25) -> float:
    """Weighted lower-tail CVaR for a minimization cost, splitting boundary mass exactly."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    values = np.asarray(costs, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    if values.shape != probs.shape or np.any(probs < -1e-12) or probs.sum() <= 0:
        raise ValueError("Cost/probability arrays are invalid.")
    probs = np.maximum(probs, 0.0) / np.maximum(probs, 0.0).sum()
    order = np.argsort(values)
    mass_left = alpha
    weighted = 0.0
    for index in order:
        take = min(mass_left, float(probs[index]))
        weighted += take * float(values[index])
        mass_left -= take
        if mass_left <= 1e-14:
            break
    if mass_left > 1e-10:
        raise ArithmeticError("CVaR tail mass could not be filled from the distribution.")
    return weighted / alpha


def score_distribution(
    ansatz: CircuitAnsatz,
    probabilities: np.ndarray,
    *,
    objective_kind: ObjectiveKind = "expectation",
    alpha: float = 0.25,
) -> float:
    if objective_kind == "expectation":
        return float(np.dot(ansatz.costs, probabilities))
    if objective_kind == "cvar":
        return cvar_lower_tail(ansatz.costs, probabilities, alpha)
    raise ValueError("objective_kind must be expectation or cvar.")


def exact_distribution(ansatz: CircuitAnsatz, parameters: Sequence[float]) -> np.ndarray:
    state = Statevector.from_instruction(bind_angles(ansatz, parameters)).data
    probability = np.abs(state) ** 2
    probability /= probability.sum()
    return probability


def _spsa(
    objective,
    initial: np.ndarray,
    *,
    maxiter: int,
    seed: int,
    a: float = 0.15,
    c: float = 0.12,
    alpha: float = 0.602,
    gamma: float = 0.101,
) -> OptimizeResult:
    """Small deterministic SPSA implementation, using two evaluations per update."""
    rng = np.random.default_rng(seed)
    x = initial.astype(float, copy=True)
    best_x, best_value = x.copy(), float(objective(x))
    evaluations = 1
    for step in range(maxiter):
        ak = a / (step + 1 + 10.0) ** alpha
        ck = c / (step + 1) ** gamma
        perturbation = rng.choice((-1.0, 1.0), size=x.size)
        plus = objective(x + ck * perturbation)
        minus = objective(x - ck * perturbation)
        evaluations += 2
        gradient = ((plus - minus) / (2.0 * ck)) / perturbation
        x = (x - ak * gradient) % (2.0 * np.pi)
        value = float(objective(x))
        evaluations += 1
        if value < best_value:
            best_x, best_value = x.copy(), value
    return OptimizeResult(
        x=best_x, fun=best_value, success=False,
        message="SPSA fixed iteration budget completed; convergence was not certified.",
        nfev=evaluations,
    )


def optimize_qaoa(
    ansatz: CircuitAnsatz,
    *,
    optimizer: Literal["COBYLA", "SPSA"] = "COBYLA",
    objective_kind: ObjectiveKind = "expectation",
    seed: int = 42,
    max_evaluations: int = 60,
    alpha: float = 0.25,
    initial_parameters: Sequence[float] | None = None,
) -> VariationalFit:
    """Fit angles against exact Qiskit Statevector probabilities (noise-free expectation/CVaR)."""
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive.")
    rng = np.random.default_rng(seed)
    x0 = (
        rng.uniform(0.0, 2.0 * np.pi, len(ansatz.gamma) + len(ansatz.beta))
        if initial_parameters is None else np.asarray(initial_parameters, dtype=float)
    )
    if x0.shape != (len(ansatz.gamma) + len(ansatz.beta),):
        raise ValueError("initial_parameters has an invalid length.")
    evaluations = 0

    def objective(x):
        nonlocal evaluations
        evaluations += 1
        probability = exact_distribution(ansatz, x)
        return score_distribution(ansatz, probability, objective_kind=objective_kind, alpha=alpha)

    started = __import__("time").perf_counter()
    if optimizer == "COBYLA":
        result = minimize(
            objective, x0, method="COBYLA", bounds=[(0.0, 2 * np.pi)] * len(x0),
            options={"maxiter": max_evaluations, "rhobeg": 0.5, "tol": 1e-5},
        )
    elif optimizer == "SPSA":
        # One baseline plus three samples per update keeps the evaluation budget
        # comparable with COBYLA, whose maxiter counts objective calls.
        iterations = max(1, (max_evaluations - 1) // 3)
        result = _spsa(objective, x0, maxiter=iterations, seed=seed)
    else:
        raise ValueError("optimizer must be COBYLA or SPSA.")
    return VariationalFit(
        tuple(float(x) for x in result.x), optimizer, objective_kind, float(result.fun),
        evaluations, bool(result.success), str(result.message), seed,
        __import__("time").perf_counter() - started,
    )


def interpolated_parameters(shallow: Sequence[float], new_depth: int) -> tuple[float, ...]:
    """Interpolate separate gamma/beta schedules to seed a deeper p-layer circuit."""
    values = np.asarray(shallow, dtype=float)
    if values.ndim != 1 or values.size == 0 or values.size % 2:
        raise ValueError("shallow parameters must contain equal nonempty gamma and beta schedules.")
    old_depth = values.size // 2
    if new_depth < old_depth:
        raise ValueError("new_depth must be at least the existing depth.")
    if new_depth == old_depth:
        return tuple(float(v) for v in values)
    schedules = []
    x_old = np.linspace(0.0, 1.0, old_depth)
    x_new = np.linspace(0.0, 1.0, new_depth)
    for schedule in (values[:old_depth], values[old_depth:]):
        schedules.extend(np.interp(x_new, x_old, schedule).tolist())
    return tuple(float(v) for v in schedules)
