"""Training-only and final Stage 2 data interfaces plus an exact oracle.

The optimizer consumes the ten Stage 1 candidates explicitly supplied by its
caller. It never reads Stage1_Test or chooses candidates itself. This keeps
candidate ranking upstream and makes the development/final boundary auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Candidate:
    customer_id: str
    probability: float
    expected_deposit_value: int
    contact_cost: int
    capacity_units: int

    @property
    def expected_net_value(self) -> float:
        return self.probability * self.expected_deposit_value - self.contact_cost


@dataclass(frozen=True)
class Stage2Problem:
    candidates: tuple[Candidate, ...]
    selected_count: int
    budget_limit: int
    capacity_limit: int

    def __post_init__(self) -> None:
        ids = [c.customer_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Stage 2 candidate customer_id values must be unique.")
        if len(self.candidates) < self.selected_count or self.selected_count < 1:
            raise ValueError("Candidate count must be at least the positive selected_count.")
        if self.budget_limit < 0 or self.capacity_limit < 0:
            raise ValueError("Budget and capacity limits must be nonnegative.")
        if not all(np.isfinite(c.probability) and 0.0 <= c.probability <= 1.0 for c in self.candidates):
            raise ValueError("Candidate probabilities must be finite and in [0, 1].")
        if not all(c.expected_deposit_value >= 0 and c.contact_cost >= 0 and c.capacity_units >= 0 for c in self.candidates):
            raise ValueError("Stage 2 values, costs, and capacity units must be nonnegative.")


@dataclass(frozen=True)
class SelectionEvaluation:
    customer_ids: tuple[str, ...]
    selected_count: int
    total_contact_cost: int
    total_capacity: int
    objective: float
    feasible: bool


def load_stage2_source(workbook_path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read only the operational inputs and public Stage 2 constraint sheet."""
    operational = pd.read_excel(
        workbook_path,
        sheet_name="Stage2_Data",
        engine="openpyxl",
        usecols=["customer_id", "contact_cost", "expected_deposit_value", "capacity_units"],
    )
    constraint_frame = pd.read_excel(
        workbook_path, sheet_name="Stage2_Constraints", engine="openpyxl"
    )
    required = {"customer_id", "contact_cost", "expected_deposit_value", "capacity_units"}
    if set(operational.columns) != required:
        raise ValueError(f"Unexpected Stage2_Data columns: {list(operational.columns)}")
    if operational.customer_id.isna().any() or operational.customer_id.astype(str).duplicated().any():
        raise ValueError("Stage2_Data must have unique, nonmissing customer IDs.")
    operational = operational.copy()
    operational["customer_id"] = operational.customer_id.astype(str)
    for col in ("contact_cost", "expected_deposit_value", "capacity_units"):
        numeric = pd.to_numeric(operational[col], errors="raise")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"Stage2_Data {col} must contain nonmissing integer values.")
        operational[col] = numeric.astype(int)

    constraints: dict[str, int] = {}
    symbols = {"N": "candidate_count", "K": "selected_count", "C": "budget_limit", "H": "capacity_limit"}
    if not {"symbol", "value"}.issubset(constraint_frame.columns):
        raise ValueError("Stage2_Constraints must contain symbol and value columns.")
    for row in constraint_frame.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        if symbol in symbols:
            constraints[symbols[symbol]] = int(getattr(row, "value"))
    if set(constraints) != set(symbols.values()):
        raise ValueError("Stage2_Constraints is missing N, K, C, or H.")
    return operational, constraints


def problem_from_candidates(
    candidates: pd.DataFrame,
    operational: pd.DataFrame,
    constraints: dict[str, int],
) -> Stage2Problem:
    """Join explicit top-ten probabilities to Stage2_Data by exact ID."""
    if "customer_id" not in candidates or "probability" not in candidates:
        raise ValueError("Candidate frame requires customer_id and probability columns.")
    if candidates.customer_id.isna().any() or candidates.customer_id.astype(str).duplicated().any():
        raise ValueError("Candidate IDs must be unique and nonmissing.")
    if len(candidates) != constraints["candidate_count"]:
        raise ValueError(
            f"Expected {constraints['candidate_count']} Stage 1 candidates, received {len(candidates)}."
        )
    frame = candidates.loc[:, ["customer_id", "probability"]].copy()
    frame["customer_id"] = frame.customer_id.astype(str)
    frame["probability"] = pd.to_numeric(frame.probability, errors="raise")
    joined = frame.merge(operational, on="customer_id", how="left", validate="one_to_one", indicator=True)
    missing = joined.loc[joined._merge != "both", "customer_id"].tolist()
    if missing:
        raise ValueError(f"No Stage2_Data inputs found for candidate IDs: {missing}")
    joined = joined.drop(columns="_merge")
    items = tuple(
        Candidate(
            customer_id=row.customer_id,
            probability=float(row.probability),
            expected_deposit_value=int(row.expected_deposit_value),
            contact_cost=int(row.contact_cost),
            capacity_units=int(row.capacity_units),
        )
        for row in joined.itertuples(index=False)
    )
    return Stage2Problem(
        items,
        selected_count=constraints["selected_count"],
        budget_limit=constraints["budget_limit"],
        capacity_limit=constraints["capacity_limit"],
    )


def evaluate_selection(problem: Stage2Problem, selected: Iterable[str]) -> SelectionEvaluation:
    """Recompute all constraints and the objective from source coefficients."""
    ids = tuple(str(value) for value in selected)
    if len(ids) != len(set(ids)):
        raise ValueError("Selection contains duplicate customer IDs.")
    by_id = {candidate.customer_id: candidate for candidate in problem.candidates}
    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        raise ValueError(f"Selection contains IDs outside its candidate pool: {unknown}")
    chosen = [by_id[customer_id] for customer_id in ids]
    total_cost = sum(candidate.contact_cost for candidate in chosen)
    total_capacity = sum(candidate.capacity_units for candidate in chosen)
    objective = float(sum(candidate.expected_net_value for candidate in chosen))
    feasible = (
        len(ids) == problem.selected_count
        and total_cost <= problem.budget_limit
        and total_capacity <= problem.capacity_limit
    )
    return SelectionEvaluation(
        tuple(sorted(ids)), len(ids), total_cost, total_capacity, objective, feasible
    )


def enumerate_optimum(problem: Stage2Problem) -> tuple[SelectionEvaluation, list[SelectionEvaluation]]:
    """Enumerate every fixed-cardinality subset; return optimum and feasible set."""
    by_id = {candidate.customer_id: candidate for candidate in problem.candidates}
    all_selections = (
        evaluate_selection(problem, ids)
        for ids in combinations(by_id, problem.selected_count)
    )
    feasible = [evaluation for evaluation in all_selections if evaluation.feasible]
    if not feasible:
        raise ValueError("Stage 2 instance has no feasible exact-cardinality selection.")
    # Deterministic tie break: lexicographically smallest sorted customer ID tuple.
    feasible.sort(key=lambda row: (-row.objective, row.customer_ids))
    return feasible[0], feasible


@dataclass(frozen=True)
class QUBO:
    """A minimization QUBO stored as constant + linear + upper-triangle terms."""

    variable_names: tuple[str, ...]
    constant: float
    linear: tuple[float, ...]
    quadratic: dict[tuple[int, int], float]
    penalty: float
    x_variables: int
    slack_weights: dict[str, tuple[int, ...]]
    _candidates: tuple[Candidate, ...]
    _budget_limit: int
    _capacity_limit: int

    def energy(self, bits: Sequence[int]) -> float:
        if len(bits) != len(self.variable_names) or any(bit not in (0, 1) for bit in bits):
            raise ValueError("QUBO bit vector must match its binary variable count.")
        value = self.constant + sum(coefficient * bit for coefficient, bit in zip(self.linear, bits))
        value += sum(coefficient * bits[i] * bits[j] for (i, j), coefficient in self.quadratic.items())
        return float(value)

    def candidate_bits(self, selection: Sequence[int]) -> tuple[int, ...]:
        """Add exact binary slacks when both inequalities are satisfied."""
        if len(selection) != self.x_variables or any(bit not in (0, 1) for bit in selection):
            raise ValueError("Selection vector must be a binary vector over candidate variables.")
        # The coefficient vectors were created as consecutive subset-sum encodings.
        budget_used = sum(c.contact_cost * bit for c, bit in zip(self._candidates, selection))
        capacity_used = sum(c.capacity_units * bit for c, bit in zip(self._candidates, selection))
        if budget_used > self._budget_limit or capacity_used > self._capacity_limit:
            raise ValueError("Cannot encode positive slack for an infeasible selection.")
        return tuple(selection) + _encode_consecutive_slack(
            self.slack_weights["budget"], self._budget_limit - budget_used
        ) + _encode_consecutive_slack(
            self.slack_weights["capacity"], self._capacity_limit - capacity_used
        )


def _consecutive_slack_weights(maximum: int) -> tuple[int, ...]:
    """Construct binary subset weights representing every integer 0..maximum."""
    if maximum < 0:
        raise ValueError("Slack maximum must be nonnegative.")
    weights: list[int] = []
    represented = 0
    next_power = 1
    while represented < maximum:
        # Powers of two cover the prefix, and the last weight is capped to the
        # remaining range; the standard complete-sequence condition holds.
        weight = min(next_power, maximum - represented)
        weights.append(weight)
        represented += weight
        next_power *= 2
    return tuple(weights)


def _encode_consecutive_slack(weights: Sequence[int], target: int) -> tuple[int, ...]:
    """Return a deterministic subset-sum representation (weights cover a prefix)."""
    # Dynamic programming avoids assuming powers-of-two; this also handles a
    # capped final coefficient such as [1, 2, 4, 8, 10] for a maximum of 25.
    reachable: dict[int, tuple[int, ...]] = {0: tuple(0 for _ in weights)}
    for index, weight in enumerate(weights):
        for subtotal, bits in list(reachable.items()):
            candidate = subtotal + weight
            if candidate <= target and candidate not in reachable:
                updated = list(bits)
                updated[index] = 1
                reachable[candidate] = tuple(updated)
    if target not in reachable:
        raise ValueError(f"Slack encoding cannot represent {target}.")
    return reachable[target]


def build_stage2_qubo(problem: Stage2Problem) -> QUBO:
    """Build an exact-cardinality, two-slack QUBO for independent auditing.

    Minimizing `-objective` is the objective convention. A penalty larger than
    the full objective range makes any state violating one or more equalities
    strictly worse than every zero-penalty feasible state.
    """
    n = len(problem.candidates)
    budget_slack = _consecutive_slack_weights(problem.budget_limit)
    capacity_slack = _consecutive_slack_weights(problem.capacity_limit)
    names = tuple([f"x_{i}" for i in range(n)]
                  + [f"s_budget_{i}" for i in range(len(budget_slack))]
                  + [f"s_capacity_{i}" for i in range(len(capacity_slack))])
    net = [c.expected_net_value for c in problem.candidates]
    objective_range = sum(abs(value) for value in net)
    penalty = objective_range + 1.0
    linear = [0.0] * len(names)
    quadratic: dict[tuple[int, int], float] = {}
    constant = 0.0

    for index, value in enumerate(net):
        linear[index] -= value

    def add_squared_equality(coefficients: dict[int, float], target: float) -> None:
        nonlocal constant
        constant += penalty * target * target
        entries = sorted(coefficients.items())
        for index, coefficient in entries:
            linear[index] += penalty * (coefficient * coefficient - 2.0 * target * coefficient)
        for position, (left, left_coefficient) in enumerate(entries):
            for right, right_coefficient in entries[position + 1:]:
                quadratic[left, right] = quadratic.get((left, right), 0.0) + (
                    2.0 * penalty * left_coefficient * right_coefficient
                )

    add_squared_equality({i: 1.0 for i in range(n)}, problem.selected_count)
    budget_coefficients = {i: c.contact_cost for i, c in enumerate(problem.candidates)}
    budget_coefficients.update({n + i: value for i, value in enumerate(budget_slack)})
    add_squared_equality(budget_coefficients, problem.budget_limit)
    cap_offset = n + len(budget_slack)
    capacity_coefficients = {i: c.capacity_units for i, c in enumerate(problem.candidates)}
    capacity_coefficients.update({cap_offset + i: value for i, value in enumerate(capacity_slack)})
    add_squared_equality(capacity_coefficients, problem.capacity_limit)

    return QUBO(
        variable_names=names,
        constant=constant,
        linear=tuple(linear),
        quadratic=quadratic,
        penalty=penalty,
        x_variables=n,
        slack_weights={"budget": budget_slack, "capacity": capacity_slack},
        _candidates=problem.candidates,
        _budget_limit=problem.budget_limit,
        _capacity_limit=problem.capacity_limit,
    )


def solve_scipy_milp(problem: Stage2Problem) -> SelectionEvaluation:
    """Solve the linear integer model independently of the subset enumerator."""
    n = len(problem.candidates)
    matrix = np.asarray([
        np.ones(n),
        [candidate.contact_cost for candidate in problem.candidates],
        [candidate.capacity_units for candidate in problem.candidates],
    ], dtype=float)
    lower = np.asarray([problem.selected_count, -np.inf, -np.inf], dtype=float)
    upper = np.asarray([
        problem.selected_count, problem.budget_limit, problem.capacity_limit
    ], dtype=float)
    result = milp(
        c=-np.asarray([candidate.expected_net_value for candidate in problem.candidates]),
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise ValueError(f"Independent MILP optimizer failed: {result.message}")
    selected = [
        candidate.customer_id
        for candidate, value in zip(problem.candidates, result.x, strict=True)
        if value > 0.5
    ]
    evaluation = evaluate_selection(problem, selected)
    if not evaluation.feasible:
        raise AssertionError("MILP returned a selection that fails independent feasibility checks.")
    return evaluation

