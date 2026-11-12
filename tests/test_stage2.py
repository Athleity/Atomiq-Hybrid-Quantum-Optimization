from itertools import combinations
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

from atomiq.stage2 import (
    evaluate_selection,
    build_stage2_qubo,
    enumerate_optimum,
    load_stage2_source,
    problem_from_candidates,
    solve_scipy_milp,
)
from atomiq.stage2_quantum import build_xy_qaoa_circuit, exact_qaoa_distribution


WORKBOOK = "data/raw/bank_campaign_participant_release.xlsx"


def _fixture_problem():
    operational, constraints = load_stage2_source(WORKBOOK)
    selected = operational.head(10).customer_id.astype(str).tolist()
    candidates = pd.DataFrame({
        "customer_id": selected,
        "probability": [0.10 + 0.01 * index for index in range(10)],
    })
    return problem_from_candidates(candidates, operational, constraints)


def test_stage2_loader_reads_only_operational_and_constraint_sheets():
    original = pd.read_excel
    requested_sheets = []

    def guarded_read_excel(*args, **kwargs):
        requested_sheets.append(kwargs.get("sheet_name"))
        assert kwargs.get("sheet_name") in {"Stage2_Data", "Stage2_Constraints"}
        return original(*args, **kwargs)

    with patch("atomiq.stage2.pd.read_excel", side_effect=guarded_read_excel):
        operational, constraints = load_stage2_source(WORKBOOK)
    assert requested_sheets == ["Stage2_Data", "Stage2_Constraints"]
    assert len(operational) == 400
    assert constraints == {
        "candidate_count": 10,
        "selected_count": 6,
        "budget_limit": 25,
        "capacity_limit": 22,
    }


def test_stage2_objective_and_constraints_are_recomputed_independently():
    problem = _fixture_problem()
    chosen = [candidate.customer_id for candidate in problem.candidates[:6]]
    result = evaluate_selection(problem, chosen)
    lookup = {candidate.customer_id: candidate for candidate in problem.candidates}
    independent_objective = sum(
        lookup[customer_id].probability * lookup[customer_id].expected_deposit_value
        - lookup[customer_id].contact_cost
        for customer_id in chosen
    )
    assert result.selected_count == 6
    assert result.total_contact_cost == sum(lookup[c].contact_cost for c in chosen)
    assert result.total_capacity == sum(lookup[c].capacity_units for c in chosen)
    assert result.objective == pytest.approx(independent_objective, abs=1e-12)
    assert result.feasible == (
        result.total_contact_cost <= problem.budget_limit
        and result.total_capacity <= problem.capacity_limit
    )


def test_exact_oracle_checks_all_210_six_of_ten_combinations():
    problem = _fixture_problem()
    best, feasible = enumerate_optimum(problem)
    assert len(list(combinations(problem.candidates, 6))) == 210
    assert 1 <= len(feasible) <= 210
    assert best.feasible and best.selected_count == 6
    assert best.total_contact_cost <= 25
    assert best.total_capacity <= 22
    independent_best = max(
        (
            sum(candidate.expected_net_value for candidate in subset),
            tuple(sorted(candidate.customer_id for candidate in subset)),
        )
        for subset in combinations(problem.candidates, 6)
        if sum(candidate.contact_cost for candidate in subset) <= 25
        and sum(candidate.capacity_units for candidate in subset) <= 22
    )
    assert best.objective == pytest.approx(independent_best[0])
    assert best.customer_ids == independent_best[1]


def test_scipy_milp_independently_matches_exhaustive_oracle():
    problem = _fixture_problem()
    enumerated, _ = enumerate_optimum(problem)
    milp_result = solve_scipy_milp(problem)
    assert milp_result.feasible
    assert milp_result.selected_count == 6
    assert milp_result.objective == pytest.approx(enumerated.objective, abs=1e-8)


def test_selection_rejects_duplicate_and_unknown_ids():
    problem = _fixture_problem()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_selection(problem, [problem.candidates[0].customer_id] * 6)
    with pytest.raises(ValueError, match="outside its candidate pool"):
        evaluate_selection(problem, ["unknown"] + [c.customer_id for c in problem.candidates[:5]])


def test_probability_join_requires_exact_candidate_ids():
    operational, constraints = load_stage2_source(WORKBOOK)
    candidates = pd.DataFrame({
        "customer_id": [f"not-in-stage2-{i}" for i in range(10)],
        "probability": [0.2] * 10,
    })
    with pytest.raises(ValueError, match="No Stage2_Data inputs found"):
        problem_from_candidates(candidates, operational, constraints)


def test_slack_qubo_encodes_exact_feasible_states_with_matching_objective():
    problem = _fixture_problem()
    qubo = build_stage2_qubo(problem)
    optimum, feasible = enumerate_optimum(problem)
    assert qubo.x_variables == 10
    assert len(qubo.variable_names) == 20
    assert qubo.penalty > sum(abs(c.expected_net_value) for c in problem.candidates)
    exact_count_energies = []
    for row in feasible:
        chosen = set(row.customer_ids)
        selection = tuple(int(c.customer_id in chosen) for c in problem.candidates)
        expanded_bits = qubo.candidate_bits(selection)
        assert len(expanded_bits) == 20
        assert qubo.energy(expanded_bits) == pytest.approx(-row.objective, abs=1e-10)
        exact_count_energies.append(qubo.energy(expanded_bits))
    assert min(exact_count_energies) == pytest.approx(-optimum.objective, abs=1e-10)


def test_binary_slack_weights_represent_every_required_integer():
    problem = _fixture_problem()
    qubo = build_stage2_qubo(problem)
    from atomiq.stage2 import _encode_consecutive_slack

    for key, maximum in (("budget", 25), ("capacity", 22)):
        weights = qubo.slack_weights[key]
        assert sum(weights) == maximum
        for target in range(maximum + 1):
            encoded = _encode_consecutive_slack(weights, target)
            assert sum(value * bit for value, bit in zip(weights, encoded)) == target


def test_dicke_state_and_xy_mixer_preserve_exact_six_cardinality():
    problem = _fixture_problem()
    circuit = build_xy_qaoa_circuit(problem, gamma=[0.17], beta=[0.31], topology="cycle")
    probabilities = exact_qaoa_distribution(problem, circuit)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.count_nonzero(probabilities > 1e-12) == 210
    outside_sector = sum(
        probability
        for mask, probability in enumerate(probabilities)
        if mask.bit_count() != 6
    )
    assert outside_sector < 1e-12

