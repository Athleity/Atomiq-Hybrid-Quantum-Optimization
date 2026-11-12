from itertools import combinations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from atomiq.stage2 import Candidate, Stage2Problem, evaluate_selection
from atomiq.stage2_if_qaoa import (
    _append_slack_qpe,
    append_if_cost_phase,
    build_if_qaoa_circuit,
    selector_distribution,
    signed_register_width,
)


def _binding_problem():
    candidates = tuple(
        Candidate(f"C{i}", probability=[0.8, 0.75, 0.7, 0.05][i],
                  expected_deposit_value=[20, 22, 18, 23][i], contact_cost=[2, 2, 1, 4][i],
                  capacity_units=[1, 2, 2, 1][i])
        for i in range(4)
    )
    return Stage2Problem(candidates, selected_count=2, budget_limit=3, capacity_limit=3)


def test_signed_qpe_register_width_contains_every_slack_without_wraparound():
    problem = _binding_problem()
    for index, (coefficients, limit) in enumerate((
        ([c.contact_cost for c in problem.candidates], problem.budget_limit),
        ([c.capacity_units for c in problem.candidates], problem.capacity_limit),
    )):
        width = signed_register_width(limit, coefficients)
        assert max(limit, abs(limit - sum(coefficients))) < 2 ** (width - 1)
        assert width == (4 if index == 0 else 3)


@pytest.mark.parametrize("limit,weights", [(2.5, [1, 2]), (-1, [1, 2]), (2, [1, 1.5])])
def test_signed_register_rejects_noninteger_or_negative_constraint_data(limit, weights):
    with pytest.raises(ValueError):
        signed_register_width(limit, weights)


def test_qpe_encodes_positive_and_negative_slack_sign_bit_exactly():
    problem = _binding_problem()
    coefficients = [c.contact_cost for c in problem.candidates]
    width = signed_register_width(problem.budget_limit, coefficients)
    selector = list(range(4))
    estimate = list(range(4, 4 + width))
    for bits in ([1, 0, 0, 0], [1, 1, 0, 0]):
        circuit = QuantumCircuit(4 + width)
        for index, bit in enumerate(bits):
            if bit:
                circuit.x(selector[index])
        _append_slack_qpe(
            circuit, problem, selector, estimate,
            weights=coefficients, limit=problem.budget_limit,
        )
        state = Statevector.from_instruction(circuit).data
        probabilities = np.abs(state) ** 2
        observed = int(np.argmax(probabilities)) >> 4
        slack = problem.budget_limit - sum(
            weight * bit for weight, bit in zip(coefficients, bits)
        )
        encoded = observed
        if encoded >= 2 ** (width - 1):
            encoded -= 2**width
        assert encoded == slack
        assert probabilities.max() == pytest.approx(1.0, abs=1e-10)


def test_if_oracle_applies_objective_phase_only_to_double_feasible_selections():
    problem = _binding_problem()
    n = len(problem.candidates)
    budget_weights = [c.contact_cost for c in problem.candidates]
    capacity_weights = [c.capacity_units for c in problem.candidates]
    b_width = signed_register_width(problem.budget_limit, budget_weights)
    h_width = signed_register_width(problem.capacity_limit, capacity_weights)
    selector = list(range(n))
    budget = list(range(n, n + b_width))
    capacity = list(range(n + b_width, n + b_width + h_width))
    gamma = 0.137
    offset = sum(max(0.0, -candidate.expected_net_value) for candidate in problem.candidates)

    for indices in combinations(range(n), problem.selected_count):
        selected_ids = [problem.candidates[i].customer_id for i in indices]
        evaluation = evaluate_selection(problem, selected_ids)
        circuit = QuantumCircuit(n + b_width + h_width)
        for index in indices:
            circuit.x(selector[index])
        append_if_cost_phase(circuit, problem, selector, budget, capacity, gamma)
        state = Statevector.from_instruction(circuit).data
        basis_index = sum(1 << index for index in indices)
        expected_amplitude = np.exp(1j * gamma * (evaluation.objective + offset)) if evaluation.feasible else 1.0
        assert state[basis_index] == pytest.approx(expected_amplitude, abs=1e-9)
        assert np.count_nonzero(np.abs(state) > 1e-9) == 1


def test_if_qaoa_preserves_exact_cardinality_and_clean_indicator_registers():
    problem = _binding_problem()
    circuit = build_if_qaoa_circuit(problem, gamma=[0.2], beta=[0.1])
    marginal = selector_distribution(problem, circuit)
    assert marginal.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.count_nonzero(marginal > 1e-10) <= 6
    assert all(mask.bit_count() == 2 for mask, p in enumerate(marginal) if p > 1e-10)
