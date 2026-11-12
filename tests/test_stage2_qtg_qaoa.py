import numpy as np
import pytest

from atomiq.stage2 import enumerate_optimum
from atomiq.stage2_development_fixtures import four_item_binding_problem
from atomiq.stage2_qtg_qaoa import (
    qtg_amplitude_amplification_mixer,
    qtg_initial_state,
    run_qtg_qaoa,
)


def test_qtg_tree_generates_exactly_all_and_only_feasible_exact_k_states():
    problem = four_item_binding_problem()
    masks, amplitudes = qtg_initial_state(problem, bias_b=0.0)
    _, feasible = enumerate_optimum(problem)
    expected_masks = {
        sum(1 << i for i, candidate in enumerate(problem.candidates) if candidate.customer_id in row.customer_ids)
        for row in feasible
    }
    assert set(masks) == expected_masks
    assert np.linalg.norm(amplitudes) == pytest.approx(1.0, abs=1e-12)
    assert np.all(amplitudes.real > 0)
    assert np.all(amplitudes.imag == 0)
    assert all(mask.bit_count() == problem.selected_count for mask in masks)


def test_qtg_amplitude_amplification_mixer_is_unitary_on_statevector():
    problem = four_item_binding_problem()
    _, phi = qtg_initial_state(problem, bias_b=0.5)
    vector = np.array([1, 1j, -1], dtype=complex)
    vector /= np.linalg.norm(vector)
    mixed = qtg_amplitude_amplification_mixer(phi, beta=0.731, vector=vector)
    assert np.linalg.norm(mixed) == pytest.approx(1.0, abs=1e-12)
    eigenvector = qtg_amplitude_amplification_mixer(phi, 0.731, phi)
    assert eigenvector == pytest.approx(np.exp(-1j * 0.731) * phi, abs=1e-12)


def test_qtg_optimizer_is_reproducible_and_reports_only_feasible_measurements():
    problem = four_item_binding_problem()
    first = run_qtg_qaoa(problem, seed=19, bias_b=0.0, max_evaluations=30, shots=512)
    second = run_qtg_qaoa(problem, seed=19, bias_b=0.0, max_evaluations=30, shots=512)
    assert first.gamma == second.gamma and first.beta == second.beta
    assert first.expected_objective == pytest.approx(second.expected_objective, abs=1e-12)
    assert first.sampled_optimal_hits == second.sampled_optimal_hits
    assert first.feasible_probability == pytest.approx(1.0, abs=1e-12)
    assert first.sampled_feasible_rate == 1.0
    assert first.feasible_state_count == 3
    assert first.positive_amplitude_state_count == 3
    assert first.logical_data_qubits_lower_bound == 14
