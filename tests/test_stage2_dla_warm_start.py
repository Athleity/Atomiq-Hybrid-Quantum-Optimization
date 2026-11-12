import numpy as np
import pytest

from atomiq.stage2 import Candidate, Stage2Problem
from atomiq.stage2_dla_warm_start import (
    analyze_dla,
    dynamical_lie_closure_dimension,
    parameter_shift_derivative,
    run_dla_warm_start,
)


def _problem():
    items = tuple(
        Candidate(f"D{i}", probability=p, expected_deposit_value=v, contact_cost=c, capacity_units=h)
        for i, (p, v, c, h) in enumerate([
            (0.8, 30, 1, 1), (0.7, 25, 1, 1),
            (0.6, 20, 2, 2), (0.1, 10, 4, 4),
        ])
    )
    return Stage2Problem(items, selected_count=2, budget_limit=3, capacity_limit=3)


def test_rank_revealing_dla_closure_finds_su2_and_deduplicates_scaled_generators():
    x = np.array([[0, 1], [1, 0]], dtype=float) / 2
    y = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    z = np.array([[1, 0], [0, -1]], dtype=float) / 2
    dimension, complete = dynamical_lie_closure_dimension([x, y, z, 2 * x], tolerance=1e-10)
    assert complete
    assert dimension == 3


def test_xy_parameter_shift_rule_matches_finite_difference_with_two_spectral_gaps():
    theta = np.array([0.37])
    objective = lambda p: float(np.cos(p[0]) + 0.3 * np.cos(2 * p[0]))
    actual = parameter_shift_derivative(objective, theta, 0, "xy")
    expected = -np.sin(theta[0]) - 0.6 * np.sin(2 * theta[0])
    assert actual == pytest.approx(expected, abs=1e-12)


def test_dla_analysis_closes_on_the_fixed_cardinality_sector():
    report = analyze_dla(4, 2, tolerance=1e-10, max_dimension=100)
    assert report.sector_dimension == 6
    assert report.complete
    assert 0 < report.restricted_dimension <= report.restricted_theoretical_upper_bound
    assert report.restricted_dimension < report.full_dimension
    assert report.full_dimension <= report.full_sector_dimension_upper_bound


@pytest.mark.parametrize("parameterization,transferred,activated", [("shared", 2, 1), ("multi", 8, 6)])
def test_dla_warm_start_transfers_restricted_angles_and_preserves_fixed_k(
    parameterization, transferred, activated
):
    result = run_dla_warm_start(
        _problem(), seed=9, parameterization=parameterization,
        restricted_steps=3, full_steps=4, learning_rate=0.02,
    )
    assert result.transferred_parameter_count == transferred
    assert result.activated_zz_parameter_count == activated
    assert result.sector_leakage < 1e-10
    assert result.exact_feasible_probability >= 0
    assert result.exact_optimal_solution_probability <= 1 + 1e-12
    assert len(result.trace) == 7
