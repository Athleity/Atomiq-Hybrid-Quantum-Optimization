import numpy as np
import pytest

from atomiq.stage2 import Candidate, Stage2Problem
from atomiq.stage2_iws_xy import (
    _joint_distribution,
    _project_capped_simplex,
    _sector,
    run_iws_xy,
    weighted_johnson_mixer,
)


def _small_problem():
    items = tuple(
        Candidate(f"I{i}", probability=p, expected_deposit_value=v, contact_cost=c, capacity_units=h)
        for i, (p, v, c, h) in enumerate([
            (0.8, 30, 1, 1), (0.7, 25, 1, 1),
            (0.6, 20, 2, 2), (0.1, 10, 4, 4),
        ])
    )
    return Stage2Problem(items, selected_count=2, budget_limit=3, capacity_limit=3)


def test_weighted_johnson_mixer_has_biased_dicke_state_as_unique_ground_state():
    problem = _small_problem()
    masks, index = _sector(problem)
    marginals = np.array([0.7, 0.6, 0.45, 0.25])
    distribution = _joint_distribution(marginals, masks)
    ground = np.sqrt(distribution)
    mixer = weighted_johnson_mixer(marginals, masks, index)
    eigenvalues = np.linalg.eigvalsh(mixer)
    assert np.isfinite(mixer).all()
    assert mixer == pytest.approx(mixer.T, abs=1e-12)
    assert mixer @ ground == pytest.approx(-ground, abs=1e-12)
    assert eigenvalues[0] == pytest.approx(-1.0, abs=1e-12)
    assert eigenvalues[1] > eigenvalues[0] + 1e-8


def test_fixed_sum_projection_clamps_and_preserves_exact_cardinality_marginals():
    projected = _project_capped_simplex(
        np.array([0.99, 0.98, 0.01, 0.02]), total=2.0, lower=0.025, upper=0.975
    )
    assert projected.sum() == pytest.approx(2.0, abs=1e-10)
    assert projected.min() >= 0.025 - 1e-10
    assert projected.max() <= 0.975 + 1e-10


def test_iws_iterates_updates_with_fixed_angles_and_reproducible_seed():
    problem = _small_problem()
    first = run_iws_xy(
        problem, seed=17, depth=1, shots_per_iteration=8, total_shots=16,
        max_evaluations=8, inverse_temperature=2.0,
    )
    second = run_iws_xy(
        problem, seed=17, depth=1, shots_per_iteration=8, total_shots=16,
        max_evaluations=8, inverse_temperature=2.0,
    )
    assert len(first.iterations) == 2
    assert first.total_shots == 16
    assert first.gamma == second.gamma and first.beta == second.beta
    assert [row.marginal_probabilities for row in first.iterations] == [
        row.marginal_probabilities for row in second.iterations
    ]
    for row in first.iterations:
        assert sum(row.marginal_probabilities) == pytest.approx(2.0, abs=1e-10)
    assert first.mixer_ground_energy == pytest.approx(-1.0, abs=1e-10)
    assert first.mixer_ground_overlap > 1 - 1e-10
