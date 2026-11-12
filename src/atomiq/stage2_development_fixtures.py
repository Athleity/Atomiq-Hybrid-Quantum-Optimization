"""Small deterministic binding-constraint fixtures for Stage 2 method checks.

These synthetic records are not challenge participants and must never be
combined with the final challenge evaluation. They are shared across methods
so algorithm comparisons use exactly the same objective coefficients.
"""

from atomiq.stage2 import Candidate, Stage2Problem


def four_item_binding_problem() -> Stage2Problem:
    candidates = tuple(
        Candidate(
            f"SYN{i}", probability=probability,
            expected_deposit_value=value,
            contact_cost=contact_cost,
            capacity_units=capacity,
        )
        for i, (probability, value, contact_cost, capacity) in enumerate([
            (0.8, 30, 1, 1),
            (0.7, 25, 1, 1),
            (0.6, 20, 2, 2),
            (0.1, 10, 4, 4),
        ])
    )
    return Stage2Problem(
        candidates, selected_count=2, budget_limit=3, capacity_limit=3
    )
