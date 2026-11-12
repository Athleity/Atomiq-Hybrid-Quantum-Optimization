import numpy as np

from atomiq.metrics import evaluate_probabilities, stable_top_k_indices


def test_top_k_ties_use_customer_id_ascending():
    ids = np.asarray(["C3", "C1", "C2"])
    probabilities = np.asarray([0.8, 0.8, 0.8])
    np.testing.assert_array_equal(stable_top_k_indices(probabilities, ids, 2), [1, 2])


def test_metrics_include_ranking_probability_and_calibration_diagnostics():
    ids = np.asarray([f"C{i:03d}" for i in range(20)])
    y = np.asarray([0, 1] * 10)
    probabilities = np.linspace(0.05, 0.95, len(y))
    result = evaluate_probabilities(y, probabilities, ids, top_k=10)
    assert set(result) == {
        "roc_auc",
        "pr_auc",
        "precision_at_10",
        "recall_at_10",
        "brier_score",
        "log_loss",
        "ece_10_bins",
        "calibration_intercept",
        "calibration_slope",
    }
    assert all(np.isfinite(value) for value in result.values())
