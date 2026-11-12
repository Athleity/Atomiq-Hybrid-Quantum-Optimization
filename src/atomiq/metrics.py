"""Probability, ranking, and reliability metrics for Stage 1."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "precision_at_10",
    "recall_at_10",
    "brier_score",
    "log_loss",
    "ece_10_bins",
    "calibration_intercept",
    "calibration_slope",
)


def stable_top_k_indices(
    probabilities: np.ndarray, customer_ids: np.ndarray, k: int
) -> np.ndarray:
    """Rank by probability descending and customer ID ascending, including ties."""
    probabilities = np.asarray(probabilities, dtype=float)
    customer_ids = np.asarray(customer_ids, dtype=str)
    if probabilities.ndim != 1 or customer_ids.ndim != 1:
        raise ValueError("Probabilities and IDs must be one-dimensional.")
    if len(probabilities) != len(customer_ids):
        raise ValueError("Probabilities and IDs must have equal lengths.")
    if len(probabilities) == 0 or not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be non-empty and finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if len(set(customer_ids.tolist())) != len(customer_ids):
        raise ValueError("Customer IDs must be unique for deterministic ranking.")
    if not 1 <= k <= len(probabilities):
        raise ValueError(f"k must be between 1 and {len(probabilities)}.")
    return np.lexsort((customer_ids, -probabilities))[:k]


def _calibration_slope_intercept(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Fit a lightly regularized diagnostic logistic recalibration line."""
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones_like(logit), logit])
    ridge = 1e-6

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ parameters
        loss = np.logaddexp(0.0, linear).sum() - np.dot(y_true, linear)
        gradient = design.T @ (expit(linear) - y_true)
        penalty = 0.5 * ridge * np.dot(parameters, parameters)
        return float(loss + penalty), gradient + ridge * parameters

    result = minimize(
        lambda parameters: objective(parameters)[0],
        np.zeros(2),
        jac=lambda parameters: objective(parameters)[1],
        method="BFGS",
    )
    if not result.success and not np.isfinite(result.fun):
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    customer_ids: np.ndarray,
    *,
    top_k: int = 10,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Compute predeclared metrics; test labels are never passed to this routine."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or len(y) != len(p) or len(p) != len(customer_ids):
        raise ValueError("Labels, probabilities, and IDs must be aligned 1D arrays.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Each validation fold must contain both target classes.")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Predicted probabilities must be finite and in [0, 1].")

    selected = stable_top_k_indices(p, np.asarray(customer_ids, dtype=str), top_k)
    positive_count = int(y.sum())
    intercept, slope = _calibration_slope_intercept(y, p)
    bin_indices = np.minimum((p * calibration_bins).astype(int), calibration_bins - 1)
    ece = 0.0
    for bin_index in range(calibration_bins):
        mask = bin_indices == bin_index
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))

    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision_at_10": float(y[selected].mean()),
        "recall_at_10": float(y[selected].sum() / positive_count),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_bins": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def make_reliability_bins(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> list[dict[str, Any]]:
    """Return equal-width calibration bins, retaining empty bins explicitly."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    indices = np.minimum((p * bins).astype(int), bins - 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(bins):
        mask = indices == bin_index
        rows.append(
            {
                "bin": bin_index,
                "lower": bin_index / bins,
                "upper": (bin_index + 1) / bins,
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "observed_positive_rate": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    return rows
