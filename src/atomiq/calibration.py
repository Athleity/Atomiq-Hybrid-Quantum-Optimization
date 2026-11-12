"""Monotone sigmoid probability calibration using fold-local OOF scores."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


def _stable_logit(scores: np.ndarray) -> np.ndarray:
    """Avoid unnecessary probability clipping that would collapse valid ranks."""
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    return logit(np.clip(np.asarray(scores, dtype=float), lower, upper))


def positive_class_scores(estimator: Any, X: Any) -> tuple[np.ndarray, str]:
    """Get class-1 probabilities, or map binary class-1 decision margins by sigmoid."""
    try:
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        source = "estimator_predict_proba"
        classes = np.asarray(estimator.classes_)
        positive_position = np.flatnonzero(classes == 1)
        if probabilities.ndim != 2 or len(positive_position) != 1:
            raise ValueError("Estimator probability columns must include positive class 1 once.")
        scores = probabilities[:, int(positive_position[0])]
    except (AttributeError, NotImplementedError):
        margins = np.asarray(estimator.decision_function(X), dtype=float)
        classes = np.asarray(estimator.classes_)
        if margins.ndim != 1 or len(classes) != 2 or classes[1] != 1:
            raise ValueError("Decision-score calibration requires binary classes [0, 1].")
        scores = expit(margins)
        source = "fixed_sigmoid_of_decision_function"
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Estimator returned invalid class-1 scores.")
    return scores, source


class PositiveSigmoidCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Cross-fit a non-decreasing Platt map, then refit the base estimator.

    The slope is parameterized as exp(theta), so calibration can correct
    probability scale without reversing the estimator's ranking. Calibration
    scores are generated out of fold; the final base estimator is then fitted
    on all rows supplied to fit.
    """

    def __init__(self, estimator: BaseEstimator, *, ridge: float = 1e-3) -> None:
        self.estimator = estimator
        self.ridge = ridge

    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        *,
        cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> PositiveSigmoidCalibratedClassifier:
        y_array = np.asarray(y, dtype=int)
        if set(np.unique(y_array)) != {0, 1}:
            raise ValueError("Calibration training data must contain both binary classes.")
        out_of_fold_scores = np.full(len(y_array), np.nan, dtype=float)
        validation_counts = np.zeros(len(y_array), dtype=int)
        for train_indices, calibration_indices in cv_splits:
            fold_model = clone(self.estimator)
            fold_model.fit(X.iloc[train_indices], y_array[train_indices])
            scores, _ = positive_class_scores(fold_model, X.iloc[calibration_indices])
            out_of_fold_scores[calibration_indices] = scores
            validation_counts[calibration_indices] += 1
        if np.any(validation_counts != 1) or not np.isfinite(out_of_fold_scores).all():
            raise ValueError("Calibration CV must assign every training row to validation once.")

        logits = _stable_logit(out_of_fold_scores)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, log_slope = parameters
            slope = np.exp(np.clip(log_slope, -10.0, 10.0))
            linear_predictor = intercept + slope * logits
            probabilities = expit(linear_predictor)
            loss = np.mean(np.logaddexp(0.0, linear_predictor) - y_array * linear_predictor)
            loss += 0.5 * self.ridge * log_slope**2
            residual = probabilities - y_array
            gradient = np.asarray(
                [residual.mean(), np.mean(residual * slope * logits) + self.ridge * log_slope]
            )
            return float(loss), gradient

        result = minimize(
            lambda parameters: objective(parameters)[0],
            x0=np.asarray([logit(np.mean(y_array)), 0.0]),
            jac=lambda parameters: objective(parameters)[1],
            method="L-BFGS-B",
            bounds=[(None, None), (-10.0, 10.0)],
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Monotone sigmoid calibration failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.slope_ = float(np.exp(result.x[1]))
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.estimator_ = clone(self.estimator).fit(X, y_array)
        self.n_features_in_ = len(getattr(self.estimator_, "feature_names_in_", [])) or X.shape[1]
        self.calibration_oof_scores_ = out_of_fold_scores
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["estimator_", "intercept_", "slope_"])
        scores, _ = positive_class_scores(self.estimator_, X)
        calibrated = expit(self.intercept_ + self.slope_ * _stable_logit(scores))
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
