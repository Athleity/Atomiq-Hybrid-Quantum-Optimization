"""Classical model portfolio and scikit-learn-compatible CatBoost adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from atomiq.preprocessing import make_preprocessor
from atomiq.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class DecisionOnlySVC(SVC):
    """SVC that deliberately exposes only decision scores for external calibration.

    scikit-learn 1.9 deprecates its internal probability mode. Omitting the
    probability parameter from clone parameters avoids opting into that path;
    probabilities are learned with CalibratedClassifierCV in the shared runner.
    """

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        parameters = super().get_params(deep=deep)
        parameters.pop("probability", None)
        return parameters


class NativeCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """CatBoost wrapper that learns missing-value handling within each fit."""

    def __init__(
        self,
        *,
        iterations: int = 200,
        depth: int = 5,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        auto_class_weights: str | None = None,
        random_state: int = 42,
        thread_count: int = 1,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X: pd.DataFrame, y: Any) -> NativeCatBoostClassifier:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NativeCatBoostClassifier requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.numeric_features_ = tuple(name for name in NUMERIC_FEATURES if name in X.columns)
        self.categorical_features_ = tuple(name for name in CATEGORICAL_FEATURES if name in X.columns)
        if set(self.numeric_features_) | set(self.categorical_features_) != set(X.columns):
            raise ValueError("CatBoost input columns differ from the declared Stage 1 features.")

        self.numeric_medians_ = {
            name: pd.to_numeric(X[name], errors="coerce").median()
            for name in self.numeric_features_
        }
        self.categorical_modes_ = {
            name: X[name].dropna().astype(str).mode().iloc[0]
            if not X[name].dropna().empty
            else "__MISSING__"
            for name in self.categorical_features_
        }
        transformed = self._transform(X)
        self.classes_ = np.unique(np.asarray(y))
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights=self.auto_class_weights,
            random_seed=self.random_state,
            thread_count=self.thread_count,
            verbose=False,
            allow_writing_files=False,
            loss_function="Logloss",
        )
        self.model_.fit(transformed, y, cat_features=list(self.categorical_features_))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError("CatBoost feature names/order do not match fitted data.")
        result = X.copy()
        for name in self.numeric_features_:
            result[name] = pd.to_numeric(result[name], errors="coerce").fillna(
                self.numeric_medians_[name]
            )
        for name in self.categorical_features_:
            result[name] = result[name].where(result[name].notna(), self.categorical_modes_[name])
            result[name] = result[name].astype(str)
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before predict_proba.")
        return self.model_.predict_proba(self._transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    parameter_distributions: dict[str, list[Any]]


def _pipeline(preprocessor: object, classifier: BaseEstimator) -> Pipeline:
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def build_model_portfolio(seed: int) -> list[ModelSpec]:
    """Return the predeclared models and compact randomized search spaces."""
    dense = lambda scale: make_preprocessor(scale_numeric=scale, dense=True)
    baseline_lr = _pipeline(
        dense(True),
        LogisticRegression(C=np.inf, l1_ratio=0.0, solver="lbfgs", max_iter=3000),
    )
    regularized_lr = _pipeline(
        dense(True),
        LogisticRegression(solver="saga", max_iter=20000, tol=1e-4, random_state=seed),
    )
    linear_svm = _pipeline(
        dense(True), DecisionOnlySVC(kernel="linear", random_state=seed, cache_size=256)
    )
    rbf_svm = _pipeline(
        dense(True), DecisionOnlySVC(kernel="rbf", random_state=seed, cache_size=256)
    )
    forest = _pipeline(
        dense(False), RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=1)
    )
    extra_trees = _pipeline(
        dense(False), ExtraTreesClassifier(n_estimators=160, random_state=seed, n_jobs=1)
    )
    hist_gradient = _pipeline(
        dense(False), HistGradientBoostingClassifier(random_state=seed, early_stopping=False)
    )
    catboost = NativeCatBoostClassifier(random_state=seed, thread_count=1)
    lightgbm = _pipeline(
        dense(False), LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1)
    )
    prior = DummyClassifier(strategy="prior")

    return [
        ModelSpec("prior_baseline", prior, {}),
        ModelSpec("logistic_unregularized", baseline_lr, {}),
        ModelSpec(
            "logistic_regularized",
            regularized_lr,
            {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            "svm_linear",
            linear_svm,
            {
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            "svm_rbf",
            rbf_svm,
            {
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            "random_forest",
            forest,
            {
                "model__max_features": ["sqrt", 0.5, 0.8],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            "extra_trees",
            extra_trees,
            {
                "model__max_features": ["sqrt", 0.5, 0.8],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            "hist_gradient_boosting",
            hist_gradient,
            {
                "model__learning_rate": [0.03, 0.06, 0.12],
                "model__max_iter": [60, 100, 160],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0.0, 1.0, 5.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            "catboost_native",
            catboost,
            {
                "iterations": [40, 80, 140],
                "depth": [3, 4, 5],
                "learning_rate": [0.03, 0.06, 0.1],
                "l2_leaf_reg": [2.0, 5.0, 10.0],
                "auto_class_weights": [None, "Balanced"],
            },
        ),
        ModelSpec(
            "lightgbm",
            lightgbm,
            {
                "model__n_estimators": [80, 140, 220],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__num_leaves": [7, 15, 31],
                "model__max_depth": [-1, 3, 5],
                "model__class_weight": [None, "balanced"],
                "model__reg_lambda": [0.0, 1.0, 5.0],
            },
        ),
    ]
