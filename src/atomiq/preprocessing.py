"""Fold-local mixed-type preprocessing shared by classical and quantum adapters."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from atomiq.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor(*, scale_numeric: bool, dense: bool = True) -> ColumnTransformer:
    """Create an unfitted transformer; callers fit it inside each training fold."""
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                    dtype=float,
                ),
            ),
        ]
    )
    numeric = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense else 1.0,
        verbose_feature_names_out=False,
    )
