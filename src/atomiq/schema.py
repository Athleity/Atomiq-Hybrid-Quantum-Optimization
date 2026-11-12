"""Strict Stage 1 training-data loading without touching held-out labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ID_COLUMN = "customer_id"
TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"
NUMERIC_FEATURES = ("age", "balance", "day", "campaign", "pdays", "previous")
CATEGORICAL_FEATURES = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
)
FEATURE_COLUMNS = (ID_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
TRAIN_COLUMNS = (
    ID_COLUMN,
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class TrainingData:
    """Model inputs, row identifiers, and binary labels from Stage1_Train only."""

    customer_ids: pd.Series
    features: pd.DataFrame
    target: pd.Series


def load_training_data(workbook_path: str | Path) -> TrainingData:
    """Read and validate the labeled training sheet; never reads README or test labels."""
    frame = pd.read_excel(workbook_path, sheet_name="Stage1_Train", engine="openpyxl")
    if tuple(frame.columns) != TRAIN_COLUMNS:
        raise ValueError(
            "Unexpected Stage1_Train schema. "
            f"Expected {TRAIN_COLUMNS!r}, received {tuple(frame.columns)!r}."
        )
    if frame.empty or frame.isna().any().any():
        raise ValueError("Stage1_Train must contain rows with no missing cells.")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Stage1_Train customer_id values must be unique.")

    labels = set(frame[TARGET_COLUMN].unique())
    expected = {POSITIVE_LABEL, NEGATIVE_LABEL}
    if labels != expected:
        raise ValueError(f"Unexpected target labels: expected {expected}, received {labels}.")

    customer_ids = frame[ID_COLUMN].astype(str).rename(ID_COLUMN)
    target = frame[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).astype("int8")
    features = frame.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].copy()
    if ID_COLUMN in features or TARGET_COLUMN in features or "duration" in features:
        raise AssertionError("Identifier, target, and duration must never be predictors.")
    if tuple(features.columns) != (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES):
        raise AssertionError("Feature order unexpectedly changed during loading.")
    return TrainingData(customer_ids, features, target.rename(TARGET_COLUMN))


def load_stage1_inference_data(workbook_path: str | Path) -> tuple[pd.Series, pd.DataFrame]:
    """Load Stage1_Test identifiers and predictors without requesting any labels.

    ``usecols`` is deliberately explicit: even if a future workbook version adds
    a target-like column, this inference path never reads it into a DataFrame.
    """
    columns = (ID_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    frame = pd.read_excel(
        workbook_path,
        sheet_name="Stage1_Test",
        engine="openpyxl",
        usecols=list(columns),
    )
    if set(frame.columns) != set(columns) or len(frame.columns) != len(columns):
        raise ValueError(
            "Unexpected Stage1_Test feature schema. "
            f"Expected columns {columns!r}; received {tuple(frame.columns)!r}."
        )
    if frame.empty or frame.isna().any().any():
        raise ValueError("Stage1_Test must contain nonempty, complete inference features.")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Stage1_Test customer_id values must be unique.")
    customer_ids = frame[ID_COLUMN].astype(str).rename(ID_COLUMN)
    features = frame.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].copy()
    if tuple(features.columns) != (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES):
        raise AssertionError("Feature order unexpectedly changed during inference loading.")
    return customer_ids, features
