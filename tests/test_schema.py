from pathlib import Path

import pandas as pd

import atomiq.schema as schema
from atomiq.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_training_data,
)


WORKBOOK = Path("data/raw/bank_campaign_participant_release.xlsx")


def test_training_loader_uses_expected_schema_and_only_training_labels():
    data = load_training_data(WORKBOOK)
    assert len(data.target) == 300
    assert int(data.target.sum()) == 35
    assert int((data.target == 0).sum()) == 265
    assert data.customer_ids.is_unique
    assert tuple(data.features.columns) == (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def test_identifier_target_and_duration_are_not_predictors():
    data = load_training_data(WORKBOOK)
    assert "customer_id" not in data.features.columns
    assert "y" not in data.features.columns
    assert "duration" not in data.features.columns


def test_test_inference_loader_requests_only_id_and_features(monkeypatch):
    expected_columns = ["customer_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    values = {column: ["C0001"] if column == "customer_id" else [1] for column in expected_columns}
    observed = {}

    def fake_read_excel(path, *, sheet_name, engine, usecols):
        observed.update({"sheet_name": sheet_name, "usecols": list(usecols)})
        return pd.DataFrame(values)

    monkeypatch.setattr(schema.pd, "read_excel", fake_read_excel)
    ids, features = schema.load_stage1_inference_data("workbook.xlsx")
    assert observed["sheet_name"] == "Stage1_Test"
    assert set(observed["usecols"]) == set(expected_columns)
    assert "y" not in observed["usecols"]
    assert "duration" not in observed["usecols"]
    assert ids.tolist() == ["C0001"]
    assert tuple(features.columns) == (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
