"""Small, synthetic workbook fixtures for tests that must run without challenge data."""

from pathlib import Path

import pandas as pd
import pytest

from atomiq.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture
def synthetic_workbook(tmp_path: Path) -> Path:
    """Create the minimum valid Stage 1/2 workbook in a temporary directory.

    These records are generated solely for interface and algorithm tests. They
    are not challenge participants and do not encode any held-out labels.
    """
    train_rows = []
    for index in range(60):
        row = {
            "customer_id": f"SYN{index:04d}",
            "age": 20 + index % 55,
            "job": ("admin.", "technician", "services")[index % 3],
            "marital": ("single", "married")[index % 2],
            "education": ("primary", "secondary", "tertiary")[index % 3],
            "default": ("no", "yes")[index % 2],
            "balance": (index * 37) % 4000 - 1000,
            "housing": ("no", "yes")[index % 2],
            "loan": ("no", "yes")[index % 3 == 0],
            "contact": ("cellular", "telephone")[index % 2],
            "day": 1 + index % 28,
            "month": ("jan", "may", "oct")[index % 3],
            "campaign": 1 + index % 5,
            "pdays": -1 if index % 4 == 0 else index % 90,
            "previous": index % 4,
            "poutcome": ("unknown", "failure", "success")[index % 3],
            "y": "yes" if index % 5 == 0 else "no",
        }
        train_rows.append(row)
    train = pd.DataFrame(train_rows)

    # Keep the expected features populated, even though inference tests stub the
    # loader; this makes the workbook schema internally coherent for future use.
    test_rows = []
    for index in range(4):
        row = {"customer_id": f"SYNTEST{index:03d}"}
        row.update({name: 30 + index for name in NUMERIC_FEATURES})
        row.update({name: "unknown" for name in CATEGORICAL_FEATURES})
        test_rows.append(row)
    stage1_test = pd.DataFrame(test_rows)

    stage2_data = pd.DataFrame(
        {
            "customer_id": [f"SYN{index:04d}" for index in range(10)],
            "contact_cost": [1 + index % 4 for index in range(10)],
            "expected_deposit_value": [20 + (index * 7) % 30 for index in range(10)],
            "capacity_units": [1 + (index * 3) % 5 for index in range(10)],
        }
    )
    stage2_constraints = pd.DataFrame(
        {"symbol": ["N", "K", "C", "H"], "value": [10, 6, 25, 22]}
    )
    path = tmp_path / "synthetic_challenge.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        train.to_excel(writer, sheet_name="Stage1_Train", index=False)
        stage1_test.to_excel(writer, sheet_name="Stage1_Test", index=False)
        stage2_data.to_excel(writer, sheet_name="Stage2_Data", index=False)
        stage2_constraints.to_excel(writer, sheet_name="Stage2_Constraints", index=False)
    return path
