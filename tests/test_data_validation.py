import numpy as np
import pandas as pd

from src.data_validation.schema import ValidationSchema, RuleThresholds
from src.data_validation.validators import run_validations


def _base_frames():
    train = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER"],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "nameOrig": ["C1", "C2", "C3", "C4"],
            "oldbalanceOrg": [1000.0, 500.0, 300.0, 800.0],
            "newbalanceOrig": [1100.0, 300.0, 0.0, 400.0],
            "nameDest": ["C5", "M1", "C6", "C7"],
            "oldbalanceDest": [0.0, 100.0, 10.0, 200.0],
            "newbalanceDest": [100.0, 300.0, 310.0, 600.0],
            "urgency_level": [0, 1, 0, 2],
            "id": [1, 2, 3, 4],
        }
    )
    test = train.drop(columns=["urgency_level"]).copy()
    return train, test


def _schema_from_frames(train, test):
    thresholds = RuleThresholds(balance_error_threshold=1.0, balance_error_high_threshold=10.0)
    return ValidationSchema(
        train_columns=list(train.columns),
        test_columns=list(test.columns),
        dtypes={col: str(dtype) for col, dtype in train.dtypes.items()},
        allowed_type_values=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        expected_type_values=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        thresholds=thresholds,
    )


def _check_map(results):
    return {c["name"]: c for c in results.get("checks", [])}


def test_missing_values_flagged():
    train, test = _base_frames()
    train.loc[0, "amount"] = np.nan
    schema = _schema_from_frames(train, test)
    results = run_validations(train, test, schema)
    checks = _check_map(results)
    assert checks["train_missing_values"]["status"] in {"warn", "fail"}


def test_bad_category_warns():
    train, test = _base_frames()
    train.loc[0, "type"] = "UNKNOWN"
    schema = _schema_from_frames(train, test)
    results = run_validations(train, test, schema)
    checks = _check_map(results)
    assert checks["type_invalid_categories"]["status"] == "fail"


def test_negative_amount_balance():
    train, test = _base_frames()
    train.loc[1, "amount"] = -5.0
    train.loc[2, "oldbalanceOrg"] = -10.0
    schema = _schema_from_frames(train, test)
    results = run_validations(train, test, schema)
    checks = _check_map(results)
    assert checks["amount_negative"]["status"] == "fail"
    assert checks["oldbalanceOrg_negative"]["status"] == "fail"


def test_balance_relationships_flagged():
    train, test = _base_frames()
    train.loc[0, "oldbalanceOrg"] = 100.0
    train.loc[0, "newbalanceOrig"] = 100.0
    schema = _schema_from_frames(train, test)
    results = run_validations(train, test, schema)
    checks = _check_map(results)
    assert checks["balance_error_orig"]["status"] in {"warn", "fail"}


def test_drift_detection_runs():
    train, test = _base_frames()
    test.loc[:, "amount"] = [10.0, 20.0, 30.0, 40.0]
    schema = _schema_from_frames(train, test)
    results = run_validations(train, test, schema)
    assert "drift" in results
    assert "numeric" in results["drift"]
