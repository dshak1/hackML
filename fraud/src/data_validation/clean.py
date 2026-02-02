from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .schema import ValidationSchema
from .validators import _detect_amount_spikes, _expected_balance_deltas


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coerce_string(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def _build_features(
    df: pd.DataFrame,
    schema: ValidationSchema,
    amount_spike_values: List[float],
    amount_outlier_threshold: float,
) -> pd.DataFrame:
    amount = df["amount"].astype(float)
    orig_delta = df["oldbalanceOrg"] - df["newbalanceOrig"]
    dest_delta = df["newbalanceDest"] - df["oldbalanceDest"]

    expected_orig, expected_dest = _expected_balance_deltas(df)
    balance_error_orig = (orig_delta - expected_orig).abs()
    balance_error_dest = (dest_delta - expected_dest).abs()

    amount_negative_flag = amount < 0
    amount_is_round_number = np.isclose(amount % 1, 0)
    amount_spike_flag = amount.isin(amount_spike_values)
    amount_outlier_flag = amount > amount_outlier_threshold
    is_merchant_dest = df["nameDest"].astype("string").str.startswith("M")

    features = pd.DataFrame(
        {
            schema.id_col: df[schema.id_col] if schema.id_col in df.columns else None,
            "log1p_amount": np.log1p(amount.where(amount >= 0)),
            "orig_delta": orig_delta,
            "dest_delta": dest_delta,
            "balance_error_orig": balance_error_orig,
            "balance_error_dest": balance_error_dest,
            "balance_error_orig_high": balance_error_orig
            > schema.thresholds.balance_error_high_threshold,
            "balance_error_dest_high": balance_error_dest
            > schema.thresholds.balance_error_high_threshold,
            "is_merchant_dest": is_merchant_dest,
            "amount_is_round_number": amount_is_round_number,
            "amount_spike_flag": amount_spike_flag,
            "amount_outlier_flag": amount_outlier_flag,
            "amount_negative_flag": amount_negative_flag,
            "orig_balance_zero_before": df["oldbalanceOrg"] == 0,
            "dest_balance_zero_before": df["oldbalanceDest"] == 0,
            "orig_balance_zero_after": df["newbalanceOrig"] == 0,
            "dest_balance_zero_after": df["newbalanceDest"] == 0,
        }
    )

    return features


def clean_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: ValidationSchema,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_train = train_df.copy()
    clean_test = test_df.copy()

    numeric_cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        schema.id_col,
        schema.target_col,
    ]
    string_cols = ["type", "nameOrig", "nameDest"]

    clean_train = _coerce_numeric(clean_train, numeric_cols)
    clean_test = _coerce_numeric(clean_test, numeric_cols)
    clean_train = _coerce_string(clean_train, string_cols)
    clean_test = _coerce_string(clean_test, string_cols)

    amount = clean_train["amount"].astype(float)
    outlier_threshold = float(
        amount.quantile(schema.thresholds.amount_outlier_quantile)
    )
    amount_spike_values = _detect_amount_spikes(
        amount,
        schema.thresholds.amount_spike_top_n,
        schema.thresholds.amount_spike_min_count,
    )

    train_features = _build_features(
        clean_train, schema, amount_spike_values, outlier_threshold
    )
    train_features["split"] = "train"
    test_features = _build_features(
        clean_test, schema, amount_spike_values, outlier_threshold
    )
    test_features["split"] = "test"

    validation_features = pd.concat([train_features, test_features], ignore_index=True)

    return clean_train, clean_test, validation_features
