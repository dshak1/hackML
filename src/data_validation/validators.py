from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import ValidationSchema


LOGGER = logging.getLogger(__name__)


def _add_check(
    results: Dict[str, object],
    name: str,
    status: str,
    severity: Optional[str],
    message: str,
    count: Optional[int] = None,
    examples: Optional[List[Dict[str, object]]] = None,
    details: Optional[Dict[str, object]] = None,
) -> None:
    results["checks"].append(
        {
            "name": name,
            "status": status,
            "severity": severity,
            "message": message,
            "count": count,
            "examples": examples or [],
            "details": details or {},
        }
    )


def _sample_df(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df
    return df.sample(sample_size, random_state=random_state)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def _collect_examples(
    df: pd.DataFrame,
    mask: pd.Series,
    columns: List[str],
    limit: int = 5,
) -> List[Dict[str, object]]:
    if mask is None or mask.sum() == 0:
        return []
    cols = [c for c in columns if c in df.columns]
    sample = df.loc[mask, cols].head(limit)
    return sample.to_dict(orient="records")


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0, 1.0
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    data_all = np.sort(np.concatenate([x_sorted, y_sorted]))
    cdf_x = np.searchsorted(x_sorted, data_all, side="right") / len(x_sorted)
    cdf_y = np.searchsorted(y_sorted, data_all, side="right") / len(y_sorted)
    ks_stat = float(np.max(np.abs(cdf_x - cdf_y)))
    en = np.sqrt(len(x_sorted) * len(y_sorted) / (len(x_sorted) + len(y_sorted)))
    p_value = float(min(1.0, 2.0 * np.exp(-2.0 * (ks_stat * en) ** 2)))
    return ks_stat, p_value


def _psi(train_freq: pd.Series, test_freq: pd.Series, eps: float = 1e-6) -> float:
    idx = train_freq.index.union(test_freq.index)
    train = train_freq.reindex(idx).fillna(0).astype(float)
    test = test_freq.reindex(idx).fillna(0).astype(float)
    train = train / max(train.sum(), eps)
    test = test / max(test.sum(), eps)
    return float(((train - test) * np.log((train + eps) / (test + eps))).sum())


def _expected_balance_deltas(
    df: pd.DataFrame, amount_col: str = "amount", type_col: str = "type"
) -> Tuple[pd.Series, pd.Series]:
    amount = df[amount_col].astype(float)
    expected_orig = amount.copy()
    expected_dest = amount.copy()

    if type_col in df.columns:
        cash_in_mask = df[type_col] == "CASH_IN"
        expected_orig.loc[cash_in_mask] = -amount.loc[cash_in_mask]

    return expected_orig, expected_dest


def _detect_amount_spikes(
    amount: pd.Series, top_n: int, min_count: int
) -> List[float]:
    counts = amount.value_counts()
    spikes = counts[counts >= min_count].head(top_n).index.tolist()
    return [float(x) for x in spikes]


def run_validations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: ValidationSchema,
) -> Dict[str, object]:
    results: Dict[str, object] = {
        "schema": schema.as_dict(),
        "train": {
            "shape": train_df.shape,
            "columns": list(train_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in train_df.dtypes.items()},
        },
        "test": {
            "shape": test_df.shape,
            "columns": list(test_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in test_df.dtypes.items()},
        },
        "checks": [],
        "drift": {},
        "class_distribution": {},
    }

    # Schema validation
    expected_train = set(schema.train_columns)
    expected_test = set(schema.test_columns)
    missing_train = sorted(expected_train - set(train_df.columns))
    extra_train = sorted(set(train_df.columns) - expected_train)
    missing_test = sorted(expected_test - set(test_df.columns))
    extra_test = sorted(set(test_df.columns) - expected_test)

    if missing_train or missing_test:
        _add_check(
            results,
            "schema_missing_columns",
            "fail",
            "high",
            "Missing expected columns.",
            count=len(missing_train) + len(missing_test),
            details={"missing_train": missing_train, "missing_test": missing_test},
        )
    else:
        _add_check(
            results,
            "schema_missing_columns",
            "pass",
            None,
            "All expected columns are present.",
        )

    if extra_train or extra_test:
        _add_check(
            results,
            "schema_extra_columns",
            "warn",
            "low",
            "Unexpected extra columns detected.",
            count=len(extra_train) + len(extra_test),
            details={"extra_train": extra_train, "extra_test": extra_test},
        )
    else:
        _add_check(
            results,
            "schema_extra_columns",
            "pass",
            None,
            "No unexpected columns detected.",
        )

    # Missing/NaN/Inf checks
    for label, df in [("train", train_df), ("test", test_df)]:
        missing_counts = df.isna().sum()
        missing_total = int(missing_counts.sum())
        if missing_total > 0:
            _add_check(
                results,
                f"{label}_missing_values",
                "warn",
                "medium",
                f"{label} has missing values.",
                count=missing_total,
                details={"missing_by_column": missing_counts[missing_counts > 0].to_dict()},
            )
        else:
            _add_check(
                results,
                f"{label}_missing_values",
                "pass",
                None,
                f"{label} has no missing values.",
            )

        numeric_cols = _numeric_columns(df)
        if numeric_cols:
            values = df[numeric_cols].to_numpy()
            inf_count = int(np.isinf(values).sum())
            if inf_count > 0:
                _add_check(
                    results,
                    f"{label}_inf_values",
                    "fail",
                    "high",
                    f"{label} has infinite values.",
                    count=inf_count,
                )
            else:
                _add_check(
                    results,
                    f"{label}_inf_values",
                    "pass",
                    None,
                    f"{label} has no infinite values.",
                )

    # Duplicate checks
    for label, df in [("train", train_df), ("test", test_df)]:
        dup_rows = int(df.duplicated().sum())
        if dup_rows > 0:
            _add_check(
                results,
                f"{label}_duplicate_rows",
                "warn",
                "medium",
                f"{label} has duplicate rows.",
                count=dup_rows,
            )
        else:
            _add_check(
                results,
                f"{label}_duplicate_rows",
                "pass",
                None,
                f"{label} has no duplicate rows.",
            )

        if schema.id_col in df.columns:
            dup_ids = int(df[schema.id_col].duplicated().sum())
            if dup_ids > 0:
                _add_check(
                    results,
                    f"{label}_duplicate_ids",
                    "fail",
                    "high",
                    f"{label} has duplicate ids.",
                    count=dup_ids,
                )
            else:
                _add_check(
                    results,
                    f"{label}_duplicate_ids",
                    "pass",
                    None,
                    f"{label} has no duplicate ids.",
                )

    # Type/domain checks
    if "step" in train_df.columns:
        step_series = train_df["step"]
        step_negative = int((step_series < 0).sum())
        if step_negative > 0:
            _add_check(
                results,
                "step_negative",
                "fail",
                "high",
                "Negative step values detected.",
                count=step_negative,
                examples=_collect_examples(
                    train_df, step_series < 0, [schema.id_col, "step", "type"]
                ),
            )
        else:
            _add_check(results, "step_negative", "pass", None, "No negative step values.")

        step_max = int(step_series.max())
        if step_max > schema.thresholds.step_max_reasonable:
            _add_check(
                results,
                "step_max_exceeds_threshold",
                "warn",
                "low",
                "Step values exceed configured maximum.",
                count=int((step_series > schema.thresholds.step_max_reasonable).sum()),
                details={
                    "observed_max": step_max,
                    "threshold": schema.thresholds.step_max_reasonable,
                },
            )
        else:
            _add_check(
                results,
                "step_max_exceeds_threshold",
                "pass",
                None,
                "Step values within expected range.",
            )

    if "type" in train_df.columns:
        observed_types = sorted(train_df["type"].dropna().unique().tolist())
        allowed = set(schema.allowed_type_values)
        invalid_mask = ~train_df["type"].isin(allowed)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            _add_check(
                results,
                "type_invalid_categories",
                "fail",
                "high",
                "Invalid transaction types detected.",
                count=invalid_count,
                examples=_collect_examples(
                    train_df, invalid_mask, [schema.id_col, "type"]
                ),
                details={"allowed": sorted(allowed), "observed": observed_types},
            )
        else:
            _add_check(
                results,
                "type_invalid_categories",
                "pass",
                None,
                "All transaction types are within allowed set.",
                details={"allowed": sorted(allowed), "observed": observed_types},
            )

        expected = set(schema.expected_type_values)
        extra_vs_expected = sorted(set(observed_types) - expected)
        missing_vs_expected = sorted(expected - set(observed_types))
        if extra_vs_expected or missing_vs_expected:
            _add_check(
                results,
                "type_expected_mismatch",
                "warn",
                "low",
                "Observed transaction types differ from canonical expected set.",
                count=len(extra_vs_expected) + len(missing_vs_expected),
                details={
                    "expected": sorted(expected),
                    "observed": observed_types,
                    "extra": extra_vs_expected,
                    "missing": missing_vs_expected,
                },
            )
        else:
            _add_check(
                results,
                "type_expected_mismatch",
                "pass",
                None,
                "Observed transaction types match expected set.",
            )

    if "amount" in train_df.columns:
        amount = train_df["amount"].astype(float)
        negative_amount = int((amount < 0).sum())
        if negative_amount > 0:
            _add_check(
                results,
                "amount_negative",
                "fail",
                "high",
                "Negative amounts detected.",
                count=negative_amount,
                examples=_collect_examples(
                    train_df, amount < 0, [schema.id_col, "amount", "type"]
                ),
            )
        else:
            _add_check(results, "amount_negative", "pass", None, "No negative amounts.")

        outlier_threshold = float(
            amount.quantile(schema.thresholds.amount_outlier_quantile)
        )
        outlier_count = int((amount > outlier_threshold).sum())
        _add_check(
            results,
            "amount_outliers",
            "warn",
            "medium",
            "Extreme amount outliers detected (quantile-based).",
            count=outlier_count,
            details={
                "quantile": schema.thresholds.amount_outlier_quantile,
                "threshold": outlier_threshold,
            },
        )

        amount_sample = _sample_df(
            train_df, schema.thresholds.check_sample_size, schema.thresholds.random_state
        )["amount"]
        spike_values = _detect_amount_spikes(
            amount_sample,
            schema.thresholds.amount_spike_top_n,
            schema.thresholds.amount_spike_min_count,
        )
        spike_count = int(train_df["amount"].isin(spike_values).sum()) if spike_values else 0
        _add_check(
            results,
            "amount_spikes",
            "warn",
            "low",
            "Repeated exact amount spikes detected.",
            count=spike_count,
            details={"spike_values": spike_values},
        )

    balance_cols = [
        col
        for col in [
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]
        if col in train_df.columns
    ]
    if balance_cols:
        for col in balance_cols:
            negative_count = int((train_df[col] < 0).sum())
            if negative_count > 0:
                _add_check(
                    results,
                    f"{col}_negative",
                    "fail",
                    "high",
                    f"Negative balances detected in {col}.",
                    count=negative_count,
                    examples=_collect_examples(
                        train_df, train_df[col] < 0, [schema.id_col, col, "type"]
                    ),
                )
            else:
                _add_check(
                    results,
                    f"{col}_negative",
                    "pass",
                    None,
                    f"No negative balances in {col}.",
                )

    for label, df in [("train", train_df), ("test", test_df)]:
        if "nameOrig" in df.columns and "nameDest" in df.columns:
            for col in ["nameOrig", "nameDest"]:
                series = df[col].astype(str)
                empty_count = int((series.str.len() == 0).sum())
                regex = schema.thresholds.name_regex
                pattern_mask = ~series.str.match(regex)
                pattern_count = int(pattern_mask.sum())
                prefixes = series.str[0].value_counts().to_dict()
                if empty_count > 0 or pattern_count > 0:
                    _add_check(
                        results,
                        f"{label}_{col}_pattern",
                        "warn",
                        "medium",
                        f"Unusual {col} formats detected.",
                        count=empty_count + pattern_count,
                        examples=_collect_examples(
                            df, pattern_mask | (series.str.len() == 0), [schema.id_col, col]
                        ),
                        details={"prefixes": prefixes, "regex": regex},
                    )
                else:
                    _add_check(
                        results,
                        f"{label}_{col}_pattern",
                        "pass",
                        None,
                        f"{col} formats look consistent.",
                        details={"prefixes": prefixes, "regex": regex},
                    )

    # Cross-field consistency checks (sampled for speed)
    cross_sample = _sample_df(
        train_df, schema.thresholds.check_sample_size, schema.thresholds.random_state
    )
    balance_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "amount"]
    if all(col in cross_sample.columns for col in balance_cols) and not any(col in schema.exclude_fields for col in balance_cols):
        orig_delta = cross_sample["oldbalanceOrg"] - cross_sample["newbalanceOrig"]
        dest_delta = cross_sample["newbalanceDest"] - cross_sample["oldbalanceDest"]
        expected_orig, expected_dest = _expected_balance_deltas(cross_sample)
        orig_error = (orig_delta - expected_orig).abs()
        dest_error = (dest_delta - expected_dest).abs()

        orig_large = int((orig_error > schema.thresholds.balance_error_threshold).sum())
        dest_large = int((dest_error > schema.thresholds.balance_error_threshold).sum())
        orig_high = int((orig_error > schema.thresholds.balance_error_high_threshold).sum())
        dest_high = int((dest_error > schema.thresholds.balance_error_high_threshold).sum())

        _add_check(
            results,
            "balance_error_orig",
            "warn",
            "medium",
            "Origin balance deltas deviate from expected amounts (sampled).",
            count=orig_large,
            details={
                "sample_size": len(cross_sample),
                "threshold": schema.thresholds.balance_error_threshold,
                "high_threshold": schema.thresholds.balance_error_high_threshold,
                "high_error_count": orig_high,
            },
        )
        _add_check(
            results,
            "balance_error_dest",
            "warn",
            "medium",
            "Destination balance deltas deviate from expected amounts (sampled).",
            count=dest_large,
            details={
                "sample_size": len(cross_sample),
                "threshold": schema.thresholds.balance_error_threshold,
                "high_threshold": schema.thresholds.balance_error_high_threshold,
                "high_error_count": dest_high,
            },
        )

    # Train vs test drift checks
    numeric_cols = [c for c in _numeric_columns(train_df) if c in test_df.columns]
    drift_numeric = []
    for col in numeric_cols:
        train_sample = _sample_df(
            train_df[[col]], schema.thresholds.drift_sample_size, schema.thresholds.random_state
        )[col].to_numpy()
        test_sample = _sample_df(
            test_df[[col]], schema.thresholds.drift_sample_size, schema.thresholds.random_state
        )[col].to_numpy()
        ks_stat, p_value = _ks_statistic(train_sample, test_sample)
        drift_numeric.append(
            {
                "column": col,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "train_mean": float(np.nanmean(train_sample)),
                "test_mean": float(np.nanmean(test_sample)),
            }
        )
    categorical_cols = [
        c for c in train_df.columns if train_df[c].dtype == "object" and c in test_df.columns
    ]
    drift_categorical = []
    for col in categorical_cols:
        train_freq = train_df[col].value_counts(normalize=True)
        test_freq = test_df[col].value_counts(normalize=True)
        psi_value = _psi(train_freq, test_freq)
        drift_categorical.append(
            {
                "column": col,
                "psi": psi_value,
                "train_top": train_freq.head(5).to_dict(),
                "test_top": test_freq.head(5).to_dict(),
            }
        )

    drift_numeric_sorted = sorted(drift_numeric, key=lambda x: x["ks_stat"], reverse=True)
    drift_categorical_sorted = sorted(
        drift_categorical, key=lambda x: x["psi"], reverse=True
    )
    results["drift"] = {
        "numeric": drift_numeric_sorted,
        "categorical": drift_categorical_sorted,
        "top_numeric": drift_numeric_sorted[:5],
        "top_categorical": drift_categorical_sorted[:5],
    }

    # Class distribution report (train only)
    if schema.target_col in train_df.columns:
        dist = train_df[schema.target_col].value_counts().to_dict()
        total = len(train_df)
        dist_pct = {k: v / total for k, v in dist.items()}
        results["class_distribution"] = {
            "counts": dist,
            "percentages": dist_pct,
        }

    return results
