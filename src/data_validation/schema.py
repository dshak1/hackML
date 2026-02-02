from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


DEFAULT_ALLOWED_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
DEFAULT_TARGET_VALUES = [0, 1, 2, 3]


@dataclass
class RuleThresholds:
    step_min_reasonable: int = 0
    step_max_reasonable: int = 1000
    amount_outlier_quantile: float = 0.999
    amount_spike_top_n: int = 10
    amount_spike_min_count: int = 100
    balance_error_threshold: float = 1.0
    balance_error_high_threshold: float = 100.0
    name_regex: str = r"^[A-Z]\d+$"
    name_prefixes: List[str] = field(default_factory=lambda: ["C", "M"])
    drift_sample_size: int = 200000
    check_sample_size: int = 200000
    random_state: int = 42


@dataclass
class ValidationSchema:
    train_columns: List[str]
    test_columns: List[str]
    dtypes: Dict[str, str]
    allowed_type_values: List[str] = field(default_factory=lambda: DEFAULT_ALLOWED_TYPES.copy())
    expected_type_values: List[str] = field(default_factory=lambda: DEFAULT_ALLOWED_TYPES.copy())
    target_values: List[int] = field(default_factory=lambda: DEFAULT_TARGET_VALUES.copy())
    id_col: str = "id"
    target_col: str = "urgency_level"
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)
    exclude_fields: List[str] = field(default_factory=list)
    mode: str = "warn"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def infer_schema(train_df, test_df, mode: str = "warn") -> ValidationSchema:
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)

    allowed_types = set()
    if "type" in train_df.columns:
        allowed_types.update(train_df["type"].dropna().unique().tolist())
    if "type" in test_df.columns:
        allowed_types.update(test_df["type"].dropna().unique().tolist())
    if not allowed_types:
        allowed_types = set(DEFAULT_ALLOWED_TYPES)

    step_min = None
    step_max = None
    for df in (train_df, test_df):
        if "step" in df.columns:
            col_min = df["step"].min()
            col_max = df["step"].max()
            step_min = col_min if step_min is None else min(step_min, col_min)
            step_max = col_max if step_max is None else max(step_max, col_max)

    thresholds = RuleThresholds(
        step_min_reasonable=int(step_min) if step_min is not None else 0,
        step_max_reasonable=int(step_max) if step_max is not None else 1000,
    )

    dtypes = {col: str(train_df[col].dtype) for col in train_df.columns}

    # Fields excluded from model training (balance fields)
    exclude_fields = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

    return ValidationSchema(
        train_columns=train_cols,
        test_columns=test_cols,
        dtypes=dtypes,
        allowed_type_values=sorted(allowed_types),
        expected_type_values=DEFAULT_ALLOWED_TYPES.copy(),
        thresholds=thresholds,
        exclude_fields=exclude_fields,
        mode=mode,
    )


def build_schema(
    train_df,
    test_df,
    mode: str = "warn",
) -> ValidationSchema:
    return infer_schema(train_df, test_df, mode=mode)
