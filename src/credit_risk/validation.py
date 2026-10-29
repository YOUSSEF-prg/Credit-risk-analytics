"""Schema validation and data-quality checks.

Validation raises :class:`SchemaError` for problems that make the data unusable
(wrong columns, non-binary target, duplicate IDs). Data-quality findings that are
plausible but undocumented (for example ``EDUCATION = 0``) are reported, not fixed:
treatment decisions belong in preprocessing and are documented there.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from credit_risk.data import (
    BILL_AMOUNTS,
    ID_COLUMN,
    PAY_AMOUNTS,
    PAY_STATUS,
    RAW_COLUMNS,
    RAW_TARGET,
    TARGET,
)

# Codes listed in the UCI variable description.
DOCUMENTED_CODES: dict[str, set[int]] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
# UCI documents -1 (paid duly) and 1..9 (months of delay). -2 and 0 occur but are undocumented.
DOCUMENTED_PAY_STATUS: set[int] = {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9}


class SchemaError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def validate_raw_schema(raw: pd.DataFrame, expected_rows: int | None = None) -> None:
    """Check the raw UCI table: exact columns, integer types, unique IDs, binary target."""
    columns = list(raw.columns)
    if columns != RAW_COLUMNS:
        missing = sorted(set(RAW_COLUMNS) - set(columns))
        extra = sorted(set(columns) - set(RAW_COLUMNS))
        raise SchemaError(
            f"Unexpected columns. Missing: {missing}; unexpected: {extra}; "
            "or the column order differs from the UCI file."
        )
    _validate_common(raw, target=RAW_TARGET, expected_rows=expected_rows)


def validate_standardized_schema(df: pd.DataFrame, expected_rows: int | None = None) -> None:
    """Check a table produced by :func:`credit_risk.data.standardize_columns`."""
    _validate_common(df, target=TARGET, expected_rows=expected_rows)


def _validate_common(df: pd.DataFrame, target: str, expected_rows: int | None) -> None:
    if expected_rows is not None and len(df) != expected_rows:
        raise SchemaError(f"Expected {expected_rows} rows, found {len(df)}")
    non_integer = [c for c in df.columns if not pd.api.types.is_integer_dtype(df[c])]
    if non_integer:
        raise SchemaError(f"Columns with non-integer values: {non_integer}")
    if df.isna().any().any():
        raise SchemaError("Dataset contains missing values")
    if not df[ID_COLUMN].is_unique:
        raise SchemaError("Customer ID column contains duplicates")
    target_values = set(df[target].unique())
    if not target_values <= {0, 1} or len(target_values) < 2:
        raise SchemaError(f"Target must contain both 0 and 1 only, found {sorted(target_values)}")


def data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise data-quality findings for a standardised DataFrame.

    Returns only JSON-serialisable Python types.
    """
    features = [c for c in df.columns if c not in {ID_COLUMN, TARGET}]
    n_rows = len(df)

    undocumented: dict[str, dict[str, int]] = {}
    for column, allowed in DOCUMENTED_CODES.items():
        counts = df.loc[~df[column].isin(allowed), column].value_counts().sort_index()
        undocumented[column] = {str(int(k)): int(v) for k, v in counts.items()}
    for column in PAY_STATUS:
        counts = df.loc[~df[column].isin(DOCUMENTED_PAY_STATUS), column].value_counts()
        undocumented[column] = {str(int(k)): int(v) for k, v in counts.sort_index().items()}

    inactive = (df[BILL_AMOUNTS] == 0).all(axis=1) & (df[PAY_AMOUNTS] == 0).all(axis=1)
    duplicated_features = df.duplicated(subset=features, keep=False)
    conflicting = (
        df[duplicated_features].groupby(features, sort=False)[TARGET].nunique().gt(1).sum()
    )

    return {
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "n_features": len(features),
        "missing_values_total": int(df.isna().sum().sum()),
        "id_unique": bool(df[ID_COLUMN].is_unique),
        "target": {
            "name": TARGET,
            "default_rate": float(df[TARGET].mean()),
            "counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        },
        "undocumented_codes": undocumented,
        "rows_with_undocumented_demographic_code": int(
            sum(~df[c].isin(a) for c, a in DOCUMENTED_CODES.items()).gt(0).sum()
        ),
        "negative_bill_amounts": {c: int((df[c] < 0).sum()) for c in BILL_AMOUNTS},
        "negative_payment_amounts": {c: int((df[c] < 0).sum()) for c in PAY_AMOUNTS},
        "inactive_accounts_all_zero_bills_and_payments": int(inactive.sum()),
        "duplicate_feature_rows_excluding_id": int(df.duplicated(subset=features).sum()),
        "duplicate_feature_groups_with_conflicting_target": int(conflicting),
        "ranges": {
            c: {"min": int(df[c].min()), "max": int(df[c].max())}
            for c in ["LIMIT_BAL", "AGE", *BILL_AMOUNTS, *PAY_AMOUNTS]
        },
        "bill_exceeds_limit_rows": int((df[BILL_AMOUNTS].max(axis=1) > df["LIMIT_BAL"]).sum()),
    }


# ----------------------------------------------------------------------------------------
# Validation of user-entered feature vectors (dashboard Credit Risk Explorer)
# ----------------------------------------------------------------------------------------

INPUT_RULES: dict[str, tuple[float, float]] = {
    "LIMIT_BAL": (10_000, 1_000_000),
    "AGE": (21, 79),
    **{f"PAY_{i}": (-2, 8) for i in range(1, 7)},
    **{f"BILL_AMT{i}": (-400_000, 1_700_000) for i in range(1, 7)},
    **{f"PAY_AMT{i}": (0, 1_700_000) for i in range(1, 7)},
}
INPUT_CODES: dict[str, set[int]] = {"SEX": {1, 2}, "EDUCATION": {1, 2, 3, 4}, "MARRIAGE": {1, 2, 3}}


def validate_feature_input(values: dict[str, Any]) -> tuple[pd.DataFrame | None, list[str]]:
    """Check a manually entered client against the ranges observed in the dataset.

    Ranges are the observed min/max of the UCI data, rounded outward. Inputs outside them
    are rejected rather than extrapolated: the model has never seen such values. Returns
    a one-row DataFrame in model column order, or ``None`` plus a list of error messages.
    """
    from credit_risk.data import FEATURES

    errors: list[str] = []
    row: dict[str, int] = {}
    for column in FEATURES:
        raw = values.get(column)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or raw == "":
            errors.append(f"{column}: value is missing")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{column}: {raw!r} is not a number")
            continue
        if not number.is_integer():
            errors.append(f"{column}: must be a whole number")
            continue
        number = int(number)
        if column in INPUT_CODES and number not in INPUT_CODES[column]:
            errors.append(f"{column}: {number} is not a valid code {sorted(INPUT_CODES[column])}")
        elif column in INPUT_RULES:
            low, high = INPUT_RULES[column]
            if not low <= number <= high:
                errors.append(
                    f"{column}: {number:,} is outside the observed range [{low:,}, {high:,}]"
                )
        row[column] = number
    if errors:
        return None, errors
    return pd.DataFrame([row], columns=FEATURES), []
