from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import (
    FEATURES,
    ID_COLUMN,
    TARGET,
    DataSplits,
    apply_splits,
    load_raw,
    make_splits,
    split_features_target,
    standardize_columns,
)
from credit_risk.validation import (
    SchemaError,
    data_quality_report,
    validate_raw_schema,
    validate_standardized_schema,
)

# --- schema and target -------------------------------------------------------------


def test_valid_synthetic_schema_passes(synthetic_raw):
    validate_raw_schema(synthetic_raw)
    validate_standardized_schema(standardize_columns(synthetic_raw))


def test_missing_column_raises(synthetic_raw):
    with pytest.raises(SchemaError, match="Missing"):
        validate_raw_schema(synthetic_raw.drop(columns=["AGE"]))


def test_non_binary_target_raises(synthetic_raw):
    bad = synthetic_raw.copy()
    bad.loc[0, "default payment next month"] = 2
    with pytest.raises(SchemaError, match="Target"):
        validate_raw_schema(bad)


def test_duplicate_ids_raise(synthetic_raw):
    bad = synthetic_raw.copy()
    bad.loc[1, "ID"] = bad.loc[0, "ID"]
    with pytest.raises(SchemaError, match="duplicates"):
        validate_raw_schema(bad)


def test_missing_values_raise(synthetic_raw):
    bad = synthetic_raw.astype({"AGE": float})
    bad.loc[0, "AGE"] = np.nan
    with pytest.raises(SchemaError):
        validate_raw_schema(bad)


def test_standardize_renames_and_does_not_mutate(synthetic_raw):
    original = synthetic_raw.copy()
    df = standardize_columns(synthetic_raw)
    assert "PAY_1" in df.columns and "PAY_0" not in df.columns
    assert TARGET in df.columns
    pd.testing.assert_frame_equal(synthetic_raw, original)


def test_features_exclude_id_and_target(synthetic_raw):
    X, y = split_features_target(standardize_columns(synthetic_raw))
    assert ID_COLUMN not in X.columns
    assert TARGET not in X.columns
    assert list(X.columns) == FEATURES
    assert set(y.unique()) <= {0, 1}


# --- splits -----------------------------------------------------------------------


def test_splits_are_disjoint_and_complete(synthetic_raw):
    df = standardize_columns(synthetic_raw)
    s = make_splits(df, random_seed=42)
    train, val, test = set(s.train_ids), set(s.validation_ids), set(s.test_ids)
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(df[ID_COLUMN])
    assert len(s.train_ids) == pytest.approx(0.6 * len(df), abs=1)
    assert len(s.test_ids) == pytest.approx(0.2 * len(df), abs=1)


def test_splits_are_reproducible_and_seed_dependent(synthetic_raw):
    df = standardize_columns(synthetic_raw)
    assert make_splits(df, random_seed=42) == make_splits(df, random_seed=42)
    assert make_splits(df, random_seed=42).test_ids != make_splits(df, random_seed=7).test_ids


def test_stratification_preserves_default_rate(synthetic_raw):
    df = standardize_columns(synthetic_raw)
    s = make_splits(df, random_seed=42)
    _, _, test = apply_splits(df, s)
    assert test[TARGET].mean() == pytest.approx(df[TARGET].mean(), abs=0.02)


def test_invalid_fractions_raise(synthetic_raw):
    df = standardize_columns(synthetic_raw)
    with pytest.raises(ValueError, match="sum to 1"):
        make_splits(df, 0.7, 0.2, 0.2)


def test_splits_json_round_trip(synthetic_raw, tmp_path):
    s = make_splits(standardize_columns(synthetic_raw))
    path = tmp_path / "splits.json"
    s.to_json(path)
    assert DataSplits.from_json(path) == s


# --- data quality -----------------------------------------------------------------


def test_quality_report_flags_undocumented_codes(synthetic_raw):
    report = data_quality_report(standardize_columns(synthetic_raw))
    assert "0" in report["undocumented_codes"]["EDUCATION"]
    assert "-2" in report["undocumented_codes"]["PAY_1"]
    assert report["missing_values_total"] == 0


# --- real dataset (skipped if unavailable) ----------------------------------------


def test_real_dataset_schema_and_known_facts(real_raw_path):
    raw = load_raw(real_raw_path)
    validate_raw_schema(raw, expected_rows=30_000)
    df = standardize_columns(raw)
    assert df[TARGET].sum() == 6_636  # 22.12% default rate
    report = data_quality_report(df)
    assert report["undocumented_codes"]["EDUCATION"] == {"0": 14, "5": 280, "6": 51}
    assert report["undocumented_codes"]["MARRIAGE"] == {"0": 54}
