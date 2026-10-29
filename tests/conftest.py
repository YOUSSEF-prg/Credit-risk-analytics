"""Shared fixtures. Tests use a small synthetic table with the real schema so they
run without the raw dataset; tests on the real file are skipped when it is absent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import load_config, resolve_path
from credit_risk.data import (
    BILL_AMOUNTS,
    PAY_AMOUNTS,
    RAW_COLUMNS,
    RAW_PAY_STATUS,
    RAW_TARGET,
)


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    data = {
        "ID": np.arange(1, n + 1),
        "LIMIT_BAL": rng.integers(1, 50, n) * 10_000,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(0, 7, n),
        "MARRIAGE": rng.integers(0, 4, n),
        "AGE": rng.integers(21, 75, n),
        **{c: rng.integers(-2, 9, n) for c in RAW_PAY_STATUS},
        **{c: rng.integers(-1_000, 200_000, n) for c in BILL_AMOUNTS},
        **{c: rng.integers(0, 20_000, n) for c in PAY_AMOUNTS},
        RAW_TARGET: (rng.random(n) < 0.22).astype(int),
    }
    return pd.DataFrame(data)[RAW_COLUMNS]


@pytest.fixture(scope="session")
def real_raw_path():
    cfg = load_config("data")["dataset"]
    path = resolve_path(cfg["raw_path"])
    if not path.exists():
        pytest.skip("Raw UCI dataset not available")
    return path
