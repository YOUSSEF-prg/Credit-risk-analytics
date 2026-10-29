"""Loading the UCI dataset, standardising column names and creating data splits.

The raw file is never modified. All transformations return new DataFrames.

Column conventions after :func:`standardize_columns`:

* ``PAY_0`` is renamed to ``PAY_1`` so that index 1 refers to September 2005
  for all three monthly blocks (``PAY_1``, ``BILL_AMT1``, ``PAY_AMT1``).
* The target ``default payment next month`` is renamed to ``default_next_month``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ID_COLUMN = "ID"
RAW_TARGET = "default payment next month"
TARGET = "default_next_month"

RAW_PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMOUNTS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNTS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]

RAW_COLUMNS = [
    ID_COLUMN,
    "LIMIT_BAL",
    *DEMOGRAPHIC,
    *RAW_PAY_STATUS,
    *BILL_AMOUNTS,
    *PAY_AMOUNTS,
    RAW_TARGET,
]
FEATURES = ["LIMIT_BAL", *DEMOGRAPHIC, *PAY_STATUS, *BILL_AMOUNTS, *PAY_AMOUNTS]

_RENAME = {"PAY_0": "PAY_1", RAW_TARGET: TARGET}


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    """Load the original UCI Excel file exactly as published.

    The file has two header rows (``X1 ... X23, Y`` and the descriptive names);
    the second row holds the column names used here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it from "
            "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients "
            "and place it in data/raw/."
        )
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            warnings.warn(
                f"Raw file checksum {actual} differs from the expected {expected_sha256}. "
                "Results may not be comparable with recorded experiments.",
                stacklevel=2,
            )
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, header=1, engine="calamine")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent column names (see module docstring)."""
    return raw.rename(columns=_RENAME).copy()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the predictive features (customer ID excluded) and the binary target."""
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()


@dataclass(frozen=True)
class DataSplits:
    """Customer IDs assigned to each partition."""

    train_ids: list[int]
    validation_ids: list[int]
    test_ids: list[int]
    random_seed: int
    fractions: dict[str, float]

    def to_json(self, path: Path) -> None:
        payload = {
            "random_seed": self.random_seed,
            "fractions": self.fractions,
            "sizes": {
                "train": len(self.train_ids),
                "validation": len(self.validation_ids),
                "test": len(self.test_ids),
            },
            "train_ids": self.train_ids,
            "validation_ids": self.validation_ids,
            "test_ids": self.test_ids,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> DataSplits:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            train_ids=payload["train_ids"],
            validation_ids=payload["validation_ids"],
            test_ids=payload["test_ids"],
            random_seed=payload["random_seed"],
            fractions=payload["fractions"],
        )


def make_splits(
    df: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True,
) -> DataSplits:
    """Create a stratified random train / validation / test partition by customer ID.

    The test set is split off first, then validation is split from the remainder,
    so the three sets are disjoint and jointly cover every row.
    """
    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {total:.4f}")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive")
    if not df[ID_COLUMN].is_unique:
        raise ValueError("Customer IDs must be unique to define splits")

    ids = df[ID_COLUMN].to_numpy()
    target = df[TARGET].to_numpy()
    rest_ids, test_ids, rest_y, _ = train_test_split(
        ids,
        target,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=target if stratify else None,
    )
    val_share_of_rest = validation_fraction / (train_fraction + validation_fraction)
    train_ids, val_ids = train_test_split(
        rest_ids,
        test_size=val_share_of_rest,
        random_state=random_seed,
        stratify=rest_y if stratify else None,
    )
    return DataSplits(
        train_ids=sorted(int(i) for i in train_ids),
        validation_ids=sorted(int(i) for i in val_ids),
        test_ids=sorted(int(i) for i in test_ids),
        random_seed=random_seed,
        fractions={
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
    )


def apply_splits(
    df: pd.DataFrame, splits: DataSplits
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the train, validation and test subsets defined by ``splits``."""
    indexed = df.set_index(ID_COLUMN, drop=False)
    return (
        indexed.loc[splits.train_ids].reset_index(drop=True),
        indexed.loc[splits.validation_ids].reset_index(drop=True),
        indexed.loc[splits.test_ids].reset_index(drop=True),
    )
