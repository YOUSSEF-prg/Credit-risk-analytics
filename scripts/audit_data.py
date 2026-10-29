"""Phase 1 entry point: validate the raw dataset, write the data-quality report,
dataset metadata and the reproducible train/validation/test split.

Usage (from the project root):
    python scripts/audit_data.py
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime

import pandas as pd
import sklearn

from credit_risk.config import load_config, resolve_path
from credit_risk.data import (
    FEATURES,
    TARGET,
    file_sha256,
    load_raw,
    make_splits,
    standardize_columns,
)
from credit_risk.validation import (
    data_quality_report,
    validate_raw_schema,
    validate_standardized_schema,
)


def main() -> None:
    cfg = load_config("data")
    ds, split_cfg, out = cfg["dataset"], cfg["split"], cfg["outputs"]
    raw_path = resolve_path(ds["raw_path"])

    raw = load_raw(raw_path, expected_sha256=ds["expected_sha256"])
    validate_raw_schema(raw, expected_rows=ds["expected_rows"])
    df = standardize_columns(raw)
    validate_standardized_schema(df, expected_rows=ds["expected_rows"])

    report = data_quality_report(df)
    report_path = resolve_path(out["quality_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    splits = make_splits(
        df,
        train_fraction=split_cfg["train_fraction"],
        validation_fraction=split_cfg["validation_fraction"],
        test_fraction=split_cfg["test_fraction"],
        random_seed=split_cfg["random_seed"],
        stratify=split_cfg["stratify"],
    )
    splits_path = resolve_path(split_cfg["output_path"])
    splits.to_json(splits_path)

    id_to_target = df.set_index("ID")[TARGET]
    split_rates = {
        name: float(id_to_target.loc[ids].mean())
        for name, ids in [
            ("train", splits.train_ids),
            ("validation", splits.validation_ids),
            ("test", splits.test_ids),
        ]
    }
    stat = raw_path.stat()
    metadata = {
        "dataset": {k: ds[k] for k in ["name", "uci_id", "source_url", "doi", "license"]},
        "raw_file": {
            "path": ds["raw_path"],
            "sha256": file_sha256(raw_path),
            "size_bytes": stat.st_size,
            "local_modified_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        },
        "features": FEATURES,
        "excluded_columns": ["ID"],
        "target": TARGET,
        "split": {
            "random_seed": splits.random_seed,
            "fractions": splits.fractions,
            "sizes": {
                "train": len(splits.train_ids),
                "validation": len(splits.validation_ids),
                "test": len(splits.test_ids),
            },
            "default_rate": split_rates,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    metadata_path = resolve_path(out["metadata"])
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Rows: {report['n_rows']}, default rate: {report['target']['default_rate']:.4f}")
    print(f"Split sizes: {metadata['split']['sizes']}")
    print(f"Split default rates: { {k: round(v, 4) for k, v in split_rates.items()} }")
    print(f"Wrote {report_path}, {metadata_path}, {splits_path}")


if __name__ == "__main__":
    main()
