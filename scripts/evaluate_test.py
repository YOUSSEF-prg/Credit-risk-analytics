"""One-time final evaluation on the held-out test set.

Protocol:

1. The model choice (``artifacts/metrics/selection.json``) and both decision thresholds
   (chosen on validation) are **frozen** in ``artifacts/metrics/frozen_decision.json``,
   together with SHA-256 hashes of the model files, *before* any test row is read.
2. The selected model and the reference models are evaluated on the test set with
   those frozen thresholds. Nothing is re-tuned or re-selected afterwards.
3. The script refuses to run again if ``test_evaluation.json`` exists, unless ``--force``
   is given. A forced re-run is recorded in the output, so repeated test access is visible.

Usage:
    python scripts/evaluate_test.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from credit_risk.config import load_config, resolve_path
from credit_risk.data import (
    ID_COLUMN,
    TARGET,
    DataSplits,
    apply_splits,
    file_sha256,
    load_raw,
    split_features_target,
    standardize_columns,
)
from credit_risk.evaluation import (
    bootstrap_confidence_intervals,
    calibration_table,
    curve_points,
    paired_bootstrap_difference,
    probability_metrics,
    risk_band_table,
    threshold_metrics,
)
from credit_risk.features import CreditFeatureEngineer
from credit_risk.models import predict_pd
from credit_risk.validation import validate_standardized_schema

REFERENCE_MODELS = ["logreg", "random_forest", "dummy_prior"]
PROFILE_COLUMNS = [
    "LIMIT_BAL", "AGE", "PAY_1", "months_delayed", "max_delay",
    "utilization_mean", "payment_to_bill_ratio",
]  # fmt: skip


def freeze_decision(metrics_dir, models_dir) -> dict:
    selection = json.loads((metrics_dir / "selection.json").read_text(encoding="utf-8"))
    chosen = selection["selected_model"]
    record = json.loads((metrics_dir / f"{chosen}.json").read_text(encoding="utf-8"))
    models = [chosen, *[m for m in REFERENCE_MODELS if m != chosen]]
    frozen = {
        "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected_model": chosen,
        "selection_rule": selection["rule"],
        "thresholds": {
            "f2_unconstrained": record["threshold_selection"]["threshold"],
            "f2_capacity_30pct": record["threshold_selection_capacity"]["threshold"],
            "reference_0_5": 0.5,
        },
        "thresholds_chosen_on": "validation",
        "reference_models": models[1:],
        "model_file_sha256": {m: file_sha256(models_dir / f"{m}.joblib") for m in models},
    }
    (metrics_dir / "frozen_decision.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    return frozen


def error_profiles(X: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float) -> pd.DataFrame:
    engineered = CreditFeatureEngineer().fit_transform(X)
    flagged = p >= threshold
    outcome = np.select(
        [flagged & (y == 1), flagged & (y == 0), ~flagged & (y == 1)], ["TP", "FP", "FN"], "TN"
    )
    frame = engineered[PROFILE_COLUMNS].assign(outcome=outcome, pd=p)
    summary = frame.groupby("outcome")[[*PROFILE_COLUMNS, "pd"]].median()
    summary.insert(0, "clients", frame.groupby("outcome").size())
    return summary.reindex(["TP", "FP", "FN", "TN"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = parser.parse_args()

    data_cfg, model_cfg = load_config("data"), load_config("models")
    metrics_dir = resolve_path(model_cfg["outputs"]["metrics_dir"])
    models_dir = resolve_path(model_cfg["outputs"]["models_dir"])
    preds_dir = resolve_path(model_cfg["outputs"]["predictions_dir"])
    out_path = metrics_dir / "test_evaluation.json"
    previous_runs = 0
    if out_path.exists():
        if not args.force:
            raise SystemExit(
                "test_evaluation.json already exists: the test set has been evaluated. "
                "Use --force only if you accept a second, recorded test access."
            )
        previous_runs = json.loads(out_path.read_text(encoding="utf-8")).get("test_access_count", 1)

    frozen = freeze_decision(metrics_dir, models_dir)  # written BEFORE test data is read
    print(f"Frozen: {frozen['selected_model']} with thresholds {frozen['thresholds']}")

    ds = data_cfg["dataset"]
    df = standardize_columns(load_raw(resolve_path(ds["raw_path"]), ds["expected_sha256"]))
    validate_standardized_schema(df, expected_rows=ds["expected_rows"])
    splits = DataSplits.from_json(resolve_path(data_cfg["split"]["output_path"]))
    _, _, test = apply_splits(df, splits)
    X_test, y_test = split_features_target(test)
    y = y_test.to_numpy()

    seed, boot = model_cfg["experiment"]["random_seed"], model_cfg["bootstrap"]
    chosen = frozen["selected_model"]
    results, predictions = {}, {}
    for name in [chosen, *frozen["reference_models"]]:
        model = joblib.load(models_dir / f"{name}.joblib")
        p = predict_pd(model, X_test).to_numpy()
        predictions[name] = p
        pd.DataFrame({ID_COLUMN: test[ID_COLUMN].values, TARGET: y, "pd": p}).to_csv(
            preds_dir / f"test_{name}.csv", index=False
        )
        validation = json.loads((metrics_dir / f"{name}.json").read_text(encoding="utf-8"))
        entry = {
            "probability_metrics": probability_metrics(y, p),
            "validation_probability_metrics": validation["probability_metrics"],
            "threshold_metrics": {
                rule: threshold_metrics(y, p, t) for rule, t in frozen["thresholds"].items()
            },
            "confidence_intervals": bootstrap_confidence_intervals(
                y,
                p,
                frozen["thresholds"]["f2_capacity_30pct"],
                n_resamples=boot["n_resamples"],
                confidence=boot["confidence"],
                random_seed=seed,
            ),  # fmt: skip
            "calibration_table": calibration_table(y, p).to_dict(orient="records"),
            "curves": curve_points(y, p),
        }
        results[name] = entry
        pm = entry["probability_metrics"]
        print(
            f"[test] {name}: ROC-AUC {pm['roc_auc']:.4f} PR-AUC {pm['pr_auc']:.4f} "
            f"Brier {pm['brier']:.4f} ECE {pm['expected_calibration_error']:.3f}"
        )

    comparisons = [
        {
            "model_a": chosen,
            "model_b": ref,
            **paired_bootstrap_difference(
                y,
                predictions[chosen],
                predictions[ref],
                metric=metric,
                n_resamples=boot["n_resamples"],
                random_seed=seed,
            ),
        }  # fmt: skip
        for ref in frozen["reference_models"]
        if ref != "dummy_prior"
        for metric in ["roc_auc", "pr_auc", "brier"]
    ]
    capacity_t = frozen["thresholds"]["f2_capacity_30pct"]
    output = {
        "evaluated_on": "test",
        "n_test": int(len(y)),
        "test_default_rate": float(y.mean()),
        "test_access_count": previous_runs + 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "frozen_decision": frozen,
        "models": results,
        "paired_comparisons": comparisons,
        "risk_bands": risk_band_table(y, predictions[chosen]).to_dict(orient="records"),
        "error_profiles_capacity_threshold": error_profiles(
            X_test, y, predictions[chosen], capacity_t
        ).to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(output, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
