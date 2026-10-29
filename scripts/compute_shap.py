"""Compute SHAP explanations for the frozen model on the test-set clients.

The model is not changed and no decision is re-made. The test PDs are those already
recorded by scripts/evaluate_test.py. SHAP only describes how the frozen model
produced them.

Outputs (artifacts/explainability/):
    shap_values_test_<model>.csv     SHAP values (log-odds) per client and feature
    features_test_<model>.csv        the feature values the model saw
    summary_<model>.json             base value, additivity check, global & grouped importance

Usage:
    python scripts/compute_shap.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import pandas as pd

from credit_risk.config import load_config, resolve_path
from credit_risk.data import (
    ID_COLUMN,
    DataSplits,
    apply_splits,
    file_sha256,
    load_raw,
    split_features_target,
    standardize_columns,
)
from credit_risk.explainability import (
    check_additivity,
    explain,
    global_importance,
    grouped_importance,
)


def main() -> None:
    data_cfg, model_cfg = load_config("data"), load_config("models")
    metrics_dir = resolve_path(model_cfg["outputs"]["metrics_dir"])
    models_dir = resolve_path(model_cfg["outputs"]["models_dir"])
    out_dir = resolve_path("artifacts/explainability")
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((metrics_dir / "frozen_decision.json").read_text(encoding="utf-8"))
    name = frozen["selected_model"]
    model_path = models_dir / f"{name}.joblib"
    if file_sha256(model_path) != frozen["model_file_sha256"][name]:
        raise SystemExit(f"{model_path} differs from the frozen model; refusing to explain it")
    model = joblib.load(model_path)

    ds = data_cfg["dataset"]
    df = standardize_columns(load_raw(resolve_path(ds["raw_path"]), ds["expected_sha256"]))
    splits = DataSplits.from_json(resolve_path(data_cfg["split"]["output_path"]))
    _, _, test = apply_splits(df, splits)
    X_test, _ = split_features_target(test)

    recorded = pd.read_csv(resolve_path(f"artifacts/predictions/test_{name}.csv"))
    if not recorded[ID_COLUMN].equals(test[ID_COLUMN].reset_index(drop=True)):
        raise SystemExit("Recorded test predictions are not aligned with the test split")

    result = explain(model, X_test)
    gap = check_additivity(result, recorded["pd"].to_numpy())

    ids = test[ID_COLUMN].to_numpy()
    result.values.assign(**{ID_COLUMN: ids}).to_csv(
        out_dir / f"shap_values_test_{name}.csv", index=False, float_format="%.6g"
    )
    result.features.assign(**{ID_COLUMN: ids}).to_csv(
        out_dir / f"features_test_{name}.csv", index=False, float_format="%.6g"
    )
    summary = {
        "model": name,
        "model_file_sha256": frozen["model_file_sha256"][name],
        "explained_split": "test",
        "n_clients": len(ids),
        "output_scale": "log-odds (logit of PD)",
        "base_value_log_odds": result.base_value,
        "additivity_max_abs_gap": gap,
        "global_importance": global_importance(result).to_dict(orient="records"),
        "grouped_importance": grouped_importance(result).to_dict(orient="records"),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "interpretation": "Predictive associations learned by the model, not causal effects.",
    }
    (out_dir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SHAP for {name} on {len(ids)} test clients; additivity gap {gap:.2e}")
    print(pd.DataFrame(summary["grouped_importance"]).to_string(index=False))


if __name__ == "__main__":
    main()
