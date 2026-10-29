"""Pre-declared model-selection rule, applied to validation results only.

Rule (fixed before the final run, documented in docs/methodology.md):

1. **Eligibility.** The model's scores must be usable as PDs: validation expected
   calibration error ≤ ``max_ece``, and mean predicted PD within ``max_mean_pd_gap`` of
   the observed default rate. This excludes uncalibrated resampled or reweighted models.
2. **Primary criterion.** Highest validation PR-AUC (average precision) among eligible
   models. PR-AUC focuses on the default class, in line with the recall-oriented
   objective.
3. **Parsimony.** Any eligible model whose PR-AUC is not significantly below the best
   (the paired-bootstrap 95% interval of the difference includes 0) is treated as
   equivalent. Among equivalent models the simplest is chosen: logistic regression
   < random forest < gradient boosting; within a family, no resampling < class weights
   < resampling; no calibration < sigmoid < isotonic.

The test set is not involved. It is used once, afterwards, to report the selected
model's performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk.data import ID_COLUMN, TARGET
from credit_risk.evaluation import paired_bootstrap_difference

FAMILY_RANK = {
    "dummy": 0,
    "logistic_regression": 1,
    "random_forest": 2,
    "balanced_random_forest": 3,
    "xgboost": 3,
    "easy_ensemble": 4,
    "rusboost": 4,
}
CALIBRATION_RANK = {"none": 0, "sigmoid": 1, "isotonic": 2}


def _strategy_rank(spec: dict[str, Any]) -> int:
    if spec.get("sampler"):
        return 2
    params = spec.get("params", {})
    if params.get("class_weight") or params.get("scale_pos_weight"):
        return 1
    return 0


def _complexity(record: dict[str, Any]) -> tuple[int, int, int, int]:
    spec = record["specification"]
    uses_features = int(spec.get("features", {}).get("add_features", True))
    return (
        FAMILY_RANK.get(spec["estimator"], 9),
        _strategy_rank(spec),
        CALIBRATION_RANK.get(record.get("calibration", "none"), 9),
        uses_features,
    )


def select_model(
    metrics_dir: Path,
    predictions_dir: Path,
    max_ece: float = 0.03,
    max_mean_pd_gap: float = 0.02,
    n_resamples: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Apply the selection rule and write ``selection.json`` to ``metrics_dir``."""
    records = {}
    for path in sorted(metrics_dir.glob("*.json")):
        if path.stem in {"summary", "comparisons", "selection"}:
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("evaluated_on") == "validation" and rec["specification"]["estimator"] != "dummy":
            records[rec["model"]] = rec

    rows = []
    for name, rec in records.items():
        pm = rec["probability_metrics"]
        gap = abs(pm["mean_predicted_pd"] - pm["observed_default_rate"])
        rows.append(
            {
                "model": name,
                "pr_auc": pm["pr_auc"],
                "roc_auc": pm["roc_auc"],
                "brier": pm["brier"],
                "ece": pm["expected_calibration_error"],
                "mean_pd_gap": gap,
                "eligible": pm["expected_calibration_error"] <= max_ece and gap <= max_mean_pd_gap,
                "complexity": _complexity(rec),
            }
        )
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    eligible = table[table["eligible"]]
    if eligible.empty:
        raise ValueError("No model satisfies the calibration eligibility criteria")
    best = eligible.iloc[0]["model"]

    best_preds = pd.read_csv(predictions_dir / f"validation_{best}.csv")
    equivalence = []
    for name in eligible["model"]:
        if name == best:
            equivalence.append({"model": name, "difference": 0.0, "low": 0.0, "high": 0.0})
            continue
        other = pd.read_csv(predictions_dir / f"validation_{name}.csv")
        if not other[ID_COLUMN].equals(best_preds[ID_COLUMN]):
            raise ValueError(f"Predictions of {name} are not aligned with {best}")
        diff = paired_bootstrap_difference(
            best_preds[TARGET].to_numpy(),
            other["pd"].to_numpy(),
            best_preds["pd"].to_numpy(),
            metric="pr_auc",
            n_resamples=n_resamples,
            random_seed=random_seed,
        )
        equivalence.append({"model": name, **{k: diff[k] for k in ["difference", "low", "high"]}})
    eq = pd.DataFrame(equivalence)
    eq["equivalent_to_best"] = eq["high"] >= 0  # interval of (other - best) includes 0
    candidates = eligible.merge(eq, on="model")
    candidates = candidates[candidates["equivalent_to_best"]]
    chosen = candidates.sort_values(["complexity", "pr_auc"], ascending=[True, False]).iloc[0]

    result = {
        "rule": {
            "eligibility": f"ECE <= {max_ece} and |mean PD - observed rate| <= {max_mean_pd_gap}",
            "primary": "highest validation PR-AUC among eligible models",
            "parsimony": "simplest model whose PR-AUC is not significantly below the best "
            "(paired bootstrap, 95%)",
        },
        "evaluated_on": "validation",
        "best_pr_auc_model": best,
        "selected_model": chosen["model"],
        "equivalent_models": candidates["model"].tolist(),
        "ranking": table.drop(columns="complexity").to_dict(orient="records"),
        "pr_auc_differences_vs_best": eq.to_dict(orient="records"),
    }
    (metrics_dir / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
