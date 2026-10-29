"""Experiment orchestration: train, tune, select a threshold and evaluate on validation.

Writes structured outputs that the dashboard and the RAG metrics tool read directly:

* ``artifacts/metrics/<model>.json``: full validation results for one model
* ``artifacts/metrics/summary.json``: comparison table over all models
* ``artifacts/predictions/validation_<model>.csv``: ID, target and predicted PD
* ``artifacts/models/<model>.joblib``: fitted pipeline

The test partition is deliberately not evaluated here. It is used once, for the final
selected models, by a separate script.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from credit_risk.calibration import fit_calibrated
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
from credit_risk.decision_analysis import select_threshold
from credit_risk.evaluation import (
    bootstrap_confidence_intervals,
    calibration_table,
    curve_points,
    group_metrics,
    paired_bootstrap_difference,
    probability_metrics,
    threshold_metrics,
)
from credit_risk.models import build_pipeline, predict_pd, tune_pipeline
from credit_risk.validation import validate_standardized_schema

AGE_BANDS = [20, 30, 40, 50, 60, 80]
CI_KEYS = ["roc_auc", "pr_auc", "brier", "recall", "f2"]


@dataclass
class PartitionedData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    val_ids: pd.Series
    raw_sha256: str
    splits: DataSplits


def load_partitions(data_cfg: dict[str, Any]) -> PartitionedData:
    """Load the raw data, validate it, and return train and validation partitions only."""
    ds = data_cfg["dataset"]
    raw_path = resolve_path(ds["raw_path"])
    df = standardize_columns(load_raw(raw_path, expected_sha256=ds["expected_sha256"]))
    validate_standardized_schema(df, expected_rows=ds["expected_rows"])
    splits = DataSplits.from_json(resolve_path(data_cfg["split"]["output_path"]))
    train, val, _test = apply_splits(df, splits)  # test intentionally unused
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(val)
    return PartitionedData(
        X_train, y_train, X_val, y_val, val[ID_COLUMN], file_sha256(raw_path), splits
    )


def evaluate_on_validation(
    y_val: pd.Series,
    p_val: pd.Series,
    X_val: pd.DataFrame,
    beta: float,
    bootstrap_cfg: dict[str, Any],
    random_seed: int,
    max_flagged_rate: float | None = None,
) -> dict[str, Any]:
    """Threshold selection (F-beta) plus all metrics and intervals on validation data.

    Two decision rules are evaluated: unconstrained F-beta, and F-beta subject to a
    capacity constraint (at most ``max_flagged_rate`` of clients flagged).
    """
    y, p = y_val.to_numpy(), p_val.to_numpy()
    chosen = select_threshold(y, p, beta=beta)
    age_band = pd.cut(X_val["AGE"], AGE_BANDS, right=False).astype(str)
    capacity: dict[str, Any] = {}
    if max_flagged_rate is not None:
        constrained = select_threshold(y, p, beta=beta, max_flagged_rate=max_flagged_rate)
        capacity = {
            "threshold_selection_capacity": constrained,
            "threshold_metrics_capacity": threshold_metrics(y, p, constrained["threshold"]),
        }
    return {
        **capacity,
        "threshold_selection": chosen,
        "probability_metrics": probability_metrics(y, p),
        "threshold_metrics": threshold_metrics(y, p, chosen["threshold"]),
        "threshold_metrics_at_0_5": threshold_metrics(y, p, 0.5),
        "confidence_intervals": bootstrap_confidence_intervals(
            y,
            p,
            chosen["threshold"],
            n_resamples=bootstrap_cfg["n_resamples"],
            confidence=bootstrap_cfg["confidence"],
            random_seed=random_seed,
        ),
        "calibration_table": calibration_table(y, p).to_dict(orient="records"),
        "curves": curve_points(y, p),
        "group_metrics": {
            "SEX": group_metrics(y, p, X_val["SEX"], chosen["threshold"]).to_dict("records"),
            "AGE_BAND": group_metrics(y, p, age_band, chosen["threshold"]).to_dict("records"),
        },
    }


def resolve_spec(name: str, models_cfg: dict[str, Any], metrics_dir: Path) -> dict[str, Any]:
    """Expand ``inherit: <model>`` into a full specification.

    The inherited model's specification is copied, its tuned hyperparameters (read from
    ``artifacts/metrics/<model>.json``) are fixed as parameters, its search space is
    dropped, and the keys given for ``name`` override the result. A strategy variant
    (resampling, weights, calibration) therefore reuses the base model's tuned
    configuration and changes only the strategy.
    """
    spec = dict(models_cfg[name])
    parent = spec.pop("inherit", None)
    if parent is None:
        return spec
    if parent not in models_cfg:
        raise KeyError(f"{name}: inherited model {parent!r} not in config")
    base = resolve_spec(parent, models_cfg, metrics_dir)
    parent_metrics = metrics_dir / f"{parent}.json"
    if not parent_metrics.exists():
        raise FileNotFoundError(f"{name} inherits from {parent}; run {parent} first")
    tuned = json.loads(parent_metrics.read_text(encoding="utf-8"))["cv"]["best_params"]
    params = {**base.get("params", {}), **tuned, **spec.pop("params", {})}
    merged = {k: v for k, v in base.items() if k not in {"search", "description"}}
    merged.update(spec)
    merged["params"] = params
    merged["inherits_from"] = parent
    merged.setdefault("search", {})
    return merged


def _config_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def run_experiments(
    model_names: list[str] | None = None, config_name: str = "models", resume: bool = False
) -> dict[str, Any]:
    """Run the configured models in order.

    With ``resume=True``, a model is skipped if its metrics file already exists and was
    produced by an identical specification (same hash), e.g. after an interrupted run.
    """
    data_cfg, cfg = load_config("data"), load_config(config_name)
    exp, outputs = cfg["experiment"], cfg["outputs"]
    seed = exp["random_seed"]
    metrics_dir = resolve_path(outputs["metrics_dir"])
    models_dir = resolve_path(outputs["models_dir"])
    preds_dir = resolve_path(outputs["predictions_dir"])
    for d in (metrics_dir, models_dir, preds_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = load_partitions(data_cfg)
    names = model_names or list(cfg["models"])
    unknown = set(names) - set(cfg["models"])
    if unknown:
        raise KeyError(f"Unknown models: {sorted(unknown)}")

    tracker = _MlflowTracker(cfg["tracking"], exp["name"])
    results: dict[str, Any] = {}
    for name in names:
        spec = resolve_spec(name, cfg["models"], metrics_dir)
        calibration = spec.pop("calibration", "none")
        existing = metrics_dir / f"{name}.json"
        if resume and existing.exists():
            previous = json.loads(existing.read_text(encoding="utf-8"))
            same = previous.get("specification_hash") == _config_hash(spec)
            if same and previous.get("calibration", "none") == calibration:
                print(f"[{name}] skipped (resume: identical specification already evaluated)")
                results[name] = previous
                continue
        start = time.perf_counter()
        if calibration == "none":
            print(f"[{name}] {exp['cv_folds']}-fold CV on train (refit on {exp['tuning_scoring']})")
            model, cv_summary = tune_pipeline(
                build_pipeline(spec, random_seed=seed),
                spec.get("search", {}),
                data.X_train,
                data.y_train,
                scoring=exp["tuning_scoring"],
                cv_folds=exp["cv_folds"],
                random_seed=seed,
                n_jobs=exp["n_jobs"],
                search_type=spec.get("search_type", "grid"),
                n_iter=spec.get("n_iter", 20),
            )
        else:
            print(
                f"[{name}] {calibration} calibration, {exp['cv_folds']}-fold out-of-fold on train"
            )
            model = fit_calibrated(
                spec, calibration, data.X_train, data.y_train,
                cv_folds=exp["cv_folds"], random_seed=seed, n_jobs=exp["n_jobs"],
            )  # fmt: skip
            cv_summary = {"best_params": {}, "search_type": "none (calibrated wrapper)"}
        fit_seconds = time.perf_counter() - start
        p_val = predict_pd(model, data.X_val)
        evaluation = evaluate_on_validation(
            data.y_val,
            p_val,
            data.X_val,
            cfg["threshold"]["beta"],
            cfg["bootstrap"],
            seed,
            max_flagged_rate=cfg["threshold"].get("max_flagged_rate"),
        )
        record = {
            "model": name,
            "description": spec.get("description", ""),
            "specification": spec,
            "specification_hash": _config_hash(spec),
            "calibration": calibration,
            "evaluated_on": "validation",
            "n_train": len(data.y_train),
            "n_validation": len(data.y_val),
            "cv": cv_summary,
            "fit_seconds": round(fit_seconds, 2),
            "data_sha256": data.raw_sha256,
            "split_seed": data.splits.random_seed,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
            },
            **evaluation,
        }
        (metrics_dir / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        pd.DataFrame(
            {ID_COLUMN: data.val_ids.values, TARGET: data.y_val.values, "pd": p_val.values}
        ).to_csv(preds_dir / f"validation_{name}.csv", index=False)
        joblib.dump(model, models_dir / f"{name}.joblib")
        tracker.log(name, record, metrics_dir / f"{name}.json")
        results[name] = record
        pm, tm = record["probability_metrics"], record["threshold_metrics"]
        cap = record.get("threshold_metrics_capacity", {})
        print(
            f"[{name}] ROC-AUC {pm['roc_auc']:.4f} | PR-AUC {pm['pr_auc']:.4f} | "
            f"Brier {pm['brier']:.4f} | ECE {pm['expected_calibration_error']:.3f} | "
            f"F2@{tm['threshold']:.3f}: R {tm['recall']:.3f} P {tm['precision']:.3f} "
            f"flag {tm['flagged_rate']:.2f}"
            + (f" | cap@{cap['threshold']:.3f}: R {cap['recall']:.3f} P {cap['precision']:.3f}"
               f" F2 {cap['f2']:.3f}" if cap else "")
            + f" | {fit_seconds:.0f}s"
        )  # fmt: skip

    write_summary(metrics_dir)
    write_comparisons(cfg.get("comparisons", []), preds_dir, metrics_dir, cfg["bootstrap"], seed)
    if "selection" in cfg:
        from credit_risk.selection import select_model

        chosen = select_model(
            metrics_dir,
            preds_dir,
            n_resamples=cfg["bootstrap"]["n_resamples"],
            random_seed=seed,
            **cfg["selection"],
        )
        print(f"Selected model (validation rule): {chosen['selected_model']}")
    return results


def write_comparisons(
    pairs: list[list[str]],
    preds_dir: Path,
    metrics_dir: Path,
    bootstrap_cfg: dict[str, Any],
    random_seed: int,
) -> list[dict[str, Any]]:
    """Paired bootstrap differences (A - B) on validation predictions, for configured pairs."""
    rows = []
    for a, b in pairs:
        path_a, path_b = preds_dir / f"validation_{a}.csv", preds_dir / f"validation_{b}.csv"
        if not (path_a.exists() and path_b.exists()):
            continue
        pa, pb = pd.read_csv(path_a), pd.read_csv(path_b)
        if not pa[ID_COLUMN].equals(pb[ID_COLUMN]):
            raise ValueError(f"Prediction files for {a} and {b} are not aligned by ID")
        for metric in ["roc_auc", "pr_auc", "brier"]:
            result = paired_bootstrap_difference(
                pa[TARGET].to_numpy(),
                pa["pd"].to_numpy(),
                pb["pd"].to_numpy(),
                metric=metric,
                n_resamples=bootstrap_cfg["n_resamples"],
                confidence=bootstrap_cfg["confidence"],
                random_seed=random_seed,
            )
            rows.append({"model_a": a, "model_b": b, "evaluated_on": "validation", **result})
    (metrics_dir / "comparisons.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def write_summary(metrics_dir: Path) -> pd.DataFrame:
    """Rebuild ``summary.json`` from all per-model metric files in ``metrics_dir``."""
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name in {"summary.json", "comparisons.json", "selection.json"}:
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("evaluated_on") != "validation" or "model" not in rec:
            continue  # e.g. test_evaluation.json, frozen_decision.json
        pm, tm, ci = (
            rec["probability_metrics"],
            rec["threshold_metrics"],
            rec["confidence_intervals"],
        )
        rows.append(
            {
                "model": rec["model"],
                "calibration": rec.get("calibration", "none"),
                "evaluated_on": rec["evaluated_on"],
                **{k: pm[k] for k in ["roc_auc", "pr_auc", "brier", "log_loss",
                                       "expected_calibration_error", "mean_predicted_pd"]},
                "threshold": tm["threshold"],
                **{k: tm[k] for k in ["precision", "recall", "f1", "f2", "flagged_rate"]},
                **{f"{k}_ci_low": ci[k]["low"] for k in CI_KEYS},
                **{f"{k}_ci_high": ci[k]["high"] for k in CI_KEYS},
                **{f"capacity_{k}": rec.get("threshold_metrics_capacity", {}).get(k)
                   for k in ["threshold", "precision", "recall", "f1", "f2", "flagged_rate"]},
                **{f"cv_{k}": rec["cv"].get("cv_metrics_mean", {}).get(k)
                   for k in ["average_precision", "roc_auc", "brier"]},
                "cv_average_precision_std": rec["cv"].get("cv_metrics_std", {}).get(
                    "average_precision"),
                "sampler": (rec["specification"].get("sampler") or {}).get("name", "none"),
                "inherits_from": rec["specification"].get("inherits_from"),
                "fit_seconds": rec.get("fit_seconds"),
                "created_utc": rec["created_utc"],
            }
        )  # fmt: skip
    table = pd.DataFrame(rows)
    (metrics_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "Validation-set results. The test set is reserved for final evaluation.",
                "models": table.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return table


class _MlflowTracker:
    """Thin optional MLflow wrapper; the experiment JSON files are the source of truth."""

    def __init__(self, tracking_cfg: dict[str, Any], experiment_name: str):
        self.enabled = bool(tracking_cfg.get("mlflow_enabled"))
        if not self.enabled:
            return
        try:
            import mlflow

            uri = tracking_cfg["mlflow_tracking_uri"]
            if uri.startswith("sqlite:///") and not Path(uri[10:]).is_absolute():
                uri = f"sqlite:///{resolve_path(uri[10:]).as_posix()}"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001 - tracking must never break an experiment
            print(f"MLflow disabled: {exc}")
            self.enabled = False

    def log(self, run_name: str, record: dict[str, Any], artifact: Path) -> None:
        if not self.enabled:
            return
        mlflow = self._mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "estimator": record["specification"]["estimator"],
                    "calibration": record["calibration"],
                    "sampler": (record["specification"].get("sampler") or {}).get("name", "none"),
                    "spec_hash": record["specification_hash"],
                    **{f"best_{k}": v for k, v in record["cv"]["best_params"].items()},
                    "threshold_objective": record["threshold_selection"]["objective"],
                }
            )
            metrics = {f"val_{k}": v for k, v in record["probability_metrics"].items()}
            metrics.update(
                {
                    f"val_{k}": record["threshold_metrics"][k]
                    for k in ["threshold", "precision", "recall", "f1", "f2"]
                }
            )
            if "threshold_metrics_capacity" in record:
                metrics.update(
                    {
                        f"val_capacity_{k}": record["threshold_metrics_capacity"][k]
                        for k in ["threshold", "precision", "recall", "f2"]
                    }
                )
            if "best_cv_score_mean" in record["cv"]:
                metrics["cv_score_mean"] = record["cv"]["best_cv_score_mean"]
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(artifact))
