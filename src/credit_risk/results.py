"""Read-only interface to stored experiment results.

This is the single source of numerical results for the dashboard and the RAG assistant.
Every figure shown to a user comes from files written by the experiment scripts:

* ``artifacts/metrics/*.json``: per-model validation results, summary, comparisons,
  selection, frozen decision, test evaluation
* ``artifacts/predictions/{validation,test}_<model>.csv``: PDs per client
* ``artifacts/explainability/``: SHAP values of the frozen model on the test set

Nothing here trains a model, recomputes a selection or touches the test labels beyond
reading the stored evaluation output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk.config import resolve_path

META_FILES = {"summary", "comparisons", "selection", "frozen_decision", "test_evaluation"}


class ResultsNotFoundError(FileNotFoundError):
    """Raised when an expected experiment output has not been produced yet."""


@dataclass
class ResultsStore:
    root: Path = resolve_path("artifacts")

    # ------------------------------------------------------------------ helpers
    def _json(self, relative: str) -> Any:
        path = self.root / relative
        if not path.exists():
            raise ResultsNotFoundError(
                f"{path} not found. Run the experiment scripts first (see README)."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ validation results
    @cached_property
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self._json("metrics/summary.json")["models"]).set_index("model")

    def model_names(self) -> list[str]:
        return sorted(
            p.stem
            for p in (self.root / "metrics").glob("*.json")
            if p.stem not in META_FILES and not p.stem.startswith("test_")
        )

    def model_record(self, name: str) -> dict[str, Any]:
        if name in META_FILES or name not in self.model_names():
            raise KeyError(f"Unknown model {name!r}. Available: {self.model_names()}")
        return self._json(f"metrics/{name}.json")

    @cached_property
    def comparisons(self) -> pd.DataFrame:
        return pd.DataFrame(self._json("metrics/comparisons.json"))

    @cached_property
    def selection(self) -> dict[str, Any]:
        return self._json("metrics/selection.json")

    # ------------------------------------------------------------------ frozen model & test
    @cached_property
    def frozen_decision(self) -> dict[str, Any]:
        return self._json("metrics/frozen_decision.json")

    @cached_property
    def test_evaluation(self) -> dict[str, Any]:
        return self._json("metrics/test_evaluation.json")

    @property
    def selected_model(self) -> str:
        return self.frozen_decision["selected_model"]

    # ------------------------------------------------------------------ predictions
    def predictions(self, split: str, model: str) -> pd.DataFrame:
        if split not in {"validation", "test"}:
            raise ValueError("split must be 'validation' or 'test'")
        path = self.root / "predictions" / f"{split}_{model}.csv"
        if not path.exists():
            raise ResultsNotFoundError(f"No {split} predictions for {model!r} ({path})")
        return pd.read_csv(path)

    # ------------------------------------------------------------------ explainability
    def shap_summary(self, model: str | None = None) -> dict[str, Any]:
        return self._json(f"explainability/summary_{model or self.selected_model}.json")

    def shap_values(self, model: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        name = model or self.selected_model
        values = pd.read_csv(self.root / "explainability" / f"shap_values_test_{name}.csv")
        features = pd.read_csv(self.root / "explainability" / f"features_test_{name}.csv")
        return values, features

    # ------------------------------------------------------------------ lookups for the RAG tool
    def metric(self, model: str, metric: str, split: str = "validation") -> float:
        """Return one stored metric, e.g. metric('xgboost', 'brier', 'test')."""
        if split == "test":
            models = self.test_evaluation["models"]
            if model not in models:
                raise KeyError(f"{model!r} was not evaluated on the test set")
            source = models[model]["probability_metrics"]
        else:
            source = self.model_record(model)["probability_metrics"]
        if metric not in source:
            raise KeyError(f"Unknown metric {metric!r}. Available: {sorted(source)}")
        return float(source[metric])
