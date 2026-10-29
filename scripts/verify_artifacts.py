"""Check that the stored artifacts are consistent with the code and data on this machine.

Checks (none of them reads test labels or changes any result):
1. The raw data file matches the recorded SHA-256.
2. The model files match the hashes frozen in ``frozen_decision.json``.
3. Reloading the frozen and reference models reproduces their stored **validation**
   predictions (max absolute difference).
4. The stored SHAP values add up to the stored test PDs.

Usage:
    python scripts/verify_artifacts.py
"""

from __future__ import annotations

import platform
import sys

import joblib
import numpy as np
import sklearn
import xgboost

from credit_risk.config import load_config, resolve_path
from credit_risk.data import file_sha256
from credit_risk.experiment import load_partitions
from credit_risk.models import predict_pd
from credit_risk.results import ResultsStore

TOLERANCE = 1e-6


def main() -> int:
    ok = True
    cfg = load_config("data")
    versions = f"scikit-learn {sklearn.__version__} | xgboost {xgboost.__version__}"
    print(f"Python {platform.python_version()} | {versions}")

    raw = resolve_path(cfg["dataset"]["raw_path"])
    same = file_sha256(raw) == cfg["dataset"]["expected_sha256"]
    print(f"[{'ok' if same else 'FAIL'}] raw data checksum")
    ok &= same

    store = ResultsStore()
    frozen = store.frozen_decision
    for name, expected in frozen["model_file_sha256"].items():
        same = file_sha256(resolve_path(f"artifacts/models/{name}.joblib")) == expected
        print(f"[{'ok' if same else 'FAIL'}] model file hash: {name}")
        ok &= same

    data = load_partitions(cfg)
    for name in frozen["model_file_sha256"]:
        model = joblib.load(resolve_path(f"artifacts/models/{name}.joblib"))
        stored = store.predictions("validation", name)
        if not stored["ID"].equals(data.val_ids.reset_index(drop=True)):
            print(f"[FAIL] validation IDs misaligned for {name}")
            ok = False
            continue
        gap = float(
            np.max(np.abs(predict_pd(model, data.X_val).to_numpy() - stored["pd"].to_numpy()))
        )
        good = gap < TOLERANCE
        status = "ok" if good else "FAIL"
        print(f"[{status}] {name}: validation predictions reproduced (max |diff| {gap:.1e})")
        ok &= good

    values, _ = store.shap_values()
    preds = store.predictions("test", store.selected_model)
    p = np.clip(preds["pd"].to_numpy(), 1e-12, 1 - 1e-12)
    logit = (
        store.shap_summary()["base_value_log_odds"]
        + values.drop(columns="ID").sum(axis=1).to_numpy()
    )
    gap = float(np.max(np.abs(logit - np.log(p / (1 - p)))))
    good = gap < 1e-3
    print(f"[{'ok' if good else 'FAIL'}] SHAP values add up to stored test PDs (max gap {gap:.1e})")
    ok &= good

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
