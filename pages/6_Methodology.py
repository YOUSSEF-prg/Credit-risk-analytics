"""Methodology and limitations, rendered from the project documentation."""

from __future__ import annotations

import streamlit as st

from credit_risk.config import resolve_path
from dashboard.common import disclaimer, page_setup

page_setup("Methodology & Limitations", "📚")
disclaimer()

st.markdown(
    """
### Appropriate and inappropriate uses
| ✅ Appropriate | ❌ Not appropriate |
|---|---|
| Studying PD modelling, calibration, class-imbalance handling and threshold analysis | Making or supporting real credit decisions about any person |
| Comparing methods under a documented, leakage-safe protocol | Claiming performance on a current or different portfolio |
| Explaining how a model uses variables (predictive associations) | Interpreting SHAP values or coefficients as causal effects |
| Illustrating decision trade-offs under stated, hypothetical costs | Computing expected credit loss, IFRS 9 provisions or regulatory capital |

### Key limitations
- **One historical cross-section** (Taiwan, April–September 2005). Temporal stability is not validated.
- **Target** is the UCI "default payment next month" flag. It is not a regulatory definition of default.
- **Undocumented codes** (`PAY_*` −2/0, `EDUCATION` 0/5/6, `MARRIAGE` 0). Interpretations of them are assumptions.
- **No exposure, LGD, margin or loss data.** Monetary figures in the simulator are illustrative units.
- **Protected attributes** (sex, marital status, age) are model inputs, at the project owner's request. Descriptive fairness diagnostics are reported. Real use would require legal and fair-lending review.
- **One random split.** Test results come from a single held-out sample (6,000 clients). The confidence intervals reflect that sample's variability only.
"""
)

docs = {"Methodology": "docs/methodology.md", "Data dictionary": "docs/data_dictionary.md"}
for tab, path in zip(st.tabs(list(docs)), docs.values(), strict=True):
    with tab:
        file = resolve_path(path)
        if file.exists():
            st.markdown(file.read_text(encoding="utf-8"))
        else:
            st.warning(f"{path} not found.")
