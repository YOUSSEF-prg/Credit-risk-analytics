"""Credit Risk Analytics dashboard: entry point and navigation.

Run from the project root, inside the ``credit-risk`` conda environment:
    conda activate credit-risk
    streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st

# Make the project packages (src/credit_risk, src/dashboard) importable even if the
# project was not pip-installed in the active environment.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED = [
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "shap",
    "plotly",
    "imblearn",
    "python_calamine",
    "yaml",
    "joblib",
]
missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
if missing:
    st.set_page_config(page_title="Credit Risk Analytics", page_icon="🏦")
    st.error(
        f"Missing packages in this Python environment: {', '.join(missing)}.\n\n"
        f"Streamlit is running from `{sys.executable}`. "
        "The dashboard needs the project's `credit-risk` conda environment:\n\n"
        "```\nconda activate credit-risk\nstreamlit run app.py\n```\n\n"
        "or double-click `scripts\\run_dashboard.bat`."
    )
    st.stop()

pages = [
    st.Page("pages/0_Home.py", title="Home", icon="🏦", default=True),
    st.Page("pages/1_Portfolio_Overview.py", title="Portfolio Overview", icon="📈"),
    st.Page("pages/2_Model_Performance.py", title="Model Performance", icon="🎯"),
    st.Page("pages/3_Decision_Simulator.py", title="Decision Simulator", icon="🎚️"),
    st.Page("pages/4_Explainability.py", title="Explainability", icon="🔍"),
    st.Page("pages/6_Methodology.py", title="Methodology", icon="📚"),
]
st.navigation(pages).run()
