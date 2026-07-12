"""
Home.py

Streamlit dashboard entry point — Project Overview page.

Run with: streamlit run streamlit_app/Home.py
Additional pages live in streamlit_app/pages/ and appear automatically in
the sidebar (Streamlit's native multipage convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `utils` importable from pages/ subdirectory scripts too — Streamlit
# adds the main script's directory to sys.path, but being explicit here
# avoids any ambiguity about where the shared helpers live.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_final_dataset, load_fema_clean, load_table  # noqa: E402

st.set_page_config(page_title="Disaster & Business Formation", page_icon="📊", layout="wide")

st.title("Do Natural Disasters Affect New Business Creation in the U.S.?")
st.caption("A reproducible empirical research project — Sustainable Entrepreneurship & Innovation")

st.divider()

# --- Objective -------------------------------------------------------------
st.header("Objective")
st.markdown(
    """
This project asks a simple question: **when a disaster is declared in a U.S. state,
does that state see more or fewer new businesses being started in the following month?**

It's built as a full, reproducible empirical pipeline — from raw government data
through cleaning, exploratory analysis, and a formal statistical model — the way
an economics or entrepreneurship research paper would approach it.
"""
)

# --- Methodology -------------------------------------------------------------
st.header("Methodology, in Brief")
st.markdown(
    """
1. **Clean and merge** two independent government datasets into a single state-by-month panel.
2. **Explore the data descriptively** first — distributions, trends, and simple comparisons —
   before touching any statistical model.
3. **Estimate a two-way fixed effects regression**, which controls for permanent differences
   between states (size, industry mix) and nationwide shocks common to a given month
   (recessions, seasonality), isolating the within-state, within-month association between
   disasters and business applications.
4. **Report the result honestly** — this is an observational, correlational study.
   It does not claim to prove that disasters cause changes in business formation.
"""
)

# --- Data sources -------------------------------------------------------------
st.header("Data Sources")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Business Formation Statistics (BFS)")
    st.markdown(
        """
- **Source:** U.S. Census Bureau
- **Measure used:** `BA_BA` — core Business Applications series, not seasonally adjusted
- **Granularity:** state, monthly
- **Coverage:** July 2004 – present
"""
    )
with col2:
    st.subheader("Disaster Declarations Summaries")
    st.markdown(
        """
- **Source:** FEMA / OpenFEMA v2
- **Measure used:** distinct federal disaster declarations per state-month
- **Granularity:** state (originally county-level, aggregated up)
- **Coverage:** 1953 – present (this analysis uses the overlap with BFS)
"""
    )

# --- Main findings, pulled live from the actual output files ---------------
st.header("Main Findings")

reg_table = load_table("twfe_regression_table.csv")
overview_table = load_table("eda_dataset_overview.csv")

if reg_table is not None:
    coef = reg_table.iloc[0]["coefficient"]
    pval = reg_table.iloc[0]["p_value"]
    ci_lower = reg_table.iloc[0]["ci_lower_95"]
    ci_upper = reg_table.iloc[0]["ci_upper_95"]
    significant = pval < 0.05

    m1, m2, m3 = st.columns(3)
    m1.metric("Coefficient (disaster_count)", f"{coef:,.1f}")
    m2.metric("p-value", f"{pval:.4f}")
    m3.metric("Statistically significant?", "Yes" if significant else "No")

    st.markdown(
        f"""
Holding each state's baseline characteristics and each month's nationwide conditions fixed,
one additional disaster declaration in a state-month is associated with a change of about
**{coef:,.0f} business applications** in that state-month (95% CI: [{ci_lower:,.0f}, {ci_upper:,.0f}]).

{"This result **is** statistically significant at the 5% level." if significant else
 "This result is **not statistically significant** at the 5% level — the confidence interval "
 "crosses zero, so we cannot rule out no association at all."}

See the **Regression Results** page for the full breakdown, and the **Conclusions** page for
what this does and doesn't tell us.
"""
    )
else:
    st.info(
        "Regression results not yet generated. Run `python -m src.modeling` from the "
        "project root, then reload this page."
    )

if overview_table is not None:
    overview = dict(zip(overview_table["metric"], overview_table["value"]))
    st.caption(
        f"Panel: {overview.get('n_states', '—')} states, "
        f"{overview.get('date_range_start', '—')} to {overview.get('date_range_end', '—')}, "
        f"{overview.get('n_rows', '—')} state-month observations."
    )

st.divider()
st.caption(
    "Use the sidebar to explore the raw data, interactive EDA charts, full regression "
    "output, and conclusions."
)
