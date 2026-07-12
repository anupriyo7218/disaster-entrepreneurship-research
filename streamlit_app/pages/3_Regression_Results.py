"""
pages/3_Regression_Results.py

Displays the Module 4 TWFE regression output and explains it in plain
language for a non-statistician reader — a recruiter or committee member
shouldn't need an econometrics background to understand what the number
means and what it doesn't prove.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import figure_path_if_exists, load_report_text, load_table, missing_pipeline_warning  # noqa: E402

st.set_page_config(page_title="Regression Results", page_icon="📉", layout="wide")
st.title("Regression Results")
st.caption("Two-Way Fixed Effects panel regression — observational and correlational only.")

reg_table = load_table("twfe_regression_table.csv")
diag_table = load_table("twfe_diagnostics.csv")

if reg_table is None or diag_table is None:
    missing_pipeline_warning("Module 4 (regression tables)", "python -m src.modeling")
    st.stop()

diagnostics = dict(zip(diag_table["metric"], diag_table["value"]))
coef = reg_table.iloc[0]["coefficient"]
se = reg_table.iloc[0]["std_error"]
tstat = reg_table.iloc[0]["t_statistic"]
pval = reg_table.iloc[0]["p_value"]
ci_lower = reg_table.iloc[0]["ci_lower_95"]
ci_upper = reg_table.iloc[0]["ci_upper_95"]
significant = pval < 0.05

# --- The model, stated plainly -------------------------------------------
st.header("The Model")
st.latex(
    r"\text{business\_applications}_{it} = \beta \cdot \text{disaster\_count}_{it} "
    r"+ \alpha_i \text{ (state fixed effects)} + \gamma_t \text{ (month-year fixed effects)} + \varepsilon_{it}"
)
st.markdown(
    """
- **State fixed effects** (`αᵢ`) absorb anything permanently different about a state —
  its size, industry mix, baseline entrepreneurial climate.
- **Month-year fixed effects** (`γₜ`) absorb anything common to *all* states in a given
  month — a national recession, a seasonal pattern, a federal policy change.
- What's left over, `β`, is the association between disaster counts and business
  applications **within** a state, **after** removing that month's nationwide trend.
"""
)

st.divider()

# --- Headline numbers ------------------------------------------------------
st.header("Headline Result")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Coefficient (β)", f"{coef:,.2f}")
m2.metric("Std. Error (clustered)", f"{se:,.2f}")
m3.metric("p-value", f"{pval:.4f}")
m4.metric("Significant at 5%?", "Yes" if significant else "No")

st.markdown(f"**95% Confidence Interval:** [{ci_lower:,.2f}, {ci_upper:,.2f}]")

if not significant:
    st.warning(
        "The confidence interval crosses zero, so this analysis **cannot rule out no "
        "association at all** between disaster declarations and business applications, "
        "once state and month-year effects are controlled for."
    )
else:
    st.success("This result is statistically significant at the 5% level.")

st.divider()

# --- Plain-language explanation for each statistic -----------------------
st.header("What Do These Numbers Mean? (No Statistics Background Needed)")

with st.expander("**Coefficient** — what is it?", expanded=True):
    st.markdown(
        f"""
The coefficient ({coef:,.2f}) is the estimated change in monthly business applications
associated with **one additional disaster** being declared in that state that month —
after accounting for that state's usual baseline and that month's nationwide conditions.
A {"negative" if coef < 0 else "positive"} number means more disasters are associated
with {"fewer" if coef < 0 else "more"} business applications in the data, on average.
"""
    )

with st.expander("**p-value** — what is it?"):
    st.markdown(
        f"""
The p-value ({pval:.4f}) answers: *"if disasters truly had no relationship at all with
business applications, how likely would we be to see a coefficient this large just by
random chance?"* A common rule of thumb is that a p-value below 0.05 counts as
"statistically significant." Here, the p-value is {"below" if significant else "above"}
0.05, so this result is {"" if significant else "**not**"} considered statistically
significant by that standard.
"""
    )

with st.expander("**Confidence interval** — what is it?"):
    st.markdown(
        f"""
The 95% confidence interval ([{ci_lower:,.2f}, {ci_upper:,.2f}]) is a range of plausible
true values for the coefficient, given the data. {"Because this range does not include "
"zero, it supports a real association." if significant else "Because this range includes "
"zero, a true effect of exactly zero (no relationship) cannot be ruled out."}
"""
    )

with st.expander("**Limitations** — what this analysis does NOT show"):
    st.markdown(
        """
- This is **observational data**, not a controlled experiment — states weren't randomly
  assigned disasters.
- Fixed effects remove a lot of confounding, but **not everything**. A state could
  experience both a disaster and an economic shift in the same month due to some other
  shared cause (e.g., a broader regional economic event).
- **Reverse causality** isn't ruled out by this design.
- The FEMA `disaster_count` measure reflects **federal declarations**, which mix physical
  severity with an administrative/political process — it's not a pure measure of disaster
  intensity.

**In short: this is an association, not proof of a causal effect.**
"""
    )

st.divider()

# --- Full diagnostics table -------------------------------------------------
st.header("Full Model Diagnostics")
st.dataframe(diag_table, use_container_width=True, hide_index=True)

st.divider()

# --- Figures ---------------------------------------------------------------
st.header("Diagnostic Plots")
figure_files = [
    ("twfe_coefficient_plot.png", "Coefficient estimate with 95% CI"),
    ("twfe_confidence_interval_plot.png", "Confidence interval range"),
    ("twfe_residual_histogram.png", "Residual distribution"),
    ("twfe_residual_qq_plot.png", "Residual Q-Q plot (normality check)"),
    ("twfe_fitted_vs_observed.png", "Fitted vs. observed values"),
]
cols = st.columns(2)
for i, (filename, caption) in enumerate(figure_files):
    path = figure_path_if_exists(filename)
    with cols[i % 2]:
        if path:
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.info(f"{caption}: not yet generated.")

st.divider()

# --- Full statistical summary (collapsible, for the technically inclined) --
full_summary = load_report_text("twfe_full_summary.txt")
if full_summary:
    with st.expander("Full statistical summary (linearmodels output)"):
        st.code(full_summary, language=None)
