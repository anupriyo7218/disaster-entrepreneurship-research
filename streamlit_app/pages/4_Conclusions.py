"""
pages/4_Conclusions.py

Ties the project together: main findings, honest limitations, and future
work. This is the page a recruiter or committee member would read last —
it should read like a paper's conclusion section, not a code summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_table  # noqa: E402

st.set_page_config(page_title="Conclusions", page_icon="📝", layout="wide")
st.title("Conclusions")

reg_table = load_table("twfe_regression_table.csv")

st.header("Main Findings")
if reg_table is not None:
    coef = reg_table.iloc[0]["coefficient"]
    pval = reg_table.iloc[0]["p_value"]
    significant = pval < 0.05
    st.markdown(
        f"""
1. **Raw, unadjusted comparison (Module 3):** disaster months show a weak positive
   correlation with business applications (Pearson r ≈ 0.11) — but this is almost
   certainly driven by the fact that large, economically active states (California,
   Texas) also accumulate the most disaster declarations.

2. **Two-way fixed effects regression (Module 4):** once state-level differences and
   nationwide month-to-month conditions are controlled for, the estimated association
   between disaster count and business applications is **{coef:,.1f}**, and is
   **{"statistically significant" if significant else "not statistically significant"}**
   at the 5% level (p = {pval:.4f}).

3. **Overall:** this analysis does **not** find strong evidence that disaster declarations
   are associated with meaningfully different business-formation activity, once obvious
   confounding factors are accounted for. The raw correlation seen before controlling for
   fixed effects appears to reflect which *kinds* of states have more disasters, rather
   than a genuine month-to-month relationship.
"""
    )
else:
    st.info("Run the full pipeline (`src.preprocessing`, `src.eda`, `src.modeling`) to populate this page.")

st.divider()

st.header("Limitations")
st.markdown(
    """
- **Observational design.** No random assignment of disasters to states — this
  analysis can identify association, not causation.
- **Disaster measure is administrative, not physical.** FEMA's disaster count reflects
  formal federal declarations, which involve a political/administrative process on top
  of physical severity. Two equally damaging events could be declared differently
  depending on state requests and federal review.
- **State-month granularity may be too coarse.** A disaster affecting one county in a
  large state is treated the same as one affecting many counties, since the panel
  doesn't have county-level resolution.
- **No lag structure.** The model only looks at same-month association. If disaster
  effects on business formation show up with a delay (e.g., 3-6 months later, once
  rebuilding begins), this model wouldn't detect it.
- **Coverage boundary.** FEMA's history goes back to 1953, but BFS only starts in 2004,
  so the analysis is necessarily limited to the post-2004 overlap.
- **Single independent variable.** No controls for other time-varying state-level
  factors (state policy changes, local economic shocks) that could confound the
  within-state estimate.
"""
)

st.divider()

st.header("Future Work")
st.markdown(
    """
- **Add a lag structure** — test whether disaster effects on business formation emerge
  with a delay rather than in the same month.
- **Incorporate disaster severity**, not just count — e.g., estimated damage cost or
  declaration type (major disaster vs. emergency), to see if bigger disasters matter more.
- **Disaggregate by industry (NAICS sector)** — construction and retail may respond
  very differently to a disaster than professional services.
- **County-level analysis**, if a compatible county-level BFS-equivalent measure becomes
  available, to increase spatial resolution.
- **Event-study design** — plot business application trends in the months before and
  after a disaster declaration, rather than a single pooled coefficient.
- **Instrument for disaster severity** using purely physical measures (e.g., NOAA wind
  speed, rainfall) to move closer to a causal identification strategy.
"""
)

st.divider()
st.caption(
    "This dashboard presents the results of a fully reproducible pipeline — see the "
    "GitHub repository README for how to re-run every step from raw data to these figures."
)
