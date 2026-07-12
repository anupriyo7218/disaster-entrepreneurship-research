"""
pages/1_Dataset_Explorer.py

Lets a user inspect the actual analysis-ready panel: dimensions, sample
rows, summary stats, missingness, and a per-state drill-down. This is
deliberately the raw merged panel, not a chart — the point is transparency
about what's actually in the data before any analysis is shown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_final_dataset, load_table, missing_pipeline_warning  # noqa: E402

st.set_page_config(page_title="Dataset Explorer", page_icon="🔎", layout="wide")
st.title("Dataset Explorer")

df = load_final_dataset()

if df is None:
    missing_pipeline_warning("Module 2 (final_analysis_dataset.csv)", "python -m src.preprocessing")
    st.stop()

# --- Top-line dimensions -----------------------------------------------
st.header("Dataset Dimensions")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("States", df["state"].nunique())
c3.metric("Earliest month", str(df["year_month"].min()))
c4.metric("Latest month", str(df["year_month"].max()))

st.divider()

# --- State picker used by the rest of the page --------------------------
st.header("Browse by State")
states = sorted(df["state"].unique())
selected_state = st.selectbox("Choose a state", states, index=states.index("CA") if "CA" in states else 0)

state_df = df[df["state"] == selected_state].sort_values("year_month")

st.subheader(f"Sample rows — {selected_state}")
display_df = state_df.copy()
display_df["year_month"] = display_df["year_month"].astype(str)
st.dataframe(display_df.head(20), use_container_width=True)

st.divider()

# --- Summary statistics -----------------------------------------------
st.header("Summary Statistics")
tab1, tab2 = st.tabs(["All states (full panel)", f"{selected_state} only"])
with tab1:
    st.dataframe(df[["business_applications", "disaster_count"]].describe(), use_container_width=True)
with tab2:
    st.dataframe(state_df[["business_applications", "disaster_count"]].describe(), use_container_width=True)

st.divider()

# --- Missing values -----------------------------------------------------
st.header("Missing-Value Summary")
missing = df.isnull().sum()
missing_df = missing[missing > 0].reset_index()
missing_df.columns = ["column", "missing_count"]
if missing_df.empty:
    st.success("No missing values in the final panel (beyond the expected unpublished recent months).")
else:
    st.dataframe(missing_df, use_container_width=True)
    st.caption(
        "Missing business_applications values correspond to months not yet published by the "
        "Census Bureau at the time this data was pulled — not a data quality issue."
    )

st.divider()

# --- Preprocessing summary, if available --------------------------------
st.header("Preprocessing Summary (from Module 2)")
overview = load_table("eda_dataset_overview.csv")
if overview is not None:
    st.dataframe(overview, use_container_width=True, hide_index=True)
else:
    st.info("Run `python -m src.eda` to generate the dataset overview table.")
