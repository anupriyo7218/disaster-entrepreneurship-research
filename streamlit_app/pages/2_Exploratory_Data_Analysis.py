"""
pages/2_Exploratory_Data_Analysis.py

Interactive versions of the Module 3 EDA plots — state selection, a time
range slider, and a disaster/no-disaster comparison — using Plotly instead
of the static Matplotlib figures so a user can actually zoom, hover, and
filter rather than just look at a fixed PNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_fema_clean, load_final_dataset, missing_pipeline_warning  # noqa: E402

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📈", layout="wide")
st.title("Exploratory Data Analysis")
st.caption("Descriptive only — no model is fit on this page.")

df = load_final_dataset()
fema_clean = load_fema_clean()

if df is None:
    missing_pipeline_warning("Module 2 (final_analysis_dataset.csv)", "python -m src.preprocessing")
    st.stop()

# --- Shared filters used across every chart on this page -----------------
st.sidebar.header("Filters")
states = sorted(df["state"].unique())
selected_states = st.sidebar.multiselect("States to include", states, default=["CA", "TX", "FL", "NY"])

min_year = int(df["year_month"].dt.year.min())
max_year = int(df["year_month"].dt.year.max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

filtered = df[
    (df["year_month"].dt.year >= year_range[0]) & (df["year_month"].dt.year <= year_range[1])
].copy()
filtered["year_month_ts"] = filtered["year_month"].dt.to_timestamp()

state_filtered = filtered[filtered["state"].isin(selected_states)] if selected_states else filtered

# --- Business applications over time, by selected states -----------------
st.header("Business Applications Over Time")
if selected_states:
    fig = px.line(
        state_filtered, x="year_month_ts", y="business_applications", color="state",
        labels={"year_month_ts": "Month", "business_applications": "Business Applications"},
        title="Monthly Business Applications by State",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one state in the sidebar to see this chart.")

# --- National total trend (unaffected by state filter, uses full range) --
st.header("National Total Trend")
national = filtered.groupby("year_month_ts", as_index=False)["business_applications"].sum()
fig_national = px.line(
    national, x="year_month_ts", y="business_applications",
    labels={"year_month_ts": "Month", "business_applications": "Total Business Applications"},
    title="All States Combined",
)
st.plotly_chart(fig_national, use_container_width=True)

st.divider()

# --- Disaster counts -----------------------------------------------------
st.header("Disaster Counts")
col1, col2 = st.columns(2)

with col1:
    st.subheader("By State")
    disaster_by_state = (
        filtered.groupby("state", as_index=False)["disaster_count"].sum()
        .sort_values("disaster_count", ascending=False)
    )
    fig_state = px.bar(
        disaster_by_state.head(20), x="state", y="disaster_count",
        title="Top 20 States by Total Disaster Count (selected year range)",
        labels={"disaster_count": "Total Disasters"},
    )
    st.plotly_chart(fig_state, use_container_width=True)

with col2:
    st.subheader("By Incident Type")
    if fema_clean is not None:
        type_filtered = fema_clean[
            (fema_clean["year_month"].dt.year >= year_range[0])
            & (fema_clean["year_month"].dt.year <= year_range[1])
        ]
        by_type = (
            type_filtered.groupby("incident_type")["disaster_number"]
            .nunique().reset_index(name="count")
            .sort_values("count", ascending=False).head(10)
        )
        fig_type = px.bar(
            by_type, x="incident_type", y="count",
            title="Top 10 Incident Types (selected year range)",
            labels={"count": "Distinct Disasters", "incident_type": "Type"},
        )
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("Run `python -m src.preprocessing` to generate fema_clean.csv for this chart.")

st.divider()

# --- Disaster vs. no-disaster comparison ---------------------------------
st.header("Business Applications: Disaster vs. No-Disaster Months")
plot_df = filtered.copy()
plot_df["had_disaster"] = plot_df["disaster_count"] > 0
plot_df["had_disaster"] = plot_df["had_disaster"].map({True: "Disaster month", False: "No disaster month"})

fig_box = px.box(
    plot_df, x="had_disaster", y="business_applications",
    title="Descriptive comparison only — not a statistical test",
    labels={"had_disaster": "", "business_applications": "Business Applications"},
)
st.plotly_chart(fig_box, use_container_width=True)

grouped_stats = plot_df.groupby("had_disaster")["business_applications"].agg(
    ["count", "mean", "median", "std"]
).round(1)
st.dataframe(grouped_stats, use_container_width=True)

st.caption(
    "⚠️ This is a raw, unadjusted comparison. It does not account for the fact that large, "
    "economically active states also tend to have more disasters. See the Regression Results "
    "page for an analysis that controls for this."
)
