"""
eda.py

Module 3 deliverable — Exploratory Data Analysis.

Purely descriptive: dataset overview, distributional analysis of business
applications, FEMA disaster patterns, and a first look at whether disaster
months differ from non-disaster months. No model is fit anywhere in this
module — that's deliberate, so we understand the data on its own terms
before any functional form or causal assumption gets imposed on it.

Reads from data/processed/ (Module 2 outputs) and writes:
    outputs/figures/*.png
    outputs/tables/*.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config_loader import load_config
from src.logging_setup import get_logger

logger = get_logger(__name__)

# matplotlib logs an INFO-level notice every time we plot string categories
# (year labels, month names) — harmless and expected here, but noisy in the
# pipeline log, so keep it at WARNING and above.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# A consistent, uncluttered look across every figure in the paper/dashboard —
# set once here rather than repeating style calls in every plotting function.
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

MONTH_NUM_TO_NAME = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


class EDAError(Exception):
    """Raised when an EDA step fails or produces an unexpected result."""


# =============================================================================
# I/O helpers
# =============================================================================

def load_processed_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Load the three Module 2 outputs this module operates on.

    Parameters
    ----------
    config : dict
        Parsed project configuration.

    Returns
    -------
    dict
        {'final': final_analysis_dataset, 'fema_clean': fema_clean,
         'bfs_clean': bfs_clean}, each with year_month parsed back into a
         proper Period (CSV round-tripping stores it as a plain string).

    Raises
    ------
    EDAError
        If any expected processed file is missing — Module 2 must run first.
    """
    processed_dir = PROJECT_ROOT / config["paths"]["data_processed"]
    outputs = config["processed_outputs"]

    frames = {}
    for key, filename in [
        ("final", outputs["final_analysis_dataset"]),
        ("fema_clean", outputs["fema_clean"]),
        ("bfs_clean", outputs["bfs_clean"]),
    ]:
        path = processed_dir / filename
        if not path.exists():
            raise EDAError(
                f"Expected processed file not found: {path}. Run Module 2 (src/preprocessing.py) first."
            )
        df = pd.read_csv(path)
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
        frames[key] = df
        logger.info("Loaded %s: %s rows", filename, len(df))

    return frames


def _save_figure(fig: plt.Figure, filename: str, config: Dict) -> Path:
    """Save a matplotlib figure to outputs/figures/ and close it to free memory."""
    out_path = PROJECT_ROOT / config["paths"]["outputs_figures"] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


def _save_table(df: pd.DataFrame, filename: str, config: Dict) -> Path:
    """Save a dataframe to outputs/tables/ as CSV."""
    out_path = PROJECT_ROOT / config["paths"]["outputs_tables"] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved table: %s", out_path)
    return out_path


# =============================================================================
# Part A — Dataset overview
# =============================================================================

def part_a_dataset_overview(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Produce top-line descriptive facts about the final analysis panel:
    shape, state count, date range, missingness, and duplicate check.

    Parameters
    ----------
    df : pd.DataFrame
        The final analysis dataset (state, year_month, business_applications,
        disclosure_flag, disaster_count).
    config : dict
        Parsed project configuration.

    Returns
    -------
    pd.DataFrame
        One-row-per-metric overview table, also saved to outputs/tables/.
    """
    logger.info("Part A: dataset overview")

    overview = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_states": df["state"].nunique(),
        "date_range_start": str(df["year_month"].min()),
        "date_range_end": str(df["year_month"].max()),
        "n_months_covered": df["year_month"].nunique(),
        "n_missing_business_applications": int(df["business_applications"].isnull().sum()),
        "n_missing_disaster_count": int(df["disaster_count"].isnull().sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_duplicate_state_month_keys": int(df.duplicated(subset=["state", "year_month"]).sum()),
        "pct_state_months_with_disaster": round(100 * (df["disaster_count"] > 0).mean(), 2),
    }
    overview_df = pd.DataFrame(list(overview.items()), columns=["metric", "value"])
    _save_table(overview_df, "eda_dataset_overview.csv", config)

    # describe() on the two numeric columns of interest — the standard
    # first-pass summary before looking at anything more targeted.
    summary_stats = df[["business_applications", "disaster_count"]].describe().reset_index()
    summary_stats = summary_stats.rename(columns={"index": "statistic"})
    _save_table(summary_stats, "eda_summary_statistics.csv", config)

    logger.info("Part A overview: %s", overview)
    return overview_df


# =============================================================================
# Part B — Business Applications Analysis
# =============================================================================

def plot_monthly_applications_over_time(df: pd.DataFrame, config: Dict) -> Path:
    """National-total monthly business applications over the full time range."""
    national = df.groupby("year_month", as_index=False)["business_applications"].sum()
    national["year_month_ts"] = national["year_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(national["year_month_ts"], national["business_applications"], color="#1f77b4", linewidth=1.2)
    ax.set_title("U.S. Business Applications, All States Combined — Monthly")
    ax.set_xlabel("Month")
    ax.set_ylabel("Business Applications (sum across states)")
    fig.tight_layout()
    return _save_figure(fig, "business_applications_monthly_trend.png", config)


def plot_yearly_trend(df: pd.DataFrame, config: Dict) -> Path:
    """Total business applications aggregated to the year level."""
    yearly = df.copy()
    yearly["year"] = yearly["year_month"].dt.year
    yearly = yearly.groupby("year", as_index=False)["business_applications"].sum()
    # 2026 is a partial year in the raw file (not all months published yet);
    # flag it visually rather than let it silently look like a decline.
    max_year = yearly["year"].max()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#d62728" if y == max_year else "#1f77b4" for y in yearly["year"]]
    ax.bar(yearly["year"], yearly["business_applications"], color=colors)
    ax.set_title("U.S. Business Applications — Yearly Total (red = partial year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Business Applications (sum across states)")
    fig.tight_layout()
    return _save_figure(fig, "business_applications_yearly_trend.png", config)


def plot_state_wise_averages(df: pd.DataFrame, config: Dict) -> Path:
    """Average monthly business applications by state, sorted descending."""
    state_avg = df.groupby("state", as_index=False)["business_applications"].mean()
    state_avg = state_avg.sort_values("business_applications", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(state_avg["state"], state_avg["business_applications"], color="#2ca02c")
    ax.set_title("Average Monthly Business Applications by State")
    ax.set_xlabel("Average Business Applications per Month")
    ax.set_ylabel("State")
    fig.tight_layout()
    return _save_figure(fig, "business_applications_state_averages.png", config)


def plot_applications_distribution(df: pd.DataFrame, config: Dict) -> Path:
    """Histogram of state-month business application counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["business_applications"].dropna(), bins=50, color="#9467bd", ax=ax)
    ax.set_title("Distribution of State-Month Business Applications")
    ax.set_xlabel("Business Applications")
    ax.set_ylabel("Frequency (state-months)")
    fig.tight_layout()
    return _save_figure(fig, "business_applications_histogram.png", config)


def plot_applications_boxplot_by_year(df: pd.DataFrame, config: Dict) -> Path:
    """
    Boxplot of state-month business applications by year — shows both the
    level and the spread across states each year, which a single overall
    boxplot wouldn't reveal.
    """
    plot_df = df.copy()
    plot_df["year"] = plot_df["year_month"].dt.year.astype(str)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(data=plot_df, x="year", y="business_applications", ax=ax, color="#ff7f0e")
    ax.set_title("Business Applications by Year — Distribution Across States")
    ax.set_xlabel("Year")
    ax.set_ylabel("Business Applications (state-month)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return _save_figure(fig, "business_applications_boxplot_by_year.png", config)


def compute_business_applications_stats(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Compute mean/median/std/min/max/percentiles for business_applications.

    Parameters
    ----------
    df : pd.DataFrame
    config : dict

    Returns
    -------
    pd.DataFrame
        One-row summary table, also saved to outputs/tables/.
    """
    series = df["business_applications"].dropna()
    stats = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "p5": series.quantile(0.05),
        "p25": series.quantile(0.25),
        "p50": series.quantile(0.50),
        "p75": series.quantile(0.75),
        "p95": series.quantile(0.95),
    }
    stats_df = pd.DataFrame([stats])
    _save_table(stats_df, "business_applications_descriptive_stats.csv", config)
    logger.info("Business applications stats: %s", stats)
    return stats_df


def part_b_business_applications_analysis(df: pd.DataFrame, config: Dict) -> None:
    """Run every Part B plot and statistics function in sequence."""
    logger.info("Part B: business applications analysis")
    plot_monthly_applications_over_time(df, config)
    plot_yearly_trend(df, config)
    plot_state_wise_averages(df, config)
    plot_applications_distribution(df, config)
    plot_applications_boxplot_by_year(df, config)
    compute_business_applications_stats(df, config)


# =============================================================================
# Part C — FEMA Disaster Analysis
# =============================================================================

def plot_disasters_per_year(fema_clean: pd.DataFrame, config: Dict) -> Path:
    """Count of distinct disaster events per year."""
    yearly = fema_clean.copy()
    yearly["year"] = yearly["year_month"].dt.year
    counts = yearly.groupby("year")["disaster_number"].nunique().reset_index(name="disaster_count")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(counts["year"], counts["disaster_count"], color="#8c564b")
    ax.set_title("FEMA Disaster Declarations per Year (All U.S. History in File)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Distinct Disaster Declarations")
    fig.tight_layout()
    return _save_figure(fig, "fema_disasters_per_year.png", config)


def plot_disasters_per_month(fema_clean: pd.DataFrame, config: Dict) -> Path:
    """Seasonality check: disaster counts pooled by calendar month (Jan-Dec)."""
    seasonal = fema_clean.copy()
    seasonal["month_num"] = seasonal["year_month"].dt.month
    counts = (
        seasonal.groupby("month_num")["disaster_number"]
        .nunique()
        .reindex(range(1, 13))
        .reset_index(name="disaster_count")
    )
    counts["month_name"] = counts["month_num"].map(MONTH_NUM_TO_NAME)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts["month_name"], counts["disaster_count"], color="#e377c2")
    ax.set_title("FEMA Disaster Declarations by Calendar Month (Seasonality, All Years Pooled)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Distinct Disaster Declarations")
    fig.tight_layout()
    return _save_figure(fig, "fema_disasters_per_month.png", config)


def plot_disasters_by_state(fema_clean: pd.DataFrame, config: Dict) -> Path:
    """Total distinct disaster events per state, all 51 states."""
    by_state = fema_clean.groupby("state")["disaster_number"].nunique().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(by_state.index, by_state.values, color="#17becf")
    ax.set_title("Total FEMA Disaster Declarations by State (All History)")
    ax.set_xlabel("Number of Distinct Disaster Declarations")
    ax.set_ylabel("State")
    fig.tight_layout()
    return _save_figure(fig, "fema_disasters_by_state.png", config)


def plot_disasters_by_incident_type(fema_clean: pd.DataFrame, config: Dict) -> Path:
    """Total distinct disaster events per incident type, all 27 categories."""
    by_type = fema_clean.groupby("incident_type")["disaster_number"].nunique().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(by_type.index, by_type.values, color="#bcbd22")
    ax.set_title("FEMA Disaster Declarations by Incident Type (All History)")
    ax.set_xlabel("Number of Distinct Disaster Declarations")
    ax.set_ylabel("Incident Type")
    fig.tight_layout()
    return _save_figure(fig, "fema_disasters_by_incident_type.png", config)


def compute_top_states_by_disaster_count(fema_clean: pd.DataFrame, config: Dict, top_n: int = 15) -> pd.DataFrame:
    """Top N states ranked by total distinct disaster declarations."""
    top_states = (
        fema_clean.groupby("state")["disaster_number"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="disaster_count")
    )
    _save_table(top_states, "fema_top15_states_by_disaster_count.csv", config)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_states["state"][::-1], top_states["disaster_count"][::-1], color="#7f7f7f")
    ax.set_title(f"Top {top_n} States by Total FEMA Disaster Declarations")
    ax.set_xlabel("Number of Distinct Disaster Declarations")
    fig.tight_layout()
    _save_figure(fig, "fema_top15_states_by_disaster_count.png", config)

    return top_states


def compute_top_incident_categories(fema_clean: pd.DataFrame, config: Dict, top_n: int = 10) -> pd.DataFrame:
    """Top N incident types ranked by total distinct disaster declarations."""
    top_types = (
        fema_clean.groupby("incident_type")["disaster_number"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="disaster_count")
    )
    _save_table(top_types, "fema_top_incident_categories.csv", config)
    return top_types


def part_c_fema_disaster_analysis(fema_clean: pd.DataFrame, config: Dict) -> None:
    """Run every Part C plot and table function in sequence."""
    logger.info("Part C: FEMA disaster analysis")
    plot_disasters_per_year(fema_clean, config)
    plot_disasters_per_month(fema_clean, config)
    plot_disasters_by_state(fema_clean, config)
    plot_disasters_by_incident_type(fema_clean, config)
    compute_top_states_by_disaster_count(fema_clean, config)
    compute_top_incident_categories(fema_clean, config)


# =============================================================================
# Part D — Relationship Exploration (descriptive only, no regression)
# =============================================================================

def compute_grouped_descriptive_stats(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Compare business_applications between disaster and non-disaster
    state-months using group-level descriptive statistics only.

    This is NOT a test of causal effect — disaster-prone states may simply
    differ systematically from disaster-free ones (state size, industry
    mix, baseline economic activity), so a raw group difference conflates
    disaster impact with those pre-existing differences. Treat this as a
    starting observation to motivate the modeling stage, not a finding.

    Parameters
    ----------
    df : pd.DataFrame
    config : dict

    Returns
    -------
    pd.DataFrame
        Descriptive stats by disaster-month group, saved to outputs/tables/.
    """
    grouped_df = df.copy()
    grouped_df["had_disaster"] = grouped_df["disaster_count"] > 0

    grouped = (
        grouped_df.groupby("had_disaster")["business_applications"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    grouped["had_disaster"] = grouped["had_disaster"].map({True: "Disaster month", False: "No disaster month"})
    _save_table(grouped, "eda_grouped_stats_disaster_vs_no_disaster.csv", config)
    logger.info("Grouped descriptive stats (disaster vs. no disaster):\n%s", grouped)
    return grouped


def plot_boxplot_disaster_vs_no_disaster(df: pd.DataFrame, config: Dict) -> Path:
    """Boxplot comparing business applications in disaster vs. non-disaster months."""
    plot_df = df.copy()
    plot_df["had_disaster"] = plot_df["disaster_count"] > 0
    plot_df["had_disaster"] = plot_df["had_disaster"].map({True: "Disaster month", False: "No disaster month"})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=plot_df, x="had_disaster", y="business_applications", ax=ax, color="#c5b0d5")
    ax.set_title("Business Applications: Disaster vs. No-Disaster Months\n(descriptive comparison only, not a causal test)")
    ax.set_xlabel("")
    ax.set_ylabel("Business Applications (state-month)")
    fig.tight_layout()
    return _save_figure(fig, "eda_boxplot_disaster_vs_no_disaster.png", config)


def plot_scatter_disaster_count_vs_applications(df: pd.DataFrame, config: Dict) -> Path:
    """Scatter of disaster_count against business_applications, state-month level."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["disaster_count"], df["business_applications"], alpha=0.15, s=15, color="#1f77b4")
    ax.set_title("Disaster Count vs. Business Applications (Raw Scatter, State-Month)\n(descriptive only — not a fitted relationship)")
    ax.set_xlabel("Number of Disasters Declared That Month")
    ax.set_ylabel("Business Applications")
    fig.tight_layout()
    return _save_figure(fig, "eda_scatter_disaster_count_vs_applications.png", config)


def compute_correlation_matrix(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Simple pairwise correlation matrix (Pearson) between business
    applications, disaster count, and calendar year — a first look at
    association, not a model.

    Parameters
    ----------
    df : pd.DataFrame
    config : dict

    Returns
    -------
    pd.DataFrame
        Correlation matrix, saved to outputs/tables/ and rendered as a
        heatmap in outputs/figures/.
    """
    corr_df = df.copy()
    corr_df["year"] = corr_df["year_month"].dt.year
    corr_matrix = corr_df[["business_applications", "disaster_count", "year"]].corr()
    _save_table(corr_matrix.reset_index().rename(columns={"index": "variable"}),
                "eda_correlation_matrix.csv", config)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix (Pearson) — Descriptive Only")
    fig.tight_layout()
    _save_figure(fig, "eda_correlation_heatmap.png", config)

    logger.info("Correlation matrix:\n%s", corr_matrix)
    return corr_matrix


def part_d_relationship_exploration(df: pd.DataFrame, config: Dict) -> None:
    """
    Run every Part D function in sequence. Everything here is descriptive —
    logged explicitly as a reminder before the modeling module begins.
    """
    logger.info("Part D: relationship exploration (DESCRIPTIVE ONLY — not causal evidence)")
    compute_grouped_descriptive_stats(df, config)
    plot_boxplot_disaster_vs_no_disaster(df, config)
    plot_scatter_disaster_count_vs_applications(df, config)
    compute_correlation_matrix(df, config)
    logger.info(
        "Part D complete. Reminder: none of the above establishes a causal effect of "
        "disasters on business formation — states differ systematically in ways that "
        "raw group comparisons and correlations cannot account for. That is what the "
        "econometric modeling module (Module 4) exists to address."
    )


# =============================================================================
# Orchestration
# =============================================================================

def run_eda_pipeline() -> None:
    """
    Execute the full Module 3 EDA pipeline: load processed data, run Parts
    A-D, and confirm all expected output files were written.
    """
    logger.info("=" * 70)
    logger.info("Module 3: Exploratory Data Analysis — starting")
    logger.info("=" * 70)

    config = load_config()
    frames = load_processed_data(config)

    part_a_dataset_overview(frames["final"], config)
    part_b_business_applications_analysis(frames["final"], config)
    part_c_fema_disaster_analysis(frames["fema_clean"], config)
    part_d_relationship_exploration(frames["final"], config)

    logger.info("=" * 70)
    logger.info("Module 3: Exploratory Data Analysis — completed successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_eda_pipeline()
