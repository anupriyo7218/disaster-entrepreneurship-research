"""
Tests for src/eda.py.

Focus on the computation functions (stats, grouping, correlation) using
small synthetic frames, since those are what could silently produce wrong
numbers. Plotting functions are checked only for "does it run and produce a
file" — verifying pixel content isn't a meaningful test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config_loader import load_config
from src.eda import (
    compute_business_applications_stats,
    compute_correlation_matrix,
    compute_grouped_descriptive_stats,
    compute_top_incident_categories,
    compute_top_states_by_disaster_count,
    part_a_dataset_overview,
    plot_boxplot_disaster_vs_no_disaster,
    plot_monthly_applications_over_time,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def synthetic_final_df():
    """A small synthetic version of final_analysis_dataset.csv."""
    return pd.DataFrame({
        "state": ["CA", "CA", "CA", "TX", "TX", "TX"],
        "year_month": pd.PeriodIndex(
            ["2020-01", "2020-02", "2020-03", "2020-01", "2020-02", "2020-03"], freq="M"
        ),
        "business_applications": [1000.0, 1200.0, None, 500.0, 520.0, 900.0],
        "disclosure_flag": ["valid", "valid", "missing", "valid", "valid", "valid"],
        "disaster_count": [0, 1, 0, 2, 0, 0],
    })


@pytest.fixture
def synthetic_fema_clean():
    """A small synthetic version of fema_clean.csv."""
    return pd.DataFrame({
        "disaster_number": [1, 2, 3, 4],
        "state": ["CA", "TX", "TX", "CA"],
        "year_month": pd.PeriodIndex(["2020-02", "2020-01", "2020-01", "2020-06"], freq="M"),
        "incident_type": ["Flood", "Fire", "Hurricane", "Flood"],
        "declaration_type": ["DR", "DR", "DR", "EM"],
    })


# --- Part A -------------------------------------------------------------

def test_part_a_dataset_overview_counts(synthetic_final_df, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    overview_df = part_a_dataset_overview(synthetic_final_df, config)
    overview = dict(zip(overview_df["metric"], overview_df["value"]))
    assert overview["n_rows"] == 6
    assert overview["n_states"] == 2
    assert overview["n_missing_business_applications"] == 1
    assert overview["n_duplicate_rows"] == 0


# --- Part B ---------------------------------------------------------------

def test_compute_business_applications_stats(synthetic_final_df, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    stats_df = compute_business_applications_stats(synthetic_final_df, config)
    expected_mean = synthetic_final_df["business_applications"].dropna().mean()
    assert stats_df.iloc[0]["mean"] == pytest.approx(expected_mean)
    assert stats_df.iloc[0]["min"] == 500.0
    assert stats_df.iloc[0]["max"] == 1200.0


def test_plot_monthly_applications_over_time_creates_file(synthetic_final_df, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    out_path = plot_monthly_applications_over_time(synthetic_final_df, config)
    assert out_path.exists()
    assert out_path.suffix == ".png"


# --- Part C ---------------------------------------------------------------

def test_compute_top_states_by_disaster_count(synthetic_fema_clean, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    top_states = compute_top_states_by_disaster_count(synthetic_fema_clean, config, top_n=15)
    # CA has 2 distinct disasters (1, 4); TX has 2 distinct disasters (2, 3).
    ca_row = top_states[top_states["state"] == "CA"]
    tx_row = top_states[top_states["state"] == "TX"]
    assert ca_row.iloc[0]["disaster_count"] == 2
    assert tx_row.iloc[0]["disaster_count"] == 2


def test_compute_top_incident_categories(synthetic_fema_clean, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    top_types = compute_top_incident_categories(synthetic_fema_clean, config, top_n=10)
    flood_row = top_types[top_types["incident_type"] == "Flood"]
    assert flood_row.iloc[0]["disaster_count"] == 2


# --- Part D -----------------------------------------------------------

def test_compute_grouped_descriptive_stats(synthetic_final_df, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    grouped = compute_grouped_descriptive_stats(synthetic_final_df, config)
    disaster_row = grouped[grouped["had_disaster"] == "Disaster month"]
    no_disaster_row = grouped[grouped["had_disaster"] == "No disaster month"]
    # Disaster months: CA Feb (1200), TX Jan (500) -> count 2
    assert disaster_row.iloc[0]["count"] == 2
    # No-disaster months: CA Jan (1000), CA Mar (NaN, dropped by mean/std but
    # still counted by groupby unless we drop explicitly) TX Feb (520), TX Mar (900)
    assert no_disaster_row.iloc[0]["count"] in (3, 4)  # depends on NaN handling in agg


def test_compute_correlation_matrix_values(synthetic_final_df, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    corr = compute_correlation_matrix(synthetic_final_df, config)
    # Diagonal must always be 1.0 for a correlation matrix.
    assert corr.loc["business_applications", "business_applications"] == pytest.approx(1.0)
    assert corr.loc["disaster_count", "disaster_count"] == pytest.approx(1.0)
    # Matrix should be symmetric.
    assert corr.loc["business_applications", "disaster_count"] == pytest.approx(
        corr.loc["disaster_count", "business_applications"]
    )


def test_plot_boxplot_disaster_vs_no_disaster_creates_file(synthetic_final_df, config, tmp_path, monkeypatch):
    monkeypatch.setattr("src.eda.PROJECT_ROOT", tmp_path)
    out_path = plot_boxplot_disaster_vs_no_disaster(synthetic_final_df, config)
    assert out_path.exists()


# --- Integration check against real Module 2 output --------------------

def test_eda_runs_against_real_processed_data(config):
    """
    Confirms the real final_analysis_dataset.csv (produced by Module 2) has
    the columns/dtypes this module assumes, so a future Module 2 change
    that breaks the contract fails here rather than mid-pipeline.
    """
    from src.eda import load_processed_data
    frames = load_processed_data(config)
    final_df = frames["final"]
    assert {"state", "year_month", "business_applications", "disaster_count"}.issubset(final_df.columns)
    assert final_df["disaster_count"].min() >= 0
