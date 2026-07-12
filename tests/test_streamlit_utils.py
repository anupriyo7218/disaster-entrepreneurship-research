"""
Tests for streamlit_app/utils.py.

Streamlit's caching decorators (st.cache_data) work fine when called
directly in a plain pytest process — they just cache per-process, which is
harmless here. These tests confirm the data-loading contract the dashboard
pages depend on: correct types, graceful None on missing files, and that
the real pipeline outputs (once generated) load without error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "streamlit_app"))

from utils import (  # noqa: E402
    load_config,
    load_fema_clean,
    load_final_dataset,
    load_report_text,
    load_table,
    figure_path_if_exists,
)


def test_load_config_returns_dict():
    config = load_config()
    assert isinstance(config, dict)
    assert "bfs" in config
    assert "fema" in config


def test_load_final_dataset_returns_dataframe_with_expected_columns():
    df = load_final_dataset()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert {"state", "year_month", "business_applications", "disaster_count"}.issubset(df.columns)
    assert isinstance(df["year_month"].dtype, pd.PeriodDtype)


def test_load_fema_clean_returns_dataframe_with_incident_type():
    df = load_fema_clean()
    assert df is not None
    assert "incident_type" in df.columns


def test_load_table_returns_none_for_nonexistent_file():
    result = load_table("this_file_does_not_exist_12345.csv")
    assert result is None


def test_load_table_returns_dataframe_for_real_file():
    result = load_table("twfe_regression_table.csv")
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "coefficient" in result.columns


def test_load_report_text_returns_none_for_nonexistent_file():
    assert load_report_text("nonexistent_report_xyz.txt") is None


def test_load_report_text_returns_string_for_real_file():
    text = load_report_text("twfe_interpretation.txt")
    assert text is not None
    assert isinstance(text, str)
    assert "causal" in text.lower()


def test_figure_path_if_exists_returns_none_for_missing_figure():
    assert figure_path_if_exists("nonexistent_figure_xyz.png") is None


def test_figure_path_if_exists_returns_path_for_real_figure():
    path = figure_path_if_exists("twfe_coefficient_plot.png")
    assert path is not None
    assert path.exists()
