"""
Tests for src/data_loader.py.

These tests validate the raw-load step only: that files load, that
structural validation catches obviously broken input, and that no
preprocessing side effects occur. Business-logic / cleaning tests belong
with the future preprocessing module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config_loader import load_config
from src.data_loader import (
    DataLoadError,
    load_bfs_data,
    load_fema_data,
    validate_datasets_loaded,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_config_loads(config):
    assert "bfs" in config
    assert "fema" in config
    assert config["bfs"]["raw_filename"] == "bfs_monthly.csv"
    assert config["fema"]["raw_filename"] == "DisasterDeclarationsSummaries.csv"


def test_load_bfs_data_returns_dataframe(config):
    df = load_bfs_data(config)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"sa", "naics_sector", "series", "geo", "year"}.issubset(df.columns)


def test_load_fema_data_returns_dataframe(config):
    df = load_fema_data(config)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"disasterNumber", "state", "incidentType", "incidentBeginDate"}.issubset(
        df.columns
    )


def test_load_bfs_data_missing_file_raises(config, tmp_path):
    bad_config = {**config, "paths": {**config["paths"], "data_raw": str(tmp_path)}}
    with pytest.raises(DataLoadError):
        load_bfs_data(bad_config)


def test_validate_datasets_loaded_passes_on_real_data(config):
    bfs_df = load_bfs_data(config)
    fema_df = load_fema_data(config)
    # Should not raise
    validate_datasets_loaded(bfs_df, fema_df)


def test_validate_datasets_loaded_rejects_empty_dataframe():
    empty_df = pd.DataFrame()
    valid_df = pd.DataFrame(
        {"disasterNumber": [1], "state": ["CA"], "incidentType": ["Flood"],
         "incidentBeginDate": ["2020-01-01"]}
    )
    with pytest.raises(DataLoadError):
        validate_datasets_loaded(empty_df, valid_df)


def test_validate_datasets_loaded_rejects_missing_columns():
    bfs_missing_cols = pd.DataFrame({"sa": ["U"], "year": [2020]})  # missing geo, series, naics_sector
    fema_ok = pd.DataFrame(
        {"disasterNumber": [1], "state": ["CA"], "incidentType": ["Flood"],
         "incidentBeginDate": ["2020-01-01"]}
    )
    with pytest.raises(DataLoadError):
        validate_datasets_loaded(bfs_missing_cols, fema_ok)
