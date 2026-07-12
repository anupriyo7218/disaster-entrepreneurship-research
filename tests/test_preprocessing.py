"""
Tests for src/preprocessing.py.

Covers the BFS cleaning, FEMA cleaning, disaster panel construction, and
merge logic using small synthetic dataframes (fast, deterministic) rather
than the full raw files, plus a couple of checks against the real data for
the assumptions the pipeline relies on (e.g. disasterNumber:state 1:1).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config_loader import load_config
from src.preprocessing import (
    PreprocessingError,
    _split_value_and_disclosure_flag,
    build_disaster_panel,
    clean_bfs,
    clean_fema_events,
    merge_datasets,
    validate_bfs_clean,
    validate_fema_clean,
    validate_merge,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


# --- Disclosure code handling ------------------------------------------------

@pytest.mark.parametrize(
    "raw_value,expected_value,expected_flag",
    [
        ("123", 123.0, "valid"),
        ("0", 0.0, "valid"),
        ("D", None, "disclosure_D"),
        ("S", None, "disclosure_S"),
        ("NA", None, "disclosure_NA"),
        (None, None, "missing"),
    ],
)
def test_split_value_and_disclosure_flag(raw_value, expected_value, expected_flag):
    value, flag = _split_value_and_disclosure_flag(raw_value)
    if expected_value is None:
        assert pd.isna(value)
    else:
        assert value == expected_value
    assert flag == expected_flag


# --- BFS cleaning -------------------------------------------------------------

def _make_synthetic_bfs() -> pd.DataFrame:
    """A minimal synthetic BFS raw frame covering the filter/melt logic."""
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    rows = [
        # Kept: state-level, BA_BA, not-seasonally-adjusted, TOTAL
        {"sa": "U", "naics_sector": "TOTAL", "series": "BA_BA", "geo": "CA", "year": 2010,
         **{m: 100 + i for i, m in enumerate(months)}},
        # Dropped: seasonally adjusted
        {"sa": "A", "naics_sector": "TOTAL", "series": "BA_BA", "geo": "CA", "year": 2010,
         **{m: 200 for m in months}},
        # Dropped: US aggregate, not a state
        {"sa": "U", "naics_sector": "TOTAL", "series": "BA_BA", "geo": "US", "year": 2010,
         **{m: 999 for m in months}},
        # Dropped: Puerto Rico, out of scope
        {"sa": "U", "naics_sector": "TOTAL", "series": "BA_BA", "geo": "PR", "year": 2010,
         **{m: 50 for m in months}},
        # Kept row with a disclosure code embedded
        {"sa": "U", "naics_sector": "TOTAL", "series": "BA_BA", "geo": "TX", "year": 2010,
         **{m: (150 if m != "mar" else "D") for m in months}},
    ]
    return pd.DataFrame(rows)


def test_clean_bfs_filters_and_reshapes(config):
    raw = _make_synthetic_bfs()
    clean = clean_bfs(raw, config)

    # Only CA and TX should survive the series/sa/naics/geo filters.
    assert set(clean["state"].unique()) == {"CA", "TX"}
    # 2 states x 12 months = 24 rows.
    assert len(clean) == 24
    assert {"state", "year_month", "business_applications", "disclosure_flag"} == set(clean.columns)


def test_clean_bfs_preserves_disclosure_code(config):
    raw = _make_synthetic_bfs()
    clean = clean_bfs(raw, config)
    march_tx = clean[(clean["state"] == "TX") & (clean["year_month"].astype(str) == "2010-03")]
    assert len(march_tx) == 1
    assert march_tx.iloc[0]["disclosure_flag"] == "disclosure_D"
    assert pd.isna(march_tx.iloc[0]["business_applications"])


def test_clean_bfs_drops_2004_prehistory_months(config):
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    raw = pd.DataFrame([{
        "sa": "U", "naics_sector": "TOTAL", "series": "BA_BA", "geo": "CA", "year": 2004,
        **{m: (None if m in ("jan", "feb", "mar", "apr", "may", "jun") else 100) for m in months},
    }])
    clean = clean_bfs(raw, config)
    # Only Jul-Dec 2004 should remain (6 rows), not the null Jan-Jun rows.
    assert len(clean) == 6
    assert clean["year_month"].astype(str).min() == "2004-07"


def test_validate_bfs_clean_rejects_duplicate_keys(config):
    bad = pd.DataFrame({
        "state": ["CA", "CA"],
        "year_month": pd.PeriodIndex(["2010-01", "2010-01"], freq="M"),
        "business_applications": [100.0, 100.0],
        "disclosure_flag": ["valid", "valid"],
    })
    with pytest.raises(PreprocessingError):
        validate_bfs_clean(bad, config)


def test_validate_bfs_clean_rejects_invalid_state(config):
    bad = pd.DataFrame({
        "state": ["ZZ"],
        "year_month": pd.PeriodIndex(["2010-01"], freq="M"),
        "business_applications": [100.0],
        "disclosure_flag": ["valid"],
    })
    with pytest.raises(PreprocessingError):
        validate_bfs_clean(bad, config)


# --- FEMA cleaning --------------------------------------------------------

def _make_synthetic_fema() -> pd.DataFrame:
    """Synthetic FEMA raw frame simulating the county-level fan-out."""
    return pd.DataFrame([
        # Same disaster, 2 counties in CA -> should collapse to 1 event.
        {"disasterNumber": 1001, "state": "CA", "incidentType": "Flood",
         "declarationType": "DR", "incidentBeginDate": "2015-06-01T00:00:00.000Z"},
        {"disasterNumber": 1001, "state": "CA", "incidentType": "Flood",
         "declarationType": "DR", "incidentBeginDate": "2015-06-01T00:00:00.000Z"},
        # Different disaster, same state -> separate event.
        {"disasterNumber": 1002, "state": "CA", "incidentType": "Fire",
         "declarationType": "DR", "incidentBeginDate": "2015-09-15T00:00:00.000Z"},
        # Out-of-scope territory -> should be dropped.
        {"disasterNumber": 1003, "state": "GU", "incidentType": "Typhoon",
         "declarationType": "DR", "incidentBeginDate": "2015-03-01T00:00:00.000Z"},
    ])


def test_clean_fema_events_collapses_county_fanout(config):
    raw = _make_synthetic_fema()
    clean = clean_fema_events(raw, config)
    # 1001 collapses from 2 rows to 1; 1002 stays as 1; 1003 (GU) dropped.
    assert len(clean) == 2
    assert set(clean["disaster_number"]) == {1001, 1002}
    assert "GU" not in clean["state"].values


def test_clean_fema_events_year_month_correct(config):
    raw = _make_synthetic_fema()
    clean = clean_fema_events(raw, config)
    row = clean[clean["disaster_number"] == 1001].iloc[0]
    assert str(row["year_month"]) == "2015-06"


def test_validate_fema_clean_rejects_duplicate_keys(config):
    bad = pd.DataFrame({
        "disaster_number": [1, 1],
        "state": ["CA", "CA"],
        "year_month": pd.PeriodIndex(["2015-06", "2015-06"], freq="M"),
        "incident_type": ["Flood", "Flood"],
        "declaration_type": ["DR", "DR"],
        "incident_begin_date": pd.to_datetime(["2015-06-01", "2015-06-01"]),
    })
    with pytest.raises(PreprocessingError):
        validate_fema_clean(bad, config)


# --- Disaster panel & merge -------------------------------------------------

def test_build_disaster_panel_counts_distinct_events(config):
    raw = _make_synthetic_fema()
    fema_clean = clean_fema_events(raw, config)
    panel = build_disaster_panel(fema_clean)
    ca_june = panel[(panel["state"] == "CA") & (panel["year_month"].astype(str) == "2015-06")]
    assert ca_june.iloc[0]["disaster_count"] == 1


def test_merge_fills_zero_for_no_disaster_months(config):
    bfs_clean = pd.DataFrame({
        "state": ["CA", "CA"],
        "year_month": pd.PeriodIndex(["2015-06", "2015-07"], freq="M"),
        "business_applications": [500.0, 520.0],
        "disclosure_flag": ["valid", "valid"],
    })
    disaster_panel = pd.DataFrame({
        "state": ["CA"],
        "year_month": pd.PeriodIndex(["2015-06"], freq="M"),
        "disaster_count": [1],
    })
    merged = merge_datasets(bfs_clean, disaster_panel)
    july_row = merged[merged["year_month"].astype(str) == "2015-07"]
    assert july_row.iloc[0]["disaster_count"] == 0
    assert merged["disaster_count"].dtype.kind == "i"  # filled, not float/NaN


def test_validate_merge_passes_on_consistent_data(config):
    bfs_clean = pd.DataFrame({
        "state": ["CA"],
        "year_month": pd.PeriodIndex(["2015-06"], freq="M"),
        "business_applications": [500.0],
        "disclosure_flag": ["valid"],
    })
    disaster_panel = pd.DataFrame({
        "state": ["CA"],
        "year_month": pd.PeriodIndex(["2015-06"], freq="M"),
        "disaster_count": [1],
    })
    merged = merge_datasets(bfs_clean, disaster_panel)
    results = validate_merge(merged, bfs_clean, disaster_panel)
    assert results["duplicate_state_month_keys"] == 0
    assert results["unmatched_fema_state_months"] == 0


# --- Assumptions checked against the real uploaded data -----------------------

def test_real_fema_disaster_number_maps_to_single_state(config):
    """
    Guards the core assumption clean_fema_events() relies on: each
    disasterNumber belongs to exactly one state. If a future data refresh
    breaks this, we want a test failure, not a silently wrong panel.
    """
    from src.data_loader import load_fema_data
    raw_fema = load_fema_data(config)
    states_per_disaster = raw_fema.groupby("disasterNumber")["state"].nunique()
    assert states_per_disaster.max() == 1
