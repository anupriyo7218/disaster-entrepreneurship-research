"""
preprocessing.py

Module 2 deliverable.

Transforms the two raw source datasets into a single tidy state-month
analysis panel:

    raw BFS (wide, all series/sa/naics/geo)  --> bfs_clean.csv
    raw FEMA (county-level declaration rows) --> fema_clean.csv, disaster_panel.csv
    bfs_clean + disaster_panel               --> final_analysis_dataset.csv

Design note: cleaning logic is kept separate from data_loader.py (Module 1)
on purpose. data_loader.py answers "can we trust the raw files load
correctly?"; this module answers "how do we turn them into an analysis-ready
panel?". Keeping the raw load side-effect-free means every transformation
below can be re-run against a known-good starting point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.data_loader import load_bfs_data, load_fema_data
from src.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# BFS stores one column per calendar month; this is the only place that
# mapping needs to live, so melting and date construction stay consistent.
MONTH_ABBR_TO_NUM: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Per the BFS data dictionary: D = suppressed (folded into higher totals),
# S = fails publication quality standards, NA = not available. All three
# are non-numeric placeholders sitting inside otherwise-numeric columns.
BFS_DISCLOSURE_CODES = {"D", "S", "NA"}


class PreprocessingError(Exception):
    """Raised when a preprocessing step fails or produces an invalid result."""


# =============================================================================
# PART A — BFS preprocessing
# =============================================================================

def _split_value_and_disclosure_flag(raw_value) -> Tuple[float, str]:
    """
    Convert a single raw BFS month-cell into (numeric_value, disclosure_flag).

    The BFS month columns mix true numeric counts with disclosure codes
    (D/S/NA) and true missing cells (pre-July-2004 / not-yet-published
    months). We need to tell these apart rather than silently coercing
    everything to NaN, since "suppressed for disclosure" and "doesn't exist
    yet" have different implications for downstream analysis.

    Parameters
    ----------
    raw_value : Any
        A single cell from a jan..dec column (string, float, or NaN).

    Returns
    -------
    (float, str)
        Numeric value (NaN if not resolvable) and one of:
        "valid", "disclosure_D", "disclosure_S", "disclosure_NA", "missing".
    """
    if pd.isna(raw_value):
        return np.nan, "missing"

    raw_str = str(raw_value).strip()
    if raw_str in BFS_DISCLOSURE_CODES:
        return np.nan, f"disclosure_{raw_str}"

    try:
        return float(raw_str), "valid"
    except ValueError:
        # Anything else unexpected — don't silently drop it, surface it.
        logger.warning("Unrecognized BFS cell value encountered: %r", raw_value)
        return np.nan, "unrecognized"


def clean_bfs(raw_bfs: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter, reshape, and tidy the raw BFS dataset into a state-month panel.

    Steps: filter to BA_BA / not-seasonally-adjusted / TOTAL NAICS, restrict
    geography to the 50 states + DC, resolve disclosure codes, and melt from
    wide (jan..dec columns) to long (one row per state-month).

    Parameters
    ----------
    raw_bfs : pd.DataFrame
        Unmodified output of load_bfs_data().
    config : dict
        Parsed project configuration.

    Returns
    -------
    pd.DataFrame
        Columns: state, year_month (period[M]), business_applications (float),
        disclosure_flag (str).

    Assumptions
    ----------
    - Exactly one row exists per (sa, naics_sector, series, geo, year) in the
      raw file (confirmed in Module 0 — zero duplicate keys), so filtering
      before melting cannot introduce row-level ambiguity.
    - "Missing" (pre-collection months / not-yet-published current months)
      and "disclosure-suppressed" are both ultimately NaN in
      business_applications, but are logged/flagged differently since they
      have different substantive meaning.
    """
    bfs_cfg = config["bfs"]
    valid_states = set(config["geography"]["valid_state_codes"])

    logger.info("BFS: starting from %s raw rows", len(raw_bfs))

    # --- Filter to the single series/adjustment/industry combination we want.
    # Doing this before melting means we reshape 51 states x ~23 years instead
    # of the full 36k-row file — cheaper and removes any chance of a NAICS
    # sub-sector or seasonally-adjusted row leaking into the panel.
    filtered = raw_bfs[
        (raw_bfs["series"] == bfs_cfg["filters"]["series"])
        & (raw_bfs["sa"] == bfs_cfg["filters"]["seasonal_adjustment"])
        & (raw_bfs["naics_sector"] == bfs_cfg["filters"]["naics_sector"])
    ].copy()
    logger.info(
        "BFS: %s rows remain after series=%s, sa=%s, naics_sector=%s filter",
        len(filtered), bfs_cfg["filters"]["series"],
        bfs_cfg["filters"]["seasonal_adjustment"], bfs_cfg["filters"]["naics_sector"],
    )

    # --- Restrict geography to 50 states + DC. This drops US/region
    # aggregates (which would double-count if left in) and Puerto Rico /
    # territories (out of scope per the finalized project geography).
    n_before_geo = len(filtered)
    filtered = filtered[filtered["geo"].isin(valid_states)].copy()
    n_dropped_geo = n_before_geo - len(filtered)
    logger.info(
        "BFS: dropped %s non-state rows (US total, regions, PR, territories); %s rows remain",
        n_dropped_geo, len(filtered),
    )

    # --- Melt wide (jan..dec) into long, resolving disclosure codes per-cell
    # rather than column-wise, since D/S can appear in any month/year combo.
    month_cols = bfs_cfg["month_columns"]
    melted = filtered.melt(
        id_vars=["geo", "year"],
        value_vars=month_cols,
        var_name="month_abbr",
        value_name="raw_value",
    )

    resolved = melted["raw_value"].apply(_split_value_and_disclosure_flag)
    melted["business_applications"] = resolved.apply(lambda t: t[0])
    melted["disclosure_flag"] = resolved.apply(lambda t: t[1])

    disclosure_counts = melted["disclosure_flag"].value_counts().to_dict()
    logger.info("BFS: disclosure flag breakdown: %s", disclosure_counts)

    # --- Build year_month as a proper Period, not a string, so month
    # arithmetic (lags, joins with FEMA) works without re-parsing later.
    melted["month_num"] = melted["month_abbr"].map(MONTH_ABBR_TO_NUM)
    # Build via a timestamp first, then convert to Period — more portable
    # across pandas versions than constructing a PeriodIndex directly from
    # separate year/month arrays.
    melted["year_month"] = pd.to_datetime(
        dict(year=melted["year"], month=melted["month_num"], day=1)
    ).dt.to_period("M")

    clean = (
        melted.rename(columns={"geo": "state"})
        .loc[:, ["state", "year_month", "business_applications", "disclosure_flag"]]
        .sort_values(["state", "year_month"])
        .reset_index(drop=True)
    )

    # BFS's monthly collection starts July 2004; the raw file still carries a
    # 2004 row with Jan-Jun as empty cells. Those rows are structurally
    # meaningless (the month didn't exist in the data collection), so we drop
    # them here rather than carrying them into the panel as "missing".
    n_before_prehistory = len(clean)
    clean = clean[~((clean["year_month"].dt.year == 2004) & (clean["year_month"].dt.month < 7))]
    logger.info(
        "BFS: dropped %s pre-collection rows (Jan-Jun 2004, before BFS monthly collection began)",
        n_before_prehistory - len(clean),
    )

    logger.info("BFS: cleaning complete — %s rows, %s states", len(clean), clean["state"].nunique())
    return clean


def validate_bfs_clean(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Run structural validation checks on the cleaned BFS panel and log results.

    Parameters
    ----------
    df : pd.DataFrame
        Output of clean_bfs().
    config : dict
        Parsed project configuration.

    Returns
    -------
    dict
        Validation results (counts of issues found), for inclusion in the
        preprocessing summary.

    Raises
    ------
    PreprocessingError
        If duplicate (state, year_month) keys or invalid state codes are found
        — these indicate a bug in clean_bfs(), not an expected data quality
        quirk, so we fail loudly rather than silently continuing.
    """
    valid_states = set(config["geography"]["valid_state_codes"])
    results: Dict = {}

    duplicate_keys = df.duplicated(subset=["state", "year_month"]).sum()
    results["duplicate_state_month_keys"] = int(duplicate_keys)
    if duplicate_keys > 0:
        raise PreprocessingError(
            f"BFS clean panel has {duplicate_keys} duplicate (state, year_month) rows."
        )

    invalid_states = set(df["state"].unique()) - valid_states
    results["invalid_state_codes"] = sorted(invalid_states)
    if invalid_states:
        raise PreprocessingError(f"BFS clean panel contains invalid state codes: {invalid_states}")

    results["missing_business_applications"] = int(df["business_applications"].isnull().sum())
    results["n_rows"] = len(df)
    results["n_states"] = int(df["state"].nunique())
    results["year_month_range"] = [str(df["year_month"].min()), str(df["year_month"].max())]

    logger.info("BFS validation passed: %s", results)
    return results


# =============================================================================
# PART B — FEMA preprocessing
# =============================================================================

def clean_fema_events(raw_fema: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter and deduplicate raw FEMA declarations to one row per
    (disasterNumber, state) — i.e. one row per distinct disaster event
    affecting a given state.

    Parameters
    ----------
    raw_fema : pd.DataFrame
        Unmodified output of load_fema_data().
    config : dict
        Parsed project configuration.

    Returns
    -------
    pd.DataFrame
        Columns: disaster_number, state, year_month, incident_type,
        declaration_type, incident_begin_date.

    Assumptions
    ----------
    - Each disasterNumber maps to exactly one state (verified in Module 0 —
      max distinct states per disasterNumber == 1). The row-level fan-out in
      the raw file is purely within-state, across counties/designated areas.
    - incidentBeginDate, incidentType, and declarationType are constant
      within a (disasterNumber, state) group (verified empirically — 0 of
      5,211 groups have more than one distinct value for any of the three),
      so keeping the first row per group loses no information.
    """
    fema_cfg = config["fema"]
    valid_states = set(config["geography"]["valid_state_codes"])

    logger.info("FEMA: starting from %s raw rows", len(raw_fema))

    # --- Restrict to 50 states + DC. This also drops the 7 territory/FAS
    # codes (AS, FM, GU, MH, MP, PW, VI) that exist in FEMA but have no BFS
    # counterpart, and — per the finalized project geography — Puerto Rico.
    filtered = raw_fema[raw_fema["state"].isin(valid_states)].copy()
    logger.info(
        "FEMA: %s rows remain after restricting to 50 states + DC (dropped %s)",
        len(filtered), len(raw_fema) - len(filtered),
    )

    # --- incidentBeginDate is our event-timing anchor (not declarationDate,
    # which lags the actual event by the federal administrative process).
    filtered["incident_begin_date"] = pd.to_datetime(
        filtered["incidentBeginDate"], errors="coerce", utc=True
    )
    n_bad_dates = filtered["incident_begin_date"].isnull().sum()
    if n_bad_dates > 0:
        logger.warning("FEMA: %s rows have unparseable incidentBeginDate and will be dropped", n_bad_dates)
        filtered = filtered[filtered["incident_begin_date"].notnull()]

    # FEMA dates come back UTC-aware; drop the tz before converting to Period
    # since we only need calendar month granularity, not time-of-day/tz precision.
    filtered["year_month"] = filtered["incident_begin_date"].dt.tz_localize(None).dt.to_period("M")

    # --- Collapse county-level fan-out. A single disaster generates many
    # rows (one per affected county); for a state-month panel we only care
    # that the disaster touched the state at all, once.
    dedup_keys = fema_cfg["dedup_keys_for_state_month_panel"]
    n_before_dedup = len(filtered)
    deduped = filtered.drop_duplicates(subset=dedup_keys, keep="first")
    logger.info(
        "FEMA: collapsed %s county-level rows into %s state-level disaster events (dedup on %s)",
        n_before_dedup, len(deduped), dedup_keys,
    )

    clean = (
        deduped.rename(columns={
            "disasterNumber": "disaster_number",
            "incidentType": "incident_type",
            "declarationType": "declaration_type",
        })
        .loc[:, ["disaster_number", "state", "year_month", "incident_type",
                 "declaration_type", "incident_begin_date"]]
        .sort_values(["state", "year_month"])
        .reset_index(drop=True)
    )

    logger.info("FEMA: cleaning complete — %s distinct state-level disaster events", len(clean))
    return clean


def validate_fema_clean(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Run structural validation checks on the cleaned FEMA events dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Output of clean_fema_events().
    config : dict
        Parsed project configuration.

    Returns
    -------
    dict
        Validation results, for inclusion in the preprocessing summary.

    Raises
    ------
    PreprocessingError
        If duplicate (disaster_number, state) keys or invalid state codes
        remain after cleaning — both indicate a logic bug, not expected data
        noise.
    """
    valid_states = set(config["geography"]["valid_state_codes"])
    results: Dict = {}

    duplicate_keys = df.duplicated(subset=["disaster_number", "state"]).sum()
    results["duplicate_event_state_keys"] = int(duplicate_keys)
    if duplicate_keys > 0:
        raise PreprocessingError(
            f"FEMA clean events have {duplicate_keys} duplicate (disaster_number, state) rows."
        )

    invalid_states = set(df["state"].unique()) - valid_states
    results["invalid_state_codes"] = sorted(invalid_states)
    if invalid_states:
        raise PreprocessingError(f"FEMA clean events contain invalid state codes: {invalid_states}")

    results["missing_year_month"] = int(df["year_month"].isnull().sum())
    results["n_events"] = len(df)
    results["n_states"] = int(df["state"].nunique())
    results["n_incident_types"] = int(df["incident_type"].nunique())

    logger.info("FEMA validation passed: %s", results)
    return results


def build_disaster_panel(fema_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cleaned FEMA events into a state-month disaster count panel.

    Parameters
    ----------
    fema_clean : pd.DataFrame
        Output of clean_fema_events() — one row per (disaster_number, state).

    Returns
    -------
    pd.DataFrame
        Columns: state, year_month, disaster_count (int). Only contains
        state-months with at least one disaster; zero-disaster months are
        filled in later at merge time against the BFS panel, which defines
        the full state-month grid.

    Assumptions
    ----------
    - "disaster_count" here means distinct federal declarations, not raw
      county-level records — this is intentional, see clean_fema_events().
    """
    panel = (
        fema_clean.groupby(["state", "year_month"])["disaster_number"]
        .nunique()
        .reset_index(name="disaster_count")
        .sort_values(["state", "year_month"])
        .reset_index(drop=True)
    )
    logger.info(
        "Disaster panel: %s state-month rows with at least one declared disaster", len(panel)
    )
    return panel


# =============================================================================
# PART C — Merge
# =============================================================================

def merge_datasets(bfs_clean: pd.DataFrame, disaster_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the cleaned BFS panel with the disaster count panel into the final
    analysis dataset.

    Parameters
    ----------
    bfs_clean : pd.DataFrame
        Output of clean_bfs() — defines the full state-month grid, since BFS
        has continuous monthly coverage for every state.
    disaster_panel : pd.DataFrame
        Output of build_disaster_panel() — sparse, only state-months with a
        declared disaster.

    Returns
    -------
    pd.DataFrame
        Columns: state, year_month, business_applications, disclosure_flag,
        disaster_count. disaster_count is 0 (not NaN) for state-months with
        no declared disaster, since "no disaster happened" is a real,
        meaningful zero — not missing data.

    Assumptions
    ----------
    - A left join from bfs_clean is correct because BFS is the panel's time
      backbone; any state-month present in FEMA but absent from BFS (there
      shouldn't be any, given both are restricted to the same 51 states)
      would be silently dropped by a left join, so we check for that in
      validation below rather than assuming it can't happen.
    """
    merged = bfs_clean.merge(disaster_panel, on=["state", "year_month"], how="left")
    merged["disaster_count"] = merged["disaster_count"].fillna(0).astype(int)
    logger.info("Merge complete: %s rows in final analysis dataset", len(merged))
    return merged


def validate_merge(
    merged: pd.DataFrame, bfs_clean: pd.DataFrame, disaster_panel: pd.DataFrame
) -> Dict:
    """
    Validate the merged final analysis dataset for duplicate keys, missing
    observations, and any FEMA state-months that failed to find a match in
    the BFS backbone (which would indicate a geography or date mismatch).

    Parameters
    ----------
    merged : pd.DataFrame
        Output of merge_datasets().
    bfs_clean : pd.DataFrame
        The BFS panel used as the join backbone.
    disaster_panel : pd.DataFrame
        The disaster panel being joined in.

    Returns
    -------
    dict
        Validation results, for inclusion in the preprocessing summary.

    Raises
    ------
    PreprocessingError
        If duplicate (state, year_month) keys exist in the merged output, or
        if the merged row count doesn't match the BFS backbone row count
        (which would mean the left join fanned out unexpectedly).
    """
    results: Dict = {}

    duplicate_keys = merged.duplicated(subset=["state", "year_month"]).sum()
    results["duplicate_state_month_keys"] = int(duplicate_keys)
    if duplicate_keys > 0:
        raise PreprocessingError(
            f"Merged dataset has {duplicate_keys} duplicate (state, year_month) rows — "
            "the disaster panel likely has an unexpected duplicate key."
        )

    if len(merged) != len(bfs_clean):
        raise PreprocessingError(
            f"Merged dataset has {len(merged)} rows but BFS backbone has {len(bfs_clean)} rows — "
            "left join should preserve BFS row count exactly."
        )

    # State-months present in the disaster panel but absent from the BFS
    # backbone would be silently dropped by the left join — check explicitly
    # rather than assuming this can't happen.
    bfs_keys = set(zip(bfs_clean["state"], bfs_clean["year_month"]))
    disaster_keys = set(zip(disaster_panel["state"], disaster_panel["year_month"]))
    unmatched_disaster_keys = disaster_keys - bfs_keys
    results["unmatched_fema_state_months"] = len(unmatched_disaster_keys)
    if unmatched_disaster_keys:
        logger.warning(
            "%s FEMA state-months had no matching BFS row and were dropped by the join: %s",
            len(unmatched_disaster_keys),
            sorted(unmatched_disaster_keys)[:10],
        )

    results["missing_business_applications"] = int(merged["business_applications"].isnull().sum())
    results["missing_disaster_count"] = int(merged["disaster_count"].isnull().sum())
    results["n_rows"] = len(merged)
    results["n_states"] = int(merged["state"].nunique())
    results["zero_disaster_months"] = int((merged["disaster_count"] == 0).sum())
    results["nonzero_disaster_months"] = int((merged["disaster_count"] > 0).sum())

    logger.info("Merge validation passed: %s", results)
    return results


# =============================================================================
# Orchestration
# =============================================================================

def _save_processed(df: pd.DataFrame, filename: str, config: Dict) -> Path:
    """
    Write a processed dataframe to data/processed/ with year_month rendered
    as a plain 'YYYY-MM' string (CSV has no native Period dtype).

    Parameters
    ----------
    df : pd.DataFrame
    filename : str
    config : dict

    Returns
    -------
    Path
        The path the file was written to.
    """
    out_df = df.copy()
    if "year_month" in out_df.columns:
        out_df["year_month"] = out_df["year_month"].astype(str)

    out_path = PROJECT_ROOT / config["paths"]["data_processed"] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %s (%s rows) to %s", filename, len(out_df), out_path)
    return out_path


def _build_preprocessing_summary(
    raw_bfs: pd.DataFrame,
    raw_fema: pd.DataFrame,
    bfs_clean: pd.DataFrame,
    fema_clean: pd.DataFrame,
    disaster_panel: pd.DataFrame,
    final_dataset: pd.DataFrame,
    bfs_validation: Dict,
    fema_validation: Dict,
    merge_validation: Dict,
) -> Dict:
    """
    Assemble the end-to-end preprocessing summary: rows removed/retained at
    each stage, missing values handled, duplicates removed, final dimensions,
    and the key assumptions made along the way.

    Parameters
    ----------
    raw_bfs, raw_fema : pd.DataFrame
        Unmodified raw inputs.
    bfs_clean, fema_clean, disaster_panel, final_dataset : pd.DataFrame
        Outputs of each pipeline stage.
    bfs_validation, fema_validation, merge_validation : dict
        Results returned by the corresponding validate_* functions.

    Returns
    -------
    dict
        JSON-serializable summary.
    """
    return {
        "bfs": {
            # Note: raw_rows is in wide format (one row per sa/naics/series/geo/year
            # combination); clean_rows is long format (one row per state-month).
            # These aren't directly comparable by subtraction — see the pipeline
            # log for row counts at each intermediate filtering step.
            "raw_rows_wide_format": len(raw_bfs),
            "clean_rows_long_format": len(bfs_clean),
            "validation": bfs_validation,
        },
        "fema": {
            "raw_rows": len(raw_fema),
            "clean_event_rows": len(fema_clean),
            "county_level_rows_collapsed": len(raw_fema) - len(fema_clean),
            "disaster_panel_rows": len(disaster_panel),
            "validation": fema_validation,
        },
        "merge": {
            "final_rows": len(final_dataset),
            "final_columns": list(final_dataset.columns),
            "validation": merge_validation,
        },
        "assumptions": [
            "BFS panel restricted to series=BA_BA, sa=U (not seasonally adjusted), "
            "naics_sector=TOTAL, per the finalized project scope.",
            "BFS Jan-Jun 2004 rows dropped as structurally non-existent "
            "(monthly BFS collection began July 2004), not treated as missing data.",
            "FEMA disasterNumber verified 1:1 with state (Module 0); the only fan-out "
            "collapsed is within-state, across counties.",
            "FEMA incidentBeginDate used as the event-timing anchor, not declarationDate, "
            "to avoid administrative lag in the timing variable.",
            "disaster_count = 0 is treated as a real observed value (no disaster that "
            "state-month), not a missing observation, per the left-join fill in merge_datasets().",
            "Puerto Rico and U.S. territories excluded from both datasets — analysis is "
            "scoped to the 50 states + DC.",
        ],
    }


def run_preprocessing_pipeline() -> pd.DataFrame:
    """
    Execute the full Module 2 pipeline end-to-end: load raw data, clean BFS,
    clean FEMA, build the disaster panel, merge, validate, and save all
    outputs to data/processed/.

    Returns
    -------
    pd.DataFrame
        The final merged analysis dataset (also written to disk).
    """
    logger.info("=" * 70)
    logger.info("Module 2: Preprocessing pipeline — starting")
    logger.info("=" * 70)

    config = load_config()

    raw_bfs = load_bfs_data(config)
    raw_fema = load_fema_data(config)

    # --- Part A: BFS
    bfs_clean = clean_bfs(raw_bfs, config)
    bfs_validation = validate_bfs_clean(bfs_clean, config)
    _save_processed(bfs_clean, config["processed_outputs"]["bfs_clean"], config)

    # --- Part B: FEMA
    fema_clean = clean_fema_events(raw_fema, config)
    fema_validation = validate_fema_clean(fema_clean, config)
    _save_processed(fema_clean, config["processed_outputs"]["fema_clean"], config)

    disaster_panel = build_disaster_panel(fema_clean)
    _save_processed(disaster_panel, config["processed_outputs"]["disaster_panel"], config)

    # --- Part C: Merge
    final_dataset = merge_datasets(bfs_clean, disaster_panel)
    merge_validation = validate_merge(final_dataset, bfs_clean, disaster_panel)
    _save_processed(final_dataset, config["processed_outputs"]["final_analysis_dataset"], config)

    # --- Summary
    summary = _build_preprocessing_summary(
        raw_bfs, raw_fema, bfs_clean, fema_clean, disaster_panel, final_dataset,
        bfs_validation, fema_validation, merge_validation,
    )
    summary_path = (
        PROJECT_ROOT / config["paths"]["data_processed"]
        / config["processed_outputs"]["preprocessing_summary"]
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved preprocessing summary to %s", summary_path)

    logger.info("=" * 70)
    logger.info("Module 2: Preprocessing pipeline — completed successfully")
    logger.info("=" * 70)

    return final_dataset


if __name__ == "__main__":
    run_preprocessing_pipeline()
