"""
data_loader.py

Module 1 deliverable.

Loads the two raw source datasets for the project — Census Business
Formation Statistics (BFS) and FEMA Disaster Declarations Summaries — and
validates that they can be read successfully.

IMPORTANT: This module performs NO preprocessing, filtering, cleaning, or
transformation. Its sole purpose is to confirm the raw files load correctly
and to log basic structural information for traceability. Cleaning and
transformation logic will live in later modules (e.g. src/preprocessing.py),
kept separate so the raw-load step remains a trustworthy, side-effect-free
starting point for the pipeline.

Usage
-----
    python -m src.data_loader
or
    from src.data_loader import load_bfs_data, load_fema_data
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config_loader import load_config
from src.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class DataLoadError(Exception):
    """Raised when a raw dataset cannot be loaded or fails basic validation."""


def _resolve_raw_path(filename: str, data_raw_dir: str) -> Path:
    """
    Build an absolute path to a raw data file and confirm it exists.

    Parameters
    ----------
    filename : str
        Name of the file within the raw data directory.
    data_raw_dir : str
        Relative path (from project root) to the raw data directory.

    Returns
    -------
    Path
        Absolute path to the file.

    Raises
    ------
    DataLoadError
        If the file does not exist at the resolved path.
    """
    path = PROJECT_ROOT / data_raw_dir / filename
    if not path.exists():
        raise DataLoadError(
            f"Expected raw data file not found: {path}. "
            f"Place the source CSV in '{data_raw_dir}/' before running this script."
        )
    return path


def load_bfs_data(config: Dict) -> pd.DataFrame:
    """
    Load the raw Census Business Formation Statistics CSV.

    No filtering, type coercion, or cleaning is applied here — disclosure
    codes ('D', 'S') and missing values are left exactly as they appear in
    the source file. This keeps the raw load auditable against the original
    Census file.

    Parameters
    ----------
    config : dict
        Parsed project configuration (see config/config.yaml).

    Returns
    -------
    pd.DataFrame
        Raw BFS data, unmodified.

    Raises
    ------
    DataLoadError
        If the file is missing or cannot be parsed as CSV.
    """
    filename = config["bfs"]["raw_filename"]
    path = _resolve_raw_path(filename, config["paths"]["data_raw"])

    logger.info("Loading BFS data from %s", path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - we want to wrap any parse failure
        raise DataLoadError(f"Failed to read BFS file at {path}: {exc}") from exc

    _log_dataset_summary("BFS (Business Formation Statistics)", df)
    return df


def load_fema_data(config: Dict) -> pd.DataFrame:
    """
    Load the raw FEMA Disaster Declarations Summaries CSV.

    No filtering, deduplication, or date parsing is applied here — this is
    an unmodified read of the source file for traceability.

    Parameters
    ----------
    config : dict
        Parsed project configuration (see config/config.yaml).

    Returns
    -------
    pd.DataFrame
        Raw FEMA data, unmodified.

    Raises
    ------
    DataLoadError
        If the file is missing or cannot be parsed as CSV.
    """
    filename = config["fema"]["raw_filename"]
    path = _resolve_raw_path(filename, config["paths"]["data_raw"])

    logger.info("Loading FEMA data from %s", path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to read FEMA file at {path}: {exc}") from exc

    _log_dataset_summary("FEMA (Disaster Declarations Summaries)", df)
    return df


def _log_dataset_summary(label: str, df: pd.DataFrame) -> None:
    """
    Log shape, columns, dtypes, and missing-value counts for a loaded
    dataframe. Purely observational — does not modify df.

    Parameters
    ----------
    label : str
        Human-readable dataset name for log messages.
    df : pd.DataFrame
        The dataframe to summarize.
    """
    logger.info("[%s] shape: %s rows x %s columns", label, df.shape[0], df.shape[1])
    logger.info("[%s] columns: %s", label, list(df.columns))
    logger.debug("[%s] dtypes:\n%s", label, df.dtypes)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("[%s] no missing values in any column", label)
    else:
        logger.info("[%s] columns with missing values:\n%s", label, missing)

    n_duplicates = df.duplicated().sum()
    logger.info("[%s] fully duplicated rows: %s", label, n_duplicates)


def validate_datasets_loaded(bfs_df: pd.DataFrame, fema_df: pd.DataFrame) -> None:
    """
    Run lightweight sanity checks confirming both datasets loaded as expected.

    These are structural checks only (non-empty, expected key columns
    present) — not business-logic validation, which belongs in a later
    preprocessing/validation module.

    Parameters
    ----------
    bfs_df : pd.DataFrame
    fema_df : pd.DataFrame

    Raises
    ------
    DataLoadError
        If either dataframe is empty or missing an expected key column.
    """
    if bfs_df.empty:
        raise DataLoadError("BFS dataframe loaded but is empty.")
    if fema_df.empty:
        raise DataLoadError("FEMA dataframe loaded but is empty.")

    expected_bfs_cols = {"sa", "naics_sector", "series", "geo", "year"}
    missing_bfs_cols = expected_bfs_cols - set(bfs_df.columns)
    if missing_bfs_cols:
        raise DataLoadError(f"BFS file is missing expected columns: {missing_bfs_cols}")

    expected_fema_cols = {"disasterNumber", "state", "incidentType", "incidentBeginDate"}
    missing_fema_cols = expected_fema_cols - set(fema_df.columns)
    if missing_fema_cols:
        raise DataLoadError(f"FEMA file is missing expected columns: {missing_fema_cols}")

    logger.info("Validation passed: both datasets loaded with expected key columns present.")


def main() -> None:
    """
    Entry point: load both raw datasets, validate them, and log summaries.
    Performs no preprocessing and writes no output files.
    """
    logger.info("=" * 70)
    logger.info("Module 1: Raw data load and validation — starting")
    logger.info("=" * 70)

    config = load_config()

    bfs_df = load_bfs_data(config)
    fema_df = load_fema_data(config)

    validate_datasets_loaded(bfs_df, fema_df)

    logger.info("Module 1: Raw data load and validation — completed successfully")


if __name__ == "__main__":
    main()
