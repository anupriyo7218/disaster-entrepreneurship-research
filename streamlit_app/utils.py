"""
utils.py

Shared helpers for the Streamlit dashboard: path resolution, cached data
loaders, and small formatting functions used across every page. Centralizing
this here means each page file stays focused on layout/narrative rather than
re-implementing "how do I find the project root" five times.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import yaml

# streamlit_app/ sits one level below the project root, same as src/.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@st.cache_data
def load_config() -> dict:
    """Load config/config.yaml — cached so every page isn't re-parsing YAML."""
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _processed_path(filename: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / filename


def _table_path(filename: str) -> Path:
    return PROJECT_ROOT / "outputs" / "tables" / filename


def _report_path(filename: str) -> Path:
    return PROJECT_ROOT / "outputs" / "reports" / filename


def _figure_path(filename: str) -> Path:
    return PROJECT_ROOT / "outputs" / "figures" / filename


@st.cache_data
def load_final_dataset() -> Optional[pd.DataFrame]:
    """
    Load the Module 2 output that every other page builds on. Returns None
    (rather than raising) if Module 2 hasn't been run yet, so pages can show
    a friendly instruction instead of crashing the whole app.
    """
    path = _processed_path("final_analysis_dataset.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


@st.cache_data
def load_fema_clean() -> Optional[pd.DataFrame]:
    """Load the Module 2 FEMA event-level output (has incident_type, unlike the panel)."""
    path = _processed_path("fema_clean.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


@st.cache_data
def load_table(filename: str) -> Optional[pd.DataFrame]:
    """Load any CSV from outputs/tables/ by filename, or None if missing."""
    path = _table_path(filename)
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_report_text(filename: str) -> Optional[str]:
    """Load any text file from outputs/reports/ by filename, or None if missing."""
    path = _report_path(filename)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def figure_path_if_exists(filename: str) -> Optional[Path]:
    """Return the path to a figure in outputs/figures/ only if it actually exists."""
    path = _figure_path(filename)
    return path if path.exists() else None


def missing_pipeline_warning(module_name: str, command: str) -> None:
    """
    Standard warning shown when a page needs pipeline output that hasn't
    been generated yet — points the user at the exact command to fix it
    rather than just saying "file not found."
    """
    st.warning(
        f"⚠️ {module_name} output not found. Run `{command}` from the project "
        f"root first, then reload this page."
    )
