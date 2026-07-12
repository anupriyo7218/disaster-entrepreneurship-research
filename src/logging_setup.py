"""
logging_setup.py

Centralized logging configuration for the project. Loads settings from
config/logging_config.yaml and ensures the logs/ directory exists before
any handler tries to write to it.

Usage
-----
    from src.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("Module started")
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

import yaml

# Resolve project root as the parent of this file's parent (src/ -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
LOGGING_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "logging_config.yaml"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

_configured: bool = False


def _configure_logging() -> None:
    """
    Load logging configuration from YAML and apply it via dictConfig.
    Idempotent: only configures logging once per process.
    """
    global _configured
    if _configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if LOGGING_CONFIG_PATH.exists():
        with open(LOGGING_CONFIG_PATH, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        # Ensure relative log file paths resolve against project root regardless
        # of the working directory the script was launched from.
        for handler in config_dict.get("handlers", {}).values():
            if "filename" in handler:
                handler["filename"] = str(PROJECT_ROOT / handler["filename"])
        logging.config.dictConfig(config_dict)
    else:
        # Fallback: sane default if the YAML config is missing.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    _configure_logging()
    return logging.getLogger(name)
