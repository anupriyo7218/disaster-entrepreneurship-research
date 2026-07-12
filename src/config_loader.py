"""
config_loader.py

Small utility to load the project's YAML configuration file into a plain
Python dict. Kept separate from logging_setup.py so each module has a
single, obvious responsibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load the project YAML configuration.

    Parameters
    ----------
    config_path : Path | str
        Path to the config YAML file. Defaults to config/config.yaml
        relative to the project root.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the given path.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
