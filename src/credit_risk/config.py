"""Configuration loading and project paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file from ``configs/`` by name (without extension)."""
    path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a path from the configuration relative to the project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
