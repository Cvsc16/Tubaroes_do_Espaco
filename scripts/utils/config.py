#!/usr/bin/env python3
"""Helpers to centralize access to the project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load the top-level configuration file and return it as a dict."""

    cfg_path = path or CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_bbox(cfg: Dict[str, Any]) -> list[float] | None:
    """Return the bbox [west, south, east, north] if present."""

    return cfg.get("aoi", {}).get("bbox")


def project_root() -> Path:
    """Expose the absolute project root for scripts that need it."""

    return ROOT


__all__ = ["load_config", "get_bbox", "project_root", "CONFIG_PATH", "ROOT"]
