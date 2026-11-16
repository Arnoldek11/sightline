"""Resolve data/report locations. Override with SIGHTLINE_DATA_DIR / SIGHTLINE_REPORTS_DIR."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("SIGHTLINE_DATA_DIR", REPO_ROOT / "data")).expanduser()


def reports_dir() -> Path:
    return Path(os.environ.get("SIGHTLINE_REPORTS_DIR", REPO_ROOT / "reports")).expanduser()
