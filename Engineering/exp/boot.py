"""Add project root to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def ensure_root_on_path() -> Path:
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _ROOT
