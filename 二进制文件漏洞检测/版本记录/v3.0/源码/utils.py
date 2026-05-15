"""Shared utility helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm as _tqdm


def tqdm(iterable=None, /, **kwargs):
    """tqdm wrapper that writes to stdout (not stderr) and auto-disables when output is piped."""
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("disable", not sys.stdout.isatty())
    return _tqdm(iterable, **kwargs)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
