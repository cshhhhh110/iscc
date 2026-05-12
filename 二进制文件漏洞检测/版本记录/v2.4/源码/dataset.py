"""Dataset helpers for CSV rows and binary path mapping."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def binary_path(binaries_dir: Path, binary_id: str) -> Path:
    return binaries_dir / f"{binary_id}.exe"


def iter_binary_paths(rows: Iterable[Dict[str, str]], binaries_dir: Path) -> Iterator[Tuple[str, Path]]:
    for row in rows:
        yield row["binary_id"], binary_path(binaries_dir, row["binary_id"])
