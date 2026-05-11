"""Byte-window feature utilities for the v1.1 neural branch."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from dataset import binary_path


PathLike = Union[str, Path]

DEFAULT_BYTE_LENGTH = 4096


def extract_byte_window(binary_path_value: PathLike, byte_length: int = DEFAULT_BYTE_LENGTH) -> np.ndarray:
    """Return a deterministic fixed-length byte window for one binary.

    Long files use the first half and last half of the file. This keeps PE
    headers/import-related bytes and tail data while staying CPU-friendly.
    """

    path = Path(binary_path_value)
    raw = path.read_bytes()
    output = np.zeros(byte_length, dtype=np.uint8)
    if not raw:
        return output

    if len(raw) <= byte_length:
        selected = raw
    else:
        head_length = byte_length // 2
        tail_length = byte_length - head_length
        selected = raw[:head_length] + raw[-tail_length:]

    arr = np.frombuffer(selected, dtype=np.uint8)
    copy_length = min(len(arr), byte_length)
    output[:copy_length] = arr[:copy_length]
    return output


def rows_to_byte_matrix(
    rows: List[Dict[str, str]],
    binaries_dir: Path,
    byte_length: int = DEFAULT_BYTE_LENGTH,
    desc: str = "Extracting byte windows",
) -> Tuple[np.ndarray, List[str]]:
    matrix = np.zeros((len(rows), byte_length), dtype=np.uint8)
    binary_ids: List[str] = []

    for index, row in enumerate(tqdm(rows, desc=desc, total=len(rows))):
        binary_id = row["binary_id"]
        binary_ids.append(binary_id)
        matrix[index] = extract_byte_window(binary_path(binaries_dir, binary_id), byte_length)

    return matrix, binary_ids
