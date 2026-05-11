"""Byte-window feature utilities for the v1.2 neural branch."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pefile
from tqdm import tqdm

from dataset import binary_path


PathLike = Union[str, Path]

DEFAULT_BYTE_LENGTH = 8192


def _entry_point_offset(raw: bytes) -> Optional[int]:
    try:
        pe = pefile.PE(data=raw, fast_load=True)
        try:
            entry_rva = int(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
            entry_offset = int(pe.get_offset_from_rva(entry_rva))
        finally:
            pe.close()
    except Exception:
        return None
    if 0 <= entry_offset < len(raw):
        return entry_offset
    return None


def _window(raw: bytes, start: int, length: int) -> bytes:
    if length <= 0 or not raw:
        return b""
    if len(raw) <= length:
        return raw[:length]
    clipped_start = max(0, min(start, len(raw) - length))
    return raw[clipped_start : clipped_start + length]


def extract_byte_window(binary_path_value: PathLike, byte_length: int = DEFAULT_BYTE_LENGTH) -> np.ndarray:
    """Return a deterministic fixed-length byte window for one binary.

    Long files use the header, bytes near the PE entry point, and the tail.
    This keeps import/header signals while exposing execution-adjacent bytes.
    """

    path = Path(binary_path_value)
    output = np.zeros(byte_length, dtype=np.uint8)
    try:
        raw = path.read_bytes()
    except OSError:
        return output
    if not raw:
        return output

    if len(raw) <= byte_length:
        selected = raw
    else:
        head_length = byte_length // 3
        entry_length = byte_length // 3
        tail_length = byte_length - head_length - entry_length
        entry_offset = _entry_point_offset(raw)
        if entry_offset is None:
            entry_offset = len(raw) // 2
        entry_start = entry_offset - entry_length // 2
        selected = (
            _window(raw, 0, head_length)
            + _window(raw, entry_start, entry_length)
            + _window(raw, len(raw) - tail_length, tail_length)
        )

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
