"""Chunked float arrays with a JSON header.

The header names the variable, the semantics, the baseline, and the missing
policy. Chunks are raw little-endian float32. A missing value is a NaN in the
payload, never a sentinel that could be averaged as zero. Reload checks the
header checksum of each chunk against the bytes on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forcing_atlas.constants import MISSING_POLICY
from forcing_atlas.errors import ContractError


HEADER_NAME = "header.json"
CHUNK_PREFIX = "chunk-"


@dataclass(slots=True)
class Cube:
    values: np.ndarray
    attrs: dict[str, object]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_cube(path: Path, values: np.ndarray, attrs: dict[str, object], *, time_chunk: int = 32) -> Path:
    """Write `values` as float32 chunks along axis 0.

    Existing contents of `path` are replaced only after the new header is
    fsynced. A crash leaves the previous directory intact because the write
    goes to `path.next` and is then renamed.
    """

    array = np.asarray(values)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ContractError(f"cube values have unsupported dtype {array.dtype}")
    if np.isinf(array).any():
        raise ContractError("cube values contain infinities")
    payload = np.array(array, dtype=np.float32, copy=True)
    # Preserve NaN. float32 conversion must not turn NaN into zero.
    if np.isnan(array).any():
        if not np.isnan(payload).any():
            raise ContractError("float32 conversion dropped NaN values")
    staging = path.with_name(path.name + ".next")
    if staging.exists():
        for child in staging.iterdir():
            child.unlink()
    else:
        staging.mkdir(parents=True)
    count = payload.shape[0] if payload.ndim else 1
    step = max(1, time_chunk)
    chunks: list[dict[str, object]] = []
    if payload.ndim == 0:
        slices = [payload.reshape(1)]
    else:
        slices = [payload[start : min(start + step, count)] for start in range(0, count, step)]
    for index, block in enumerate(slices):
        raw = np.ascontiguousarray(block).tobytes()
        name = f"{CHUNK_PREFIX}{index:04d}.f32"
        target = staging / name
        with target.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        chunks.append(
            {
                "name": name,
                "shape": list(block.shape),
                "sha256": _sha256(raw),
                "bytes": len(raw),
            }
        )
    header = {
        "format": "forcing-atlas-cube/1",
        "dtype": "float32",
        "missing": MISSING_POLICY,
        "shape": list(payload.shape),
        "chunks": chunks,
        "attrs": attrs,
    }
    header_path = staging / HEADER_NAME
    encoded = json.dumps(header, indent=2, sort_keys=True).encode("utf-8")
    with header_path.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        retired = path.with_name(path.name + ".prev")
        if retired.exists():
            for child in retired.iterdir():
                child.unlink()
            retired.rmdir()
        os.replace(path, retired)
        os.replace(staging, path)
        for child in retired.iterdir():
            child.unlink()
        retired.rmdir()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, path)
    return path


def read_cube(path: Path) -> Cube:
    header_path = path / HEADER_NAME
    if not header_path.is_file():
        raise ContractError(f"{path} has no cube header")
    header = json.loads(header_path.read_text())
    if header.get("format") != "forcing-atlas-cube/1":
        raise ContractError(f"unsupported cube format {header.get('format')}")
    if header.get("missing") != MISSING_POLICY:
        raise ContractError("cube missing policy is not nan_not_zero")
    pieces: list[np.ndarray] = []
    for chunk in header["chunks"]:
        raw_path = path / chunk["name"]
        raw = raw_path.read_bytes()
        if len(raw) != chunk["bytes"]:
            raise ContractError(f"{chunk['name']} byte count changed")
        digest = _sha256(raw)
        if digest != chunk["sha256"]:
            raise ContractError(f"{chunk['name']} checksum changed")
        block = np.frombuffer(raw, dtype=np.float32).reshape(chunk["shape"])
        pieces.append(block.copy())
    if not pieces:
        raise ContractError("cube has no chunks")
    if len(header["shape"]) == 0:
        values = pieces[0].reshape(())
    else:
        values = np.concatenate(pieces, axis=0)
    if list(values.shape) != header["shape"]:
        raise ContractError(f"cube shape {list(values.shape)} does not match header {header['shape']}")
    return Cube(values, dict(header.get("attrs") or {}))
