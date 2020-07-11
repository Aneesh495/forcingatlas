"""Checksums for the local source files the studies read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forcing_atlas.catalog import PINNED


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_source_manifest(root: Path, destination: Path) -> Path:
    rows = []
    for source in PINNED:
        if source.group == "simulation-archive":
            continue
        path = root / source.file_name
        if not path.is_file():
            rows.append({"dataset_id": source.dataset_id, "present": False})
            continue
        rows.append(
            {
                "dataset_id": source.dataset_id,
                "present": True,
                "semantics": source.semantics.value,
                "file_name": source.file_name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "license_id": source.license_id,
            }
        )
    for path in sorted((root / "cmip6").glob("*.nc")) if (root / "cmip6").is_dir() else []:
        rows.append(
            {
                "dataset_id": path.name,
                "present": True,
                "semantics": "source_simulation",
                "file_name": f"cmip6/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "license_id": "cc-by-4.0",
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"files": rows}, indent=2))
    return destination
