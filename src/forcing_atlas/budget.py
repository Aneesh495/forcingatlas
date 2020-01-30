"""Per-project byte accounting.

The cap is 5_000_000_000 bytes for this project. The accountant sums the
checkout, git objects, the virtualenv, acquired data, evidence, checkpoints,
web build caches, and the project-local temp and pip cache. It does not follow
symlinks, and it does not move work into another directory to hide it.

A write that would cross the cap raises BudgetError before the bytes land.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from forcing_atlas.constants import PROJECT_DISK_CAP_BYTES
from forcing_atlas.errors import BudgetError


COUNTED_DIRS = (
    "data",
    "evidence",
    "checkpoints",
    ".venv",
    ".git",
    "web/node_modules",
    "web/.next",
    "web/out",
    "web/public/bundle",
    ".agent-local",
)


def project_root() -> Path:
    # src/forcing_atlas/budget.py -> repository root
    return Path(__file__).resolve().parents[2]


def _file_bytes(path: Path) -> int:
    try:
        stat = path.stat()
    except OSError:
        return 0
    if not stat.st_mode & 0o100000:
        # Not a regular file. Directories and symlinks contribute nothing here.
        return 0
    return int(stat.st_size)


def directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    if root.is_symlink():
        return 0
    if root.is_file():
        return _file_bytes(root)
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            candidate = current / name
            if candidate.is_symlink():
                continue
            total += _file_bytes(candidate)
    return total


@dataclass(frozen=True, slots=True)
class DiskReport:
    cap_bytes: int
    used_bytes: int
    by_directory: dict[str, int]

    @property
    def remaining_bytes(self) -> int:
        return self.cap_bytes - self.used_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "cap_bytes": self.cap_bytes,
            "used_bytes": self.used_bytes,
            "remaining_bytes": self.remaining_bytes,
            "by_directory": dict(self.by_directory),
        }


def measure(root: Path | None = None) -> DiskReport:
    base = root or project_root()
    by_directory: dict[str, int] = {}
    # Files that sit directly in the root are part of the checkout.
    loose = 0
    if base.exists():
        for child in base.iterdir():
            if child.is_symlink():
                continue
            if child.is_file():
                loose += _file_bytes(child)
    by_directory["root_files"] = loose
    used = loose
    for relative in COUNTED_DIRS:
        path = base / relative
        size = directory_bytes(path)
        by_directory[relative] = size
        used += size
    return DiskReport(PROJECT_DISK_CAP_BYTES, used, by_directory)


def assert_room(additional_bytes: int, root: Path | None = None) -> DiskReport:
    if additional_bytes < 0:
        raise BudgetError("additional_bytes cannot be negative")
    report = measure(root)
    if report.used_bytes + additional_bytes > report.cap_bytes:
        raise BudgetError(
            f"project disk cap {report.cap_bytes} bytes would be exceeded: "
            f"used {report.used_bytes}, requested additional {additional_bytes}"
        )
    return report


def assert_stage_budget(used_bytes: int, stage_cap_bytes: int, stage: str) -> None:
    """A stage cap is tighter than the project cap and is declared before the run."""

    if stage_cap_bytes <= 0:
        raise BudgetError(f"{stage} stage cap must be positive")
    if used_bytes > stage_cap_bytes:
        raise BudgetError(
            f"{stage} used {used_bytes} bytes, above its stage cap {stage_cap_bytes}"
        )
    if stage_cap_bytes > PROJECT_DISK_CAP_BYTES:
        raise BudgetError(f"{stage} stage cap exceeds the project cap")
