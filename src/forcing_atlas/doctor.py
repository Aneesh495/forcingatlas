"""Environment report for the local profile.

`python -m forcing_atlas.doctor` prints JSON. A full acceptance profile is
not declared healthy here. This command only describes the machine, the
importable numerical stack, and the disk accountant.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys

from forcing_atlas.budget import measure
from forcing_atlas.constants import MISSING_POLICY, __version__


def collect() -> dict[str, object]:
    numpy_version = None
    numpy_error = None
    try:
        import numpy as np

        numpy_version = np.__version__
        blas = None
        try:
            blas = str(np.__config__.show(stdout=False))  # type: ignore[attr-defined]
        except Exception:
            blas = "unavailable"
    except Exception as exc:  # pragma: no cover - doctor must still print
        numpy_error = str(exc)
        blas = None

    torch_version = None
    cuda = False
    try:
        import torch

        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
    except Exception:
        torch_version = None

    xarray_version = None
    try:
        import xarray

        xarray_version = xarray.__version__
    except Exception:
        xarray_version = None

    report = measure()
    return {
        "product": "forcing-atlas",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy_version,
        "numpy_error": numpy_error,
        "blas": blas,
        "torch": torch_version,
        "cuda": cuda,
        "xarray": xarray_version,
        "missing_policy": MISSING_POLICY,
        "disk": report.to_dict(),
        "tmp": os.environ.get("TMPDIR"),
        "graphviz": shutil.which("dot"),
        "full_profile": "not_assessed",
        "note": "doctor does not mark the full campaign passed",
    }


def main() -> int:
    json.dump(collect(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
