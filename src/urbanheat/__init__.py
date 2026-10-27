"""Urban heat: explain and mitigate urban surface heat with hyperspectral + thermal remote sensing."""
import importlib.util
import os
import sys
from pathlib import Path

__version__ = "0.1.0"


def _use_own_proj():
    # Other software (e.g. PostGIS) may point PROJ_LIB at an older proj.db, and rasterio then reads every
    # CRS as None. Use the PROJ database that belongs to this Python environment instead.
    spec = importlib.util.find_spec("rasterio")
    candidates = [Path(sys.prefix) / "Library" / "share" / "proj", Path(sys.prefix) / "share" / "proj"]
    if spec and spec.origin:
        candidates.insert(0, Path(spec.origin).parent / "proj_data")      # pip wheels ship their own
    for d in candidates:
        if (d / "proj.db").exists():
            os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"] = str(d)
            return


_use_own_proj()
