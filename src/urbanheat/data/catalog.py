"""The pair catalogue: Member 1's dataset_index.csv, one row per hyperspectral + Landsat pair."""
from pathlib import Path

import pandas as pd

SENSORS = ("EMIT", "ENMAP", "HYPERION")
SPLITS = ("train", "val", "test")
REQUIRED = ["sensor", "pair_no", "rel_dir", "reflectance_file", "landsat_file", "aux_file", "cell_id", "split"]


def load_index(index_file, raw_dir=None, sensors=SENSORS):
    """Index rows for `sensors`, with a unique `pair_id` (e.g. EMIT_12) and, if `raw_dir` is given,
    absolute file paths plus `on_disk` (all three rasters present)."""
    idx = pd.read_csv(index_file)
    missing = [c for c in REQUIRED if c not in idx.columns]
    if missing:
        raise ValueError(f"{index_file} lacks columns {missing}")
    idx = idx[idx.sensor.isin(sensors)].copy()
    idx["pair_id"] = idx.sensor + "_" + idx.pair_no.astype(str)
    if raw_dir is not None:
        raw = Path(raw_dir)
        for col, key in [("reflectance_file", "reflectance_path"), ("landsat_file", "landsat_path"),
                         ("aux_file", "aux_path")]:
            idx[key] = [str(raw / d / f) for d, f in zip(idx.rel_dir, idx[col])]
        idx["on_disk"] = [all(Path(r[k]).exists() for k in ("reflectance_path", "landsat_path", "aux_path"))
                          for _, r in idx.iterrows()]
    return idx.sort_values(["sensor", "pair_no"]).reset_index(drop=True)


def check_split(idx, column="split"):
    """Raise if a cell appears in more than one split (the split must be by place)."""
    n = idx.groupby("cell_id")[column].nunique()
    leaking = n[n > 1].index.tolist()
    if leaking:
        raise ValueError(f"cells in more than one split: {leaking[:10]}")


def summary(idx, column="split"):
    """Pairs per sensor and split, plus cells per sensor."""
    t = pd.crosstab(idx.sensor, idx[column], margins=True, margins_name="total")
    t["cells"] = idx.groupby("sensor").cell_id.nunique().reindex(t.index)
    t.loc["total", "cells"] = idx.cell_id.nunique()
    return t.astype(int)
