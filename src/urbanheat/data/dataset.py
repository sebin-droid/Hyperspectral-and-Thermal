"""Readers for the harmonised dataset: image patches for CNNs, the pixel table for RF / XGBoost."""
import json

import h5py
import numpy as np

from urbanheat.contracts import SENSORS

INPUT_KEYS = ["hs", "ls", "env", "thermal"]
# (key, channel) inputs given relative to the pair's median, like the anomaly target: their level differs between
# scenes for reasons unrelated to the within-city pattern (other dates' weather, the city's altitude)
RELATIVE = [("thermal", 0), ("env", 0)]


def patch_origins(valid, size, stride, min_valid):
    h, w = valid.shape
    return [(r, c) for r in range(0, h - size + 1, stride) for c in range(0, w - size + 1, stride)
            if valid[r:r + size, c:c + size].mean() >= min_valid]


class PatchDataset:
    """Patches of one split as dicts of contracts keys (+ "y" target in deg C), normalised with train stats.

    target: "anomaly" (LST minus the pair's median, the surface effect) or "absolute".
    relative: give thermal context and elevation as differences from the pair's median (default for anomaly).
    Works with torch.utils.data.DataLoader; the file is opened lazily per worker.
    """

    def __init__(self, path, split, patch=32, stride=None, min_valid=0.8, target="anomaly", normalize=True,
                 sensors=SENSORS, relative=None):
        if target not in ("anomaly", "absolute"):
            raise ValueError(target)
        self.path, self.patch, self.target, self.normalize = str(path), patch, target, normalize
        self.relative = RELATIVE if (relative if relative is not None else target == "anomaly") else []
        stride = stride or (patch // 2 if split == "train" else patch)
        self.items, self.pairs, self.medians, self._h5 = [], [], {}, None
        with h5py.File(self.path, "r") as f:
            self.wavelengths = f.attrs["wavelengths_nm"][:]
            self.inputs = json.loads(f.attrs["inputs"])
            self.norm = {k: f["norm"][k][:] for k in f["norm"]}
            for pid, g in f["pairs"].items():
                if g.attrs["split"] != split or g.attrs["sensor"] not in sensors:
                    continue
                valid = g["valid"][:].astype(bool)
                self.pairs.append(pid)
                self.medians[pid] = {(k, ch): float(np.median(g[k][ch][valid])) if valid.any() else 0.0
                                     for k, ch in self.relative}
                for r, c in patch_origins(valid, patch, stride, min_valid):
                    self.items.append((pid, r, c))

    def __len__(self):
        return len(self.items)

    def window(self, pid, rows=slice(None), cols=slice(None)):
        """Inputs, target and mask of a window of pair `pid` (the whole scene by default)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        g = self._h5["pairs"][pid]
        mask = g["valid"][rows, cols].astype(bool)
        out = {k: g[k][:, rows, cols].astype("float32") for k in INPUT_KEYS}
        out["met"] = g["met"][:].astype("float32")
        for (k, ch), m in self.medians[pid].items():
            out[k][ch] -= m
        if self.normalize:
            rel = {}
            for k, ch in self.relative:
                rel.setdefault(k, []).append(ch)
            for k in INPUT_KEYS + ["met"]:
                shape = (-1, 1, 1) if k != "met" else (-1,)
                mean = self.norm[k + "_mean"].copy()
                mean[rel.get(k, [])] = 0.0                     # relative channels are already centred
                out[k] = (out[k] - mean.reshape(shape)) / self.norm[k + "_std"].reshape(shape)
        for k in INPUT_KEYS:
            out[k][:, ~mask] = 0.0
        y = g["lst_c"][rows, cols]
        if self.target == "anomaly":
            y = y - g.attrs["lst_median_c"]
        out["y"] = np.where(mask, y, 0.0).astype("float32")
        out["mask"] = mask
        out["sensor"] = SENSORS.index(g.attrs["sensor"])
        return out

    def __getitem__(self, i):
        pid, r, c = self.items[i]
        s = self.patch
        return self.window(pid, slice(r, r + s), slice(c, c + s))

    def scene(self, pid):
        return self.window(pid)


def load_pixels(path, split=None):
    """Pixel table as a dict of arrays (hs, ls, env, thermal, met, landcover, lst_c, lst_anom, pair_id, split)."""
    with h5py.File(path, "r") as f:
        px = {k: f["pixels"][k][:] for k in f["pixels"]}
    for k in ("pair_id", "split"):
        px[k] = px[k].astype(str)
    if split is not None:
        sel = px["split"] == split
        px = {k: v[sel] for k, v in px.items()}
    return px
