"""Readers for the harmonised dataset: image patches for CNNs, the pixel table for RF / XGBoost."""
import json

import h5py
import numpy as np

from urbanheat.contracts import SENSORS

INPUT_KEYS = ["hs", "ls", "env", "thermal"]


def patch_origins(valid, size, stride, min_valid):
    h, w = valid.shape
    return [(r, c) for r in range(0, h - size + 1, stride) for c in range(0, w - size + 1, stride)
            if valid[r:r + size, c:c + size].mean() >= min_valid]


class PatchDataset:
    """Patches of one split as dicts of contracts keys (+ "y" target in deg C), normalised with train stats.

    target: "anomaly" (LST minus the pair's median, the surface effect) or "absolute".
    Works with torch.utils.data.DataLoader; the file is opened lazily per worker.
    """

    def __init__(self, path, split, patch=32, stride=None, min_valid=0.8, target="anomaly", normalize=True,
                 sensors=SENSORS):
        if target not in ("anomaly", "absolute"):
            raise ValueError(target)
        self.path, self.patch, self.target, self.normalize = str(path), patch, target, normalize
        stride = stride or (patch // 2 if split == "train" else patch)
        self.items, self._h5 = [], None
        with h5py.File(self.path, "r") as f:
            self.wavelengths = f.attrs["wavelengths_nm"][:]
            self.inputs = json.loads(f.attrs["inputs"])
            self.norm = {k: f["norm"][k][:] for k in f["norm"]}
            for pid, g in f["pairs"].items():
                if g.attrs["split"] != split or g.attrs["sensor"] not in sensors:
                    continue
                for r, c in patch_origins(g["valid"][:].astype(bool), patch, stride, min_valid):
                    self.items.append((pid, r, c))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        pid, r, c = self.items[i]
        g, s = self._h5["pairs"][pid], self.patch
        mask = g["valid"][r:r + s, c:c + s].astype(bool)
        out = {k: g[k][:, r:r + s, c:c + s].astype("float32") for k in INPUT_KEYS}
        out["met"] = g["met"][:].astype("float32")
        if self.normalize:
            for k in INPUT_KEYS + ["met"]:
                shape = (-1, 1, 1) if k != "met" else (-1,)
                out[k] = (out[k] - self.norm[k + "_mean"].reshape(shape)) / self.norm[k + "_std"].reshape(shape)
        for k in INPUT_KEYS:
            out[k][:, ~mask] = 0.0
        y = g["lst_c"][r:r + s, c:c + s]
        if self.target == "anomaly":
            y = y - g.attrs["lst_median_c"]
        out["y"] = np.where(mask, y, 0.0).astype("float32")
        out["mask"] = mask
        out["sensor"] = SENSORS.index(g.attrs["sensor"])
        return out


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
