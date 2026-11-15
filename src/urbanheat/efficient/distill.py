"""Knowledge distillation of the Member 2 teacher into a lite student, on inputs with clipped spectral indices.

With MSE on outputs, alpha*(p - y)^2 + (1 - alpha)*(p - t)^2 = (p - (alpha*y + (1 - alpha)*t))^2 + const, so
distillation is plain training on the blended target alpha*y + (1 - alpha)*t, and Member 2's train.fit is reused.
"""
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from urbanheat.contracts import LS_INPUTS
from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient import benchmark, lite
from urbanheat.models import train

# NDVI / NDBI / MNDWI hold outliers (-478 .. 1452) in urbanheat_v1.h5 that inflate their std ~15x (Member 2 handover)
INDEX_CH = [LS_INPUTS.index(k) for k in ("NDVI", "NDBI", "MNDWI")]


class Prepared:
    """A PatchDataset whose spectral indices are clipped to [-1, 1] and renormalised with clipped train stats.

    teacher: {pair_id: (H, W) teacher prediction} -> training target becomes alpha*y + (1 - alpha)*teacher.
    scene() always returns the true target, so evaluation is unaffected. Other attributes come from `ds`.
    """

    def __init__(self, ds, teacher=None, alpha=1.0):
        if not ds.normalize:
            raise ValueError("Prepared expects a normalised PatchDataset")
        self.ds, self.teacher, self.alpha = ds, teacher, alpha
        with h5py.File(ds.path, "r") as f:
            train_px = f["pixels"]["split"][:].astype(str) == "train"
            ls = f["pixels"]["ls"][:][train_px][:, INDEX_CH].astype("float64").clip(-1, 1)
        self.old = ds.norm["ls_mean"][INDEX_CH], ds.norm["ls_std"][INDEX_CH]
        self.norm = {**ds.norm, "ls_mean": ds.norm["ls_mean"].copy(), "ls_std": ds.norm["ls_std"].copy()}
        self.norm["ls_mean"][INDEX_CH], self.norm["ls_std"][INDEX_CH] = ls.mean(0), ls.std(0)

    def __getattr__(self, k):
        return getattr(self.__dict__["ds"], k)

    def __len__(self):
        return len(self.ds)

    def _clip(self, out):
        (om, os_), m = self.old, out["mask"]
        nm, ns = self.norm["ls_mean"][INDEX_CH], self.norm["ls_std"][INDEX_CH]
        raw = out["ls"][INDEX_CH] * os_[:, None, None] + om[:, None, None]
        out["ls"][INDEX_CH] = np.where(m, (raw.clip(-1, 1) - nm[:, None, None]) / ns[:, None, None], 0.0)
        return out

    def __getitem__(self, i):
        out = self._clip(self.ds[i])
        if self.teacher is not None:
            pid, r, c = self.ds.items[i]
            t = self.teacher[pid][r:r + self.ds.patch, c:c + self.ds.patch].astype("float32")
            out["y"] = np.where(out["mask"], self.alpha * out["y"] + (1 - self.alpha) * t, 0.0).astype("float32")
        return out

    def scene(self, pid):
        return self._clip(self.ds.scene(pid))


def teacher_maps(teacher, ds, device="cpu"):
    """Teacher prediction of every scene of `ds` (on the original, unclipped inputs it was trained on)."""
    return {pid: train.predict_scene(teacher, ds, pid, device)[0].astype("float16") for pid in ds.pairs}


def hardware(device):
    return torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine()


def run(h5, out_dir, teacher_dir=None, alpha=0.5, variant="fusion_cnn", width=32, thermal_width=16, separable=True,
        device="cpu", **fit_kw):
    """Train a lite student (distilled from `teacher_dir` unless alpha == 1), evaluate it on every valid pixel of
    the val and test scenes, measure its efficiency, and save runs/<name>/ like Member 2's train.save."""
    ds = {s: PatchDataset(h5, s) for s in ("train", "val", "test")}
    maps = teacher_maps(lite.load(teacher_dir, device), ds["train"], device) if alpha < 1 else None
    tr = Prepared(ds["train"], maps, alpha)
    ev_ds = {s: Prepared(ds[s]) for s in ("val", "test")}
    with h5py.File(h5, "r") as f:
        info_of = {k: dict(climate_group=g.attrs["climate_group"], sensor=g.attrs["sensor"]) for k, g in f["pairs"].items()}

    student = lite.lite_cnn(variant, width, thermal_width, separable)
    history = train.fit(student, tr, ev_ds["val"], device=device, **fit_kw)
    scores = {s: train.evaluate_scenes(student, d, info_of, device)[0] for s, d in ev_ds.items()}
    batch = next(iter(DataLoader(ev_ds["val"], batch_size=32)))
    config = dict(lite=True, variant=variant, width=width, thermal_width=thermal_width, separable=separable,
                  alpha=alpha, teacher=str(teacher_dir) if teacher_dir else None, clip_indices=True, **fit_kw)
    metrics = dict(run_id=Path(out_dir).name, model=f"lite_{variant}", target="anomaly", **scores,
                   **benchmark.measure(student, batch, device), epochs_run=len(history), hardware=hardware(device), **config)
    train.save(student, out_dir, metrics, history, config)
    return metrics, student
