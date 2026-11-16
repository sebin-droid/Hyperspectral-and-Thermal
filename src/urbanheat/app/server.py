"""Dashboard API (FastAPI): map layers, what-if on a drawn polygon, explanations and the assistant.

Run:  urbanheat serve   (or  uvicorn "urbanheat.app.server:create_app" --factory)  then open http://127.0.0.1:8000
All maps are returned as transparent PNGs with their EPSG:4326 bounds, ready for a Leaflet image overlay.
"""
import ast
import base64
import io
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from matplotlib import colormaps, colors
from PIL import Image
from pydantic import BaseModel
from rasterio.warp import transform_bounds

from urbanheat.app import assistant
from urbanheat.contracts import ENV_INPUTS, LS_INPUTS, Scenario
from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient import lite
from urbanheat.efficient.distill import Prepared
from urbanheat.explain import xai
from urbanheat.models import train
from urbanheat.scenarios import engine

STATIC = Path(__file__).parent / "static"
PRESETS = {   # typical changes from Member 1's observed data (Member 3 handover)
    "built_to_green": {"NDVI": 0.34, "NDBI": -0.20, "built_share": -0.30},
    "green_to_built": {"NDVI": -0.34, "NDBI": 0.20, "built_share": 0.30},
    "greening": {"NDVI": 0.1, "NDBI": -0.06},
}
# layer -> (label, unit, colormap, scale): "pct" = 2-98 % range, "sym" = symmetric around 0, (lo, hi) = fixed
STYLE = {
    "lst": ("Observed LST", "°C", "inferno", "pct"),
    "pred": ("Predicted LST", "°C", "inferno", "pct"),
    "anomaly": ("Predicted LST anomaly", "°C", "RdBu_r", "sym"),
    "error": ("Prediction error", "°C", "PuOr_r", "sym"),
    "rgb": ("True colour (Landsat)", "", None, None),
    "NDVI": ("NDVI (vegetation)", "", "YlGn", (-0.2, 0.9)),
    "NDBI": ("NDBI (built-up)", "", "YlOrBr", (-0.5, 0.5)),
    "MNDWI": ("MNDWI (water)", "", "Blues", (-0.8, 0.6)),
    "built_share": ("Built share", "", "Reds", (0, 1)),
    "thermal": ("Thermal context (other dates)", "°C", "inferno", "pct"),
    "gradcam_surface": ("Grad-CAM surface branch (hottest 10 %)", "", "magma", (0, 1)),
    "gradcam_thermal": ("Grad-CAM thermal branch (hottest 10 %)", "", "magma", (0, 1)),
    "delta": ("Scenario ΔLST", "°C", "RdBu_r", "sym"),
}
STYLE.update({f"attr_{g}": (f"Attribution: {g}", "°C", "RdBu_r", "sym") for g in xai.GROUPS})


class Region(BaseModel):
    pid: str
    geometry: dict | None = None          # GeoJSON geometry (EPSG:4326); None = whole scene


class ScenarioRequest(Region):
    name: str = "custom"
    changes: dict[str, float]


class Question(BaseModel):
    question: str
    pid: str | None = None
    geometry: dict | None = None


def render(name, a, mask, bounds):
    """Layer as {url (PNG data URL), bounds [[S, W], [N, E]], vmin, vmax, colors (legend stops), label, unit}."""
    label, unit, cmap, scale = STYLE[name]
    if cmap is None:                                                 # RGB, (3, H, W) reflectance
        lo, hi = np.percentile(a[:, mask], [2, 98]) if mask.any() else (0, 1)
        rgba = np.concatenate([np.clip((a - lo) / (hi - lo + 1e-9), 0, 1), mask[None].astype(float)]).transpose(1, 2, 0)
        vmin = vmax = None
    else:
        v = a[mask & np.isfinite(a)]
        if not len(v):
            v = np.zeros(1)
        if scale == "pct":
            vmin, vmax = np.percentile(v, [2, 98])
        elif scale == "sym":
            vmax = max(float(np.percentile(np.abs(v), 98)), 1e-3)
            vmin = -vmax
        else:
            vmin, vmax = scale
        rgba = colormaps[cmap](colors.Normalize(vmin, vmax, clip=True)(np.nan_to_num(a)))
        rgba[..., 3] = mask & np.isfinite(a)
    buf = io.BytesIO()
    Image.fromarray((rgba * 255).astype("uint8")).save(buf, "PNG")
    stops = [colors.to_hex(colormaps[cmap](x)) for x in np.linspace(0, 1, 7)] if cmap else []
    w, s, e, n = bounds
    return dict(name=name, label=label, unit=unit, url="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                bounds=[[s, w], [n, e]], vmin=None if vmin is None else float(vmin),
                vmax=None if vmax is None else float(vmax), colors=stops)


class Workspace:
    """Model, datasets and per-city caches behind the API."""

    def __init__(self, h5, model=None, run_dir=None, device="cpu"):
        self.h5, self.device = str(h5), device
        self.model = model if model is not None else lite.load(run_dir, device)
        self.run_dir = Path(run_dir) if run_dir else None
        self.run = self.run_dir.name if run_dir else type(self.model).__name__
        with h5py.File(self.h5, "r") as f:
            self.info = {pid: dict(sensor=g.attrs["sensor"], split=g.attrs["split"], climate=g.attrs["climate_group"],
                                   date=str(g.attrs.get("landsat_date", "")), median=float(g.attrs["lst_median_c"]),
                                   bounds=transform_bounds(g.attrs["crs"], "EPSG:4326",
                                                           *_bounds(g.attrs["transform"], g["valid"].shape)))
                         for pid, g in f["pairs"].items()}
        ds = {s: Prepared(PatchDataset(self.h5, s)) for s in sorted({i["split"] for i in self.info.values()})}
        self.ds = {pid: ds[i["split"]] for pid, i in self.info.items()}
        self.knowledge = assistant.Knowledge()
        self.scene = lru_cache(16)(lambda pid: self.ds[pid].scene(pid))
        self.attr = lru_cache(16)(lambda pid: xai.by_group(xai.integrated_gradients(self.model, self.scene(pid), device)))
        self.pred = lru_cache(32)(lambda pid: train.predict_scene(self.model, self.ds[pid], pid, device)[0])
        self.cam = lru_cache(16)(lambda pid: xai.grad_cam(self.model, self.scene(pid), device=device))

    def _check(self, pid):
        if pid not in self.info:
            raise HTTPException(404, f"unknown city {pid}")

    def raw(self, pid, key, ch):
        with h5py.File(self.h5, "r") as f:
            return f["pairs"][pid][key][ch].astype("float32")

    def layer(self, pid, name):
        self._check(pid)
        if name not in STYLE or name == "delta":
            raise HTTPException(404, f"unknown layer {name}")
        s, med = self.scene(pid), self.info[pid]["median"]
        m = s["mask"]
        if name == "lst":
            a = s["y"] + med
        elif name == "pred":
            a = self.pred(pid) + med
        elif name == "anomaly":
            a = self.pred(pid)
        elif name == "error":
            a = self.pred(pid) - s["y"]
        elif name == "rgb":
            a = self.raw(pid, "ls", slice(0, 3))[::-1]               # blue, green, red -> red, green, blue
        elif name in ("NDVI", "NDBI", "MNDWI"):
            a = self.raw(pid, "ls", LS_INPUTS.index(name)).clip(-1, 1)
        elif name == "built_share":
            a = self.raw(pid, "env", ENV_INPUTS.index("built_share"))
        elif name == "thermal":
            a = self.raw(pid, "thermal", 0)
        elif name.startswith("gradcam_"):
            cam = self.cam(pid)
            if name[8:] not in cam:
                raise HTTPException(404, f"the model has no {name[8:]} branch")
            a = cam[name[8:]]
        else:
            g = name[5:]
            if g not in self.attr(pid):
                raise HTTPException(404, f"the model does not read {g}")
            a = self.attr(pid)[g]
        return render(name, a, m, self.info[pid]["bounds"])

    def mask(self, pid, geometry):
        if geometry is None:
            return None
        where = engine.region_mask(geometry, self.h5, pid) & self.scene(pid)["mask"]
        if not where.any():
            raise HTTPException(400, "the drawn region holds no valid pixels of this city")
        return where

    def explain(self, pid, geometry=None):
        """Facts about a region (default: whole city): predicted vs observed anomaly and the factors behind it."""
        self._check(pid)
        s, where = self.scene(pid), self.mask(pid, geometry)
        sel = s["mask"] if where is None else where
        t = xai.scene_summary(self.attr(pid), s, where).sort_values("mean_shap", key=abs, ascending=False)
        factors = [dict(feature=r.feature, mean_c=round(r.mean_shap, 3)) for r in t.itertuples()]
        i = self.info[pid]
        facts = dict(city=pid, sensor=i["sensor"], climate=i["climate"], landsat_date=i["date"],
                     region="drawn polygon" if where is not None else "whole city", pixels=int(sel.sum()),
                     city_median_lst_c=round(i["median"], 2),
                     predicted_anomaly_c=f"{self.pred(pid)[sel].mean():+.2f} °C",
                     observed_anomaly_c=f"{s['y'][sel].mean():+.2f} °C",
                     top_factors=", ".join(f"{f['feature']} {f['mean_c']:+.2f} °C" for f in factors[:4]))
        out = dict(facts=facts, factors=factors, note=xai.CAVEAT)
        if where is not None:
            cam = xai.grad_cam(self.model, s, where, self.device)
            out["layers"] = [render(f"gradcam_{b}", c, s["mask"], i["bounds"]) for b, c in cam.items()]
        return out

    def scenario(self, req):
        self._check(req.pid)
        if req.geometry is None:
            raise HTTPException(400, "draw a region: the target is an anomaly from the city median, so a "
                                     "whole-city change has no meaningful effect")
        try:
            sc = Scenario(req.name, req.changes)
            where = self.mask(req.pid, req.geometry)
            d = engine.delta_lst(self.model, self.ds[req.pid], req.pid, sc, self.device, where=where)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        valid = np.isfinite(d)
        stats = dict(mean_inside_c=float(np.nanmean(d[where])), min_inside_c=float(np.nanmin(d[where])),
                     max_inside_c=float(np.nanmax(d[where])), pixels_inside=int(where.sum()),
                     mean_outside_c=float(d[valid & ~where].mean()) if (valid & ~where).any() else 0.0)
        return dict(layer=render("delta", d, valid, self.info[req.pid]["bounds"]), stats=stats,
                    scenario=dict(name=req.name, changes=req.changes), note=engine.NOTE)

    def insights(self):
        """Model-wide results for the insights panel: factor ranking per climate (explain/global_importance.csv),
        small vs Member 2's model, single-factor response tests (notebooks/member4/01_xai), test accuracy, cost."""
        d = self.run_dir
        if d is None or not (d / "explain" / "global_importance.csv").exists():
            return dict(available=False, message="No explain/ tables for this model yet: run notebooks/member4/01_xai.")
        imp = pd.read_csv(d / "explain" / "global_importance.csv")
        imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp.groupby("climate_group")["mean_abs_shap"].transform("sum")
        out = dict(available=True, model=self.run, note=xai.CAVEAT,
                   importance={c: _records(g[["feature", "mean_abs_shap", "share_pct", "direction"]])
                               for c, g in imp.groupby("climate_group")})
        top = imp[imp.climate_group == "all"].sort_values("share_pct").iloc[-1]
        findings = [f"{top.feature} carries {top.share_pct:.0f} % of all attribution over the test cities."]
        for key, f in [("comparison", "comparison.csv"), ("responses", "response_tests.csv")]:
            if (d / "explain" / f).exists():
                out[key] = _records(pd.read_csv(d / "explain" / f))
        for r in out.get("responses", []):
            if self.run in r["model"]:
                change = ", ".join(f"{k} {v:+g}" for k, v in ast.literal_eval(r["change"]).items())
                findings.append(f"{change} alone in a 32 × 32 block: {r['mean_dlst_c']:+.3f} °C on average, "
                                f"cooler in {100 * r['cities_cooling']:.0f} % of test cities.")
        out["findings"] = findings
        if (d / "metrics.json").exists():
            t = json.loads((d / "metrics.json").read_text())["test"]
            out["accuracy"] = [dict(group="all", r2=t["r2"], mae=t["mae"], n=t["n"])] + [
                dict(group=g, r2=v["r2"], mae=v["mae"], n=v["n"])
                for k in ("per_climate", "per_sensor") for g, v in t.get(k, {}).items()]
        if (d.parent / "m3_slide17.csv").exists():                   # Member 3's accuracy vs cost table
            out["efficiency"] = _records(pd.read_csv(d.parent / "m3_slide17.csv").rename(columns={"Unnamed: 0": "model"}))
        return out

    def export(self, pid, kind):
        """Path of a temporary GeoTIFF: 'pred' = predicted LST (CONTRACTS section 5), 'attr' = attribution per group."""
        self._check(pid)
        s = self.scene(pid)
        if kind == "pred":
            bands, note = {"predicted_lst_c": self.pred(pid) + self.info[pid]["median"]}, \
                "Predicted land surface temperature (deg C), not air temperature."
        elif kind == "attr":
            bands, note = self.attr(pid), xai.CAVEAT
        else:
            raise HTTPException(404, f"unknown export {kind}")
        fd, path = tempfile.mkstemp(suffix=".tif")
        os.close(fd)
        return xai.write_bands(path, bands, self.h5, pid, mask=s["mask"], model=self.run, note=note)


def _records(df):
    return [{k: (None if isinstance(v, float) and np.isnan(v) else round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items()} for r in df.to_dict("records")]


def _bounds(t, shape):
    h, w = shape
    xs, ys = (t[2], t[2] + t[0] * w), (t[5], t[5] + t[4] * h)
    return min(xs), min(ys), max(xs), max(ys)


def create_app(h5=None, run_dir=None, model=None, device="cpu"):
    """The FastAPI app. Defaults: the dataset from the config and Member 3's recommended model m3_lite16_kd."""
    if h5 is None:
        from urbanheat.config import load_config
        cfg = load_config()
        h5 = Path(cfg["data"]["processed_dir"]) / f"urbanheat_v{cfg['harmonize']['version']}.h5"
    if model is None and run_dir is None:
        run_dir = Path(__file__).resolve().parents[3] / "runs" / "m3_lite16_kd"
    ws = Workspace(h5, model, run_dir, device)
    app = FastAPI(title="Urban heat dashboard")
    app.state.ws = ws

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/cities")
    def cities():
        return dict(model=ws.run, presets=PRESETS, layers={k: v[0] for k, v in STYLE.items() if k != "delta"},
                    cities=[dict(pid=p, **{k: v for k, v in i.items() if k != "bounds"},
                                 center=[(i["bounds"][1] + i["bounds"][3]) / 2, (i["bounds"][0] + i["bounds"][2]) / 2])
                            for p, i in sorted(ws.info.items(), key=lambda x: (x[1]["split"] != "test", x[0]))])

    @app.get("/api/layer/{pid}/{name}")
    def layer(pid: str, name: str):
        return ws.layer(pid, name)

    @app.get("/api/insights")
    def insights():
        return ws.insights()

    @app.get("/api/export/{pid}/{kind}.tif")
    def export(pid: str, kind: str):
        path = ws.export(pid, kind)
        return FileResponse(path, media_type="image/tiff", filename=f"{pid}_{kind}.tif",
                            background=BackgroundTask(os.remove, path))

    @app.post("/api/explain")
    def explain(r: Region):
        return ws.explain(r.pid, r.geometry)

    @app.post("/api/scenario")
    def scenario(r: ScenarioRequest):
        return ws.scenario(r)

    @app.post("/api/ask")
    def ask(q: Question):
        facts = ws.explain(q.pid, q.geometry)["facts"] if q.pid else {}
        return assistant.ask(q.question, facts, ws.knowledge)

    return app
