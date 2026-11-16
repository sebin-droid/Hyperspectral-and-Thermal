"""Urban Heat Explorer API (FastAPI): places with data, map layers, "why is this spot hot", what-if, assistant.

Run:  urbanheat serve   (or  uvicorn "urbanheat.app.server:create_app" --factory)  then open http://127.0.0.1:8000
Only the 422 image pairs of the dataset have data, so the user picks one of them and looks at it; there is no
free drawing. Maps are transparent PNGs with EPSG:4326 bounds, ready for a Leaflet image overlay.
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
from affine import Affine
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from matplotlib import colormaps, colors
from PIL import Image
from pydantic import BaseModel
from rasterio.warp import transform as warp, transform_bounds
from scipy.ndimage import uniform_filter
from starlette.background import BackgroundTask

from urbanheat.app import assistant
from urbanheat.contracts import ENV_INPUTS, LS_INPUTS, Scenario
from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient import lite
from urbanheat.efficient.distill import Prepared
from urbanheat.explain import xai
from urbanheat.models import train
from urbanheat.scenarios import engine

STATIC = Path(__file__).parent / "static"
# place names per dataset cell, made once offline with reverse_geocoder (GeoNames, CC BY 4.0), see 03_dashboard
PLACES = Path(__file__).parent / "places.csv"
# what-if presets: typical changes from Member 1's observed data (Member 3 handover), in plain words
PRESETS = {
    "built_to_green": ("Turn built-up land into a park", {"NDVI": 0.34, "NDBI": -0.20, "built_share": -0.30}),
    "greening": ("Add street trees and grass", {"NDVI": 0.1, "NDBI": -0.06}),
    "green_to_built": ("Build on green land", {"NDVI": -0.34, "NDBI": 0.20, "built_share": 0.30}),
}
# input groups in plain words (for the "why is this spot hot" card and the assistant)
FRIENDLY = {"thermal context": "Heat seen on other days", "built_share": "Buildings and pavement",
            "NDBI": "Built-up or bare surfaces", "NDVI": "Greenery", "MNDWI": "Water",
            "hyperspectral": "Surface materials (detailed spectrum)", "landsat bands": "Surface colour and brightness",
            "elevation": "Height", "slope": "Slope"}
# layer -> (label, unit, colormap, scale, low word, high word, basic?)
# scale: "pct" = 2-98 % range, "sym" = symmetric around 0, (lo, hi) = fixed
STYLE = {
    "rgb": ("Satellite photo", "", None, None, "", "", True),
    "hotspots": ("Hot and cool spots", "°C vs. city typical", "RdBu_r", "sym", "cooler", "hotter", True),
    "lst": ("Surface temperature", "°C", "inferno", "pct", "cooler", "hotter", True),
    "NDVI": ("Greenery", "NDVI", "YlGn", (-0.2, 0.9), "bare", "very green", True),
    "built_share": ("Buildings and pavement", "share of ground", "Reds", (0, 1), "none", "fully built", True),
    "MNDWI": ("Water", "MNDWI", "Blues", (-0.8, 0.6), "dry", "water", True),
    "pred": ("Surface temperature (model)", "°C", "inferno", "pct", "cooler", "hotter", False),
    "anomaly": ("Hot and cool spots (model)", "°C vs. city typical", "RdBu_r", "sym", "cooler", "hotter", False),
    "error": ("Model error (model − measured)", "°C", "PuOr_r", "sym", "too cool", "too hot", False),
    "thermal": ("Heat seen on other days", "°C", "inferno", "pct", "cooler", "hotter", False),
    "NDBI": ("Built-up or bare surfaces", "NDBI", "YlOrBr", (-0.5, 0.5), "low", "high", False),
    "gradcam_surface": ("Where the model looked: surface (hottest 10 %)", "", "magma", (0, 1), "little", "a lot", False),
    "gradcam_thermal": ("Where the model looked: past heat (hottest 10 %)", "", "magma", (0, 1), "little", "a lot", False),
    "delta": ("What-if change", "°C", "RdBu_r", "sym", "cooler", "hotter", False),
}
STYLE.update({f"attr_{g}": (f"Why: {FRIENDLY[g].lower()}", "°C", "RdBu_r", "sym", "cools", "heats", False)
              for g in xai.GROUPS})


class Spot(BaseModel):
    pid: str
    lat: float
    lng: float
    radius_m: float = 600


class WhatIf(Spot):
    preset: str


class Question(BaseModel):
    question: str
    pid: str | None = None
    lat: float | None = None
    lng: float | None = None
    radius_m: float = 600


def render(name, a, mask, bounds):
    """Layer as {url (PNG data URL), bounds [[S, W], [N, E]], vmin, vmax, colors (legend stops), label, unit, words}."""
    label, unit, cmap, scale, lo, hi, _ = STYLE[name]
    if cmap is None:                                                 # RGB, (3, H, W) reflectance
        q = np.percentile(a[:, mask], [2, 98]) if mask.any() else (0, 1)
        rgba = np.concatenate([np.clip((a - q[0]) / (q[1] - q[0] + 1e-9), 0, 1), mask[None].astype(float)]).transpose(1, 2, 0)
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
    return dict(name=name, label=label, unit=unit, low=lo, high=hi, colors=stops, bounds=[[s, w], [n, e]],
                url="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                vmin=None if vmin is None else float(vmin), vmax=None if vmax is None else float(vmax))


class Workspace:
    """Model, datasets and per-image caches behind the API."""

    def __init__(self, h5, model=None, run_dir=None, device="cpu"):
        self.h5, self.device = str(h5), device
        self.model = model if model is not None else lite.load(run_dir, device)
        self.run_dir = Path(run_dir) if run_dir else None
        self.run = self.run_dir.name if run_dir else type(self.model).__name__
        places = pd.read_csv(PLACES, keep_default_na=False).set_index("cell_id") if PLACES.exists() else pd.DataFrame()
        self.info = {}
        with h5py.File(self.h5, "r") as f:
            for pid, g in f["pairs"].items():
                t, crs, shape = Affine(*g.attrs["transform"][:6]), g.attrs["crs"], g["valid"].shape
                b = transform_bounds(crs, "EPSG:4326", *_bounds(t, shape))
                cell = str(g.attrs.get("cell_id", pid))
                p = places.loc[cell] if cell in places.index else None
                geo = crs == "EPSG:4326"
                self.info[pid] = dict(
                    sensor=g.attrs["sensor"], split=g.attrs["split"], climate=g.attrs["climate_group"],
                    date=str(g.attrs.get("landsat_date", "")), median=float(g.attrs["lst_median_c"]), bounds=b,
                    cell=cell, name=p["name"] if p is not None else f"{(b[1] + b[3]) / 2:.2f}, {(b[0] + b[2]) / 2:.2f}",
                    region=p["region"] if p is not None else "", cc=p["cc"] if p is not None else "",
                    transform=t, crs=crs, shape=shape,
                    pix_m=(abs(t.a) * 111320 * np.cos(np.radians((b[1] + b[3]) / 2)) if geo else abs(t.a),
                           abs(t.e) * 110540 if geo else abs(t.e)))
        ds = {s: Prepared(PatchDataset(self.h5, s)) for s in sorted({i["split"] for i in self.info.values()})}
        self.ds = {pid: ds[i["split"]] for pid, i in self.info.items()}
        self.knowledge = assistant.Knowledge()
        self.scene = lru_cache(16)(lambda pid: self.ds[pid].scene(pid))
        self.attr = lru_cache(16)(lambda pid: xai.by_group(xai.integrated_gradients(self.model, self.scene(pid), device)))
        self.pred = lru_cache(32)(lambda pid: train.predict_scene(self.model, self.ds[pid], pid, device)[0])
        self.cam = lru_cache(16)(lambda pid: xai.grad_cam(self.model, self.scene(pid), device=device))
        self.explain = lru_cache(64)(self.explain)          # the assistant asks again for the same spot
        self.summary = lru_cache(64)(self.summary)

    def _check(self, pid):
        if pid not in self.info:
            raise HTTPException(404, f"unknown image {pid}")

    def raw(self, pid, key, ch):
        with h5py.File(self.h5, "r") as f:
            return f["pairs"][pid][key][ch].astype("float32")

    def places(self):
        """One entry per place (dataset cell), its images newest first; test images first among places."""
        by = {}
        for pid, i in self.info.items():
            w, s, e, n = i["bounds"]
            p = by.setdefault(i["cell"], dict(cell=i["cell"], name=i["name"], region=i["region"], cc=i["cc"],
                                              climate=i["climate"], center=[(s + n) / 2, (w + e) / 2], images=[]))
            p["images"].append(dict(pid=pid, date=i["date"], sensor=i["sensor"], split=i["split"]))
        for p in by.values():
            p["images"].sort(key=lambda m: m["date"], reverse=True)
        return sorted(by.values(), key=lambda p: (p["cc"], p["name"]))

    def lonlat_to_rc(self, pid, lat, lng):
        i = self.info[pid]
        x, y = (lng, lat) if i["crs"] == "EPSG:4326" else [v[0] for v in warp("EPSG:4326", i["crs"], [lng], [lat])]
        c, r = ~i["transform"] * (x, y)
        return r, c

    def rc_to_lonlat(self, pid, r, c):
        i = self.info[pid]
        x, y = i["transform"] * (c + 0.5, r + 0.5)
        return (y, x) if i["crs"] == "EPSG:4326" else tuple(v[0] for v in warp(i["crs"], "EPSG:4326", [x], [y])[::-1])

    def spot_mask(self, pid, lat, lng, radius_m):
        """(H, W) bool: valid pixels within radius_m of the clicked point."""
        r0, c0 = self.lonlat_to_rc(pid, lat, lng)
        h, w = self.info[pid]["shape"]
        px, py = self.info[pid]["pix_m"]
        rr, cc = np.mgrid[:h, :w] + 0.5
        m = np.hypot((rr - r0) * py, (cc - c0) * px) <= max(radius_m, max(px, py))
        m &= self.scene(pid)["mask"]
        if not m.any():
            raise HTTPException(400, "No data at that spot. Click inside the coloured image of the city.")
        return m

    def layer(self, pid, name):
        self._check(pid)
        if name not in STYLE or name == "delta":
            raise HTTPException(404, f"unknown layer {name}")
        s, med = self.scene(pid), self.info[pid]["median"]
        if name == "lst":
            a = s["y"] + med
        elif name == "hotspots":
            a = s["y"]
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
        return render(name, a, s["mask"], self.info[pid]["bounds"])

    def summary(self, pid):
        """Place card: what the image shows and how well the model does on it, plus the hottest / coolest spots."""
        self._check(pid)
        s, i = self.scene(pid), self.info[pid]
        m, y, p = s["mask"], s["y"], self.pred(pid)
        px, py = i["pix_m"]
        built = self.raw(pid, "env", ENV_INPUTS.index("built_share"))[m]
        ndvi = self.raw(pid, "ls", LS_INPUTS.index("NDVI")).clip(-1, 1)[m]
        smooth = uniform_filter(np.where(m, y, 0.0), 9) / np.maximum(uniform_filter(m.astype(float), 9), 1e-6)
        spots = {}
        for k, fill, fn in (("hottest", -np.inf, np.argmax), ("coolest", np.inf, np.argmin)):
            r, c = np.unravel_index(fn(np.where(m, smooth, fill)), m.shape)   # centre of the hottest ~500 m block
            lat, lng = self.rc_to_lonlat(pid, r, c)
            spots[k] = dict(lat=float(lat), lng=float(lng), diff_c=float(smooth[r, c]))
        ss = ((y[m] - y[m].mean()) ** 2).sum()
        return dict(pid=pid, name=i["name"], region=i["region"], cc=i["cc"], date=i["date"], sensor=i["sensor"],
                    climate=i["climate"], typical_c=round(i["median"], 1),
                    hottest_c=round(float(np.percentile(y[m], 95) + i["median"]), 1),
                    coolest_c=round(float(np.percentile(y[m], 5) + i["median"]), 1),
                    area_km2=round(float(m.sum() * px * py / 1e6), 1), built_pct=round(float(100 * built.mean())),
                    green_pct=round(float(100 * (ndvi > 0.4).mean())),
                    model_error_c=round(float(np.abs(p[m] - y[m]).mean()), 2),
                    model_r2=round(float(1 - ((p[m] - y[m]) ** 2).sum() / ss), 2) if ss > 0 else None,
                    tested=i["split"] == "test", spots=spots)

    def explain(self, pid, lat, lng, radius_m):
        """Facts about a spot: measured and modelled difference from the city typical, and the reasons."""
        self._check(pid)
        s, i = self.scene(pid), self.info[pid]
        where = self.spot_mask(pid, lat, lng, radius_m)
        groups, base, pred = xai.explain_region(self.model, s, where, self.device)   # base + sum(groups) = pred
        reasons = sorted([dict(feature=g, label=FRIENDLY[g], effect_c=round(v, 2)) for g, v in groups.items()],
                         key=lambda r: -abs(r["effect_c"]))
        facts = dict(place=f"{i['name']}, {i['region']}".strip(", "), date=i["date"], spot_radius_m=int(radius_m),
                     pixels=int(where.sum()), city_typical_surface_c=round(i["median"], 1),
                     measured_surface_c=round(float(s["y"][where].mean() + i["median"]), 1),
                     measured_vs_typical_c=round(float(s["y"][where].mean()), 2),
                     model_vs_typical_c=round(pred, 2), average_place_c=round(base, 2),
                     top_factors=", ".join(f"{r['label']} {r['effect_c']:+.2f} °C" for r in reasons[:3]))
        return dict(facts=facts, reasons=reasons, note=xai.CAVEAT)

    def whatif(self, req):
        self._check(req.pid)
        if req.preset not in PRESETS:
            raise HTTPException(400, f"unknown what-if {req.preset}; choose one of {sorted(PRESETS)}")
        label, changes = PRESETS[req.preset]
        where = self.spot_mask(req.pid, req.lat, req.lng, req.radius_m)
        d = engine.delta_lst(self.model, self.ds[req.pid], req.pid, Scenario(req.preset, changes), self.device, where=where)
        valid = np.isfinite(d)
        stats = dict(mean_c=float(np.nanmean(d[where])), most_cooling_c=float(np.nanmin(d[where])),
                     most_warming_c=float(np.nanmax(d[where])), pixels=int(where.sum()),
                     mean_outside_c=float(d[valid & ~where].mean()) if (valid & ~where).any() else 0.0)
        return dict(layer=render("delta", d, valid, self.info[req.pid]["bounds"]), stats=stats, label=label,
                    changes=changes, note=engine.NOTE)

    def insights(self):
        """Model-wide results for the "About the model" panel: factor ranking per climate (explain/
        global_importance.csv), small vs Member 2's model, one-factor tests (notebooks/member4/01_xai), accuracy, cost."""
        d = self.run_dir
        if d is None or not (d / "explain" / "global_importance.csv").exists():
            return dict(available=False, message="No explain/ tables for this model yet: run notebooks/member4/01_xai.")
        imp = pd.read_csv(d / "explain" / "global_importance.csv")
        imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp.groupby("climate_group")["mean_abs_shap"].transform("sum")
        imp["label"] = imp["feature"].map(FRIENDLY)
        out = dict(available=True, model=self.run, note=xai.CAVEAT,
                   importance={c: _records(g[["feature", "label", "mean_abs_shap", "share_pct", "direction"]])
                               for c, g in imp.groupby("climate_group")})
        top = imp[imp.climate_group == "all"].sort_values("share_pct").iloc[-1]
        findings = [f"{FRIENDLY[top.feature]} explains {top.share_pct:.0f} % of the model's prediction: places that "
                    f"were hot on other days are predicted hot again."]
        for key, f in [("comparison", "comparison.csv"), ("responses", "response_tests.csv")]:
            if (d / "explain" / f).exists():
                out[key] = _records(pd.read_csv(d / "explain" / f))
        for r in out.get("responses", []):
            if self.run in r["model"]:
                change = ", ".join(f"{k} {v:+g}" for k, v in ast.literal_eval(r["change"]).items())
                findings.append(f"Changing only {change} in a 2 km block: {r['mean_dlst_c']:+.3f} °C on average "
                                f"(cooler in {100 * r['cities_cooling']:.0f} % of test cities).")
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
    xs, ys = (t.c, t.c + t.a * w), (t.f, t.f + t.e * h)
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
    app = FastAPI(title="Urban Heat Explorer")
    app.state.ws = ws

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/places")
    def places():
        return dict(model=ws.run, n_images=len(ws.info), places=ws.places(),
                    presets={k: v[0] for k, v in PRESETS.items()},
                    layers=[dict(name=k, label=v[0], basic=v[6]) for k, v in STYLE.items() if k != "delta"])

    @app.get("/api/summary/{pid}")
    def summary(pid: str):
        return ws.summary(pid)

    @app.get("/api/layer/{pid}/{name}")
    def layer(pid: str, name: str):
        return ws.layer(pid, name)

    @app.post("/api/spot")
    def spot(q: Spot):
        return ws.explain(q.pid, q.lat, q.lng, q.radius_m)

    @app.post("/api/whatif")
    def whatif(q: WhatIf):
        return ws.whatif(q)

    @app.post("/api/ask")
    def ask(q: Question):
        facts = {}
        if q.pid:                                 # the place card, plus the clicked circle if there is one
            facts = {k: v for k, v in ws.summary(q.pid).items() if k not in ("spots", "pid")}
            if q.lat is not None:
                facts.update(ws.explain(q.pid, q.lat, q.lng, q.radius_m)["facts"])
        return assistant.ask(q.question, facts, ws.knowledge)

    @app.get("/api/insights")
    def insights():
        return ws.insights()

    @app.get("/api/export/{pid}/{kind}.tif")
    def export(pid: str, kind: str):
        path = ws.export(pid, kind)
        return FileResponse(path, media_type="image/tiff", filename=f"{pid}_{kind}.tif",
                            background=BackgroundTask(os.remove, path))

    return app
