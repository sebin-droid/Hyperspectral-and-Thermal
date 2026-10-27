"""Build the harmonised dataset (docs/CONTRACTS.md section 1) from Member 1's pairs + derived layers."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio

from urbanheat.contracts import ENV_INPUTS, LS_INPUTS, MET_INPUTS
from urbanheat.data import catalog
from urbanheat.data.thermal_context import landsat_date
from urbanheat.harmonize import spatial, spectral, weather

LS_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
ATTRS = ["sensor", "cell_id", "climate_group", "profile", "split", "block_1deg", "landsat_scene"]


def _read(path, names=None):
    """(data, band descriptions, (transform, crs)); nodata -> NaN; `names` selects bands by description."""
    with rasterio.open(path) as s:
        desc = list(s.descriptions)
        data = s.read().astype("float32")
        if s.nodata is not None and np.isfinite(s.nodata):
            data[data == s.nodata] = np.nan
        geo = (s.transform, s.crs)
    return (data if names is None else np.stack([data[desc.index(n)] for n in names])), desc, geo


def sensor_wavelengths(idx):
    """Band centres of each sensor (every pair of a sensor has the same band set)."""
    out = {}
    for sensor, g in idx.groupby("sensor"):
        with rasterio.open(g.reflectance_path.iloc[0]) as s:
            out[sensor] = spectral.wavelengths(s.descriptions)
    return out


def read_pair(row, tgt_wl, hcfg, context_path):
    """All layers of one pair on the common wavelength grid and `grid_m` pixels, plus the valid mask."""
    cube, desc, (transform, crs) = _read(row.reflectance_path)
    cube /= 10000.0
    hs = spectral.apply(spectral.interp_matrix(spectral.wavelengths(desc), tgt_wl, hcfg["max_gap_nm"]), cube)
    ls, _, _ = _read(row.landsat_path, LS_BANDS + ["LST_K"])
    aux, _, _ = _read(row.aux_path, ENV_INPUTS + ["landsat_clear", "landcover_worldcover"])
    if Path(context_path).exists():
        ctx, _, _ = _read(context_path, ["lst_context_c"])
    else:
        ctx = np.full((1,) + hs.shape[1:], np.nan, "float32")

    clear = aux[3] == 1                                     # Landsat clear, same rule as Member 1's QC
    lst = np.where(clear & (ls[6] > 0), ls[6] - 273.15, np.nan)
    hs, opt = np.where(clear, hs, np.nan), np.where(clear, ls[:6], np.nan)

    f = spatial.factor(float(row.res_m), hcfg["grid_m"])
    m = hcfg["min_valid_frac"]
    hs, opt, env, ctx = (spatial.block_mean(a, f, m) for a in (hs, opt, aux[:3], ctx))
    lst = spatial.block_mean(lst, f, m)
    landcover = spatial.block_mode(np.nan_to_num(aux[4], nan=0).astype("uint8"), f)

    blue, green, red, nir, swir1, swir2 = opt
    with np.errstate(invalid="ignore", divide="ignore"):
        idxs = [(nir - red) / (nir + red), (swir1 - nir) / (swir1 + nir), (green - swir1) / (green + swir1)]
    ls_in = np.concatenate([opt, np.stack(idxs)]).astype("float32")

    valid = (np.isfinite(hs).all(0) & np.isfinite(ls_in).all(0) & np.isfinite(env).all(0) & np.isfinite(lst)
             & np.isfinite(ctx[0]))
    return dict(hs=hs, ls=ls_in, env=env, thermal=ctx, landcover=landcover, lst_c=lst.astype("float32"),
                valid=valid, transform=spatial.transform(transform, f), crs=crs)


class _Stats:
    """Running per-channel mean / std over valid train pixels."""

    def __init__(self):
        self.s, self.ss, self.n = {}, {}, {}

    def add(self, key, x, w=None):          # x: (C, N) values, or (C,) repeated w times
        x = x.astype("float64")
        s, ss, n = (x.sum(1), (x ** 2).sum(1), x.shape[1]) if w is None else (x * w, x ** 2 * w, w)
        self.s[key] = self.s.get(key, 0) + s
        self.ss[key] = self.ss.get(key, 0) + ss
        self.n[key] = self.n.get(key, 0) + n

    def result(self):
        out = {}
        for k in self.s:
            mean = self.s[k] / self.n[k]
            out[k + "_mean"] = np.atleast_1d(mean).astype("float32")
            out[k + "_std"] = np.atleast_1d(np.sqrt(np.maximum(self.ss[k] / self.n[k] - mean ** 2, 1e-12))).astype("float32")
        return out


def build(cfg, workers=4, log=print):
    dcfg, hcfg = cfg["data"], cfg["harmonize"]
    idx = catalog.load_index(dcfg["index_file"], dcfg["raw_dir"], dcfg["sensors"])
    missing = idx[~idx.on_disk]
    if len(missing):
        raise SystemExit(f"{len(missing)} pairs not on disk; run `urbanheat download` first")
    catalog.check_split(idx, cfg["split"]["column"])
    idx["landsat_date"] = [landsat_date(s) for s in idx.landsat_scene]
    ctx_root = Path(dcfg["derived_dir"]) / "thermal_context"
    n_ctx = sum((ctx_root / d / "thermal_context.tif").exists() for d in idx.rel_dir)
    log(f"{len(idx)} pairs; thermal context present for {n_ctx}")

    sw = sensor_wavelengths(idx)
    tgt_wl = spectral.common_grid(sw.values(), hcfg["wl_min_nm"], hcfg["wl_max_nm"], hcfg["wl_step_nm"],
                                  hcfg["max_gap_nm"])
    log(f"common wavelength grid: {len(tgt_wl)} bands, {tgt_wl.min():.0f}-{tgt_wl.max():.0f} nm")

    log("ERA5 weather at the overpass ...")
    met = weather.for_pairs(idx, Path(dcfg["derived_dir"]) / "weather" / "era5_open_meteo.json")

    out = Path(dcfg["processed_dir"]) / f"urbanheat_v{hcfg['version']}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.part")
    rng = np.random.default_rng(cfg["split"]["seed"])
    stats = _Stats()
    pix = {k: [] for k in ["hs", "ls", "env", "thermal", "met", "landcover", "lst_c", "lst_anom", "pair_id", "split"]}
    rows = []

    def pairs_read(ex):
        # read ahead at most 2 x workers pairs so memory stays small
        rows_ = list(idx.itertuples())
        job = lambda r: read_pair(r, tgt_wl, hcfg, ctx_root / r.rel_dir / "thermal_context.tif")
        for k in range(0, len(rows_), 2 * workers):
            chunk = rows_[k:k + 2 * workers]
            yield from zip(chunk, ex.map(job, chunk))

    with h5py.File(tmp, "w") as h5, ThreadPoolExecutor(workers) as ex:
        gp = h5.create_group("pairs")
        for j, (row, p) in enumerate(pairs_read(ex), 1):
            v = p["valid"]
            m = np.array(met[row.pair_id], "float32")
            median = float(np.median(p["lst_c"][v])) if v.any() else np.nan
            g = gp.create_group(row.pair_id)
            ch = (32, 32)
            g.create_dataset("hs", data=p["hs"].astype("float16"), compression="lzf", chunks=(len(tgt_wl),) + ch)
            for k in ["ls", "env", "thermal"]:
                g.create_dataset(k, data=p[k], compression="lzf", chunks=(p[k].shape[0],) + ch)
            g.create_dataset("met", data=m)
            g.create_dataset("landcover", data=p["landcover"], compression="lzf")
            g.create_dataset("lst_c", data=p["lst_c"], compression="lzf")
            g.create_dataset("valid", data=v.astype("uint8"), compression="lzf")
            for k in ATTRS:
                g.attrs[k] = str(getattr(row, k))
            g.attrs.update(landsat_date=str(row.landsat_date), lst_median_c=median, n_valid=int(v.sum()),
                           transform=list(p["transform"])[:6], crs=p["crs"].to_string())
            rows.append(dict(pair_id=row.pair_id, sensor=row.sensor, split=getattr(row, cfg["split"]["column"]),
                             n_valid=int(v.sum()), lst_median_c=median, height=v.shape[0], width=v.shape[1]))

            if v.any():
                vals = {k: p[k][:, v] for k in ["hs", "ls", "env", "thermal"]}
                y, anom = p["lst_c"][v], p["lst_c"][v] - median
                if getattr(row, cfg["split"]["column"]) == "train":   # normalisation from train only
                    for k, x in vals.items():
                        stats.add(k, x)
                    stats.add("met", m, w=int(v.sum()))
                    stats.add("lst", y[None])
                    stats.add("lst_anom", anom[None])
                pick = rng.choice(v.sum(), min(hcfg["pixels_per_pair"], int(v.sum())), replace=False)
                for k, x in vals.items():
                    pix[k].append(x[:, pick].T)
                pix["met"].append(np.tile(m, (pick.size, 1)))
                pix["landcover"].append(p["landcover"][v][pick])
                pix["lst_c"].append(y[pick])
                pix["lst_anom"].append(anom[pick])
                pix["pair_id"].append(np.full(pick.size, row.pair_id))
                pix["split"].append(np.full(pick.size, getattr(row, cfg["split"]["column"])))
            if j % 50 == 0 or j == len(idx):
                log(f"{j}/{len(idx)} pairs written")

        gx = h5.create_group("pixels")
        for k, parts in pix.items():
            a = np.concatenate(parts)
            gx.create_dataset(k, data=a.astype("S16") if a.dtype.kind == "U" else a, compression="lzf")
        gn = h5.create_group("norm")
        for k, a in stats.result().items():
            gn.create_dataset(k, data=a)
        h5.attrs.update(version=hcfg["version"], grid_m=hcfg["grid_m"], wavelengths_nm=tgt_wl,
                        split_column=cfg["split"]["column"],
                        index_md5=hashlib.md5(Path(dcfg["index_file"]).read_bytes()).hexdigest(),
                        inputs=json.dumps(dict(ls=LS_INPUTS, env=ENV_INPUTS, met=MET_INPUTS,
                                               thermal=["lst_context_c"])))
    tmp.replace(out)
    table = pd.DataFrame(rows)
    table.to_csv(out.with_suffix(".pairs.csv"), index=False)
    return out, table
