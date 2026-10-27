"""Thermal context layer: per pixel median Landsat LST of OTHER dates in the same season, on each pair's grid.

It is the input of the thermal branch. The pair's own Landsat scene (the target) and every scene within
`exclude_days` of it are left out, so the layer carries the place's usual heat pattern, not the answer.
"""
import datetime as dt
import time
from pathlib import Path

import numpy as np
import rasterio

QA_BAD_BITS = 0b11111          # QA_PIXEL bits 0-4: fill, dilated cloud, cirrus, cloud, cloud shadow
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
NODATA = -9999.0
BANDS = ["lst_context_c", "n_obs"]


def landsat_date(landsat_scene):
    """Acquisition date from an id like 'LC08_129050_20230227'."""
    return dt.datetime.strptime(landsat_scene.rsplit("_", 1)[-1], "%Y%m%d").date()


def season_doy(date, window):
    """(start, end) day of year of the +-window season around `date`; end < start when it wraps the new year."""
    doy = date.timetuple().tm_yday
    return (doy - window - 1) % 365 + 1, (doy + window - 1) % 365 + 1


def pair_grid(tif):
    """Earth Engine pixel grid of a pair raster: same CRS, transform and size, so outputs align pixel for pixel."""
    with rasterio.open(tif) as s:
        if s.crs is None:
            raise ValueError(f"{tif} has no readable CRS (PROJ database problem?)")
        t = s.transform
        return {"dimensions": {"width": s.width, "height": s.height},
                "affineTransform": {"scaleX": t.a, "shearX": t.b, "translateX": t.c,
                                    "shearY": t.d, "scaleY": t.e, "translateY": t.f},
                "crsCode": s.crs.to_string()}, s.profile


def context_image(ee, region, date, cfg):
    """ee.Image with bands lst_context_c (median LST, deg C) and n_obs (clear observations per pixel)."""
    start_doy, end_doy = season_doy(date, cfg["doy_window"])
    t = ee.Date(date.isoformat())
    ex = cfg["exclude_days"]
    col = ee.ImageCollection([])
    for name in cfg["collections"]:
        col = col.merge(ee.ImageCollection(name))
    col = (col.filterBounds(region)
           .filterDate(cfg["start"], cfg["end"])
           .filter(ee.Filter.calendarRange(start_doy, end_doy, "day_of_year"))
           .filter(ee.Filter.lt("CLOUD_COVER", cfg["max_scene_cloud"]))
           .filter(ee.Filter.Or(ee.Filter.date("1970-01-01", t.advance(-ex, "day")),
                                ee.Filter.date(t.advance(ex + 1, "day"), "2100-01-01"))))

    def lst(img):
        clear = img.select("QA_PIXEL").bitwiseAnd(QA_BAD_BITS).eq(0)
        return (img.select("ST_B10").multiply(ST_SCALE).add(ST_OFFSET).subtract(273.15)
                .rename("lst").updateMask(clear))

    stack = col.map(lst)
    native = ee.Image(col.first()).select("ST_B10").projection()
    out = ee.Image.cat(stack.median().rename("lst_context_c"), stack.count().rename("n_obs").toFloat())
    return out.setDefaultProjection(native), col.size()


def fetch(ee, image, grid, res_m, tries=4):
    """Pixels of `image` on `grid` as (2, H, W) float32 with NaN where no data. Coarser grids average the 30 m
    Landsat pixels; 30 m grids interpolate bilinearly."""
    img = image.reduceResolution(ee.Reducer.mean(), maxPixels=1024) if res_m > 45 else image.resample("bilinear")
    img = img.unmask(NODATA)
    for k in range(tries):
        try:
            arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": grid,
                                         "bandIds": BANDS})
            out = np.stack([arr[b] for b in BANDS]).astype("float32")
            out[out == NODATA] = np.nan
            out[1][np.isnan(out[0])] = 0
            return out
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(15 * (k + 1))


def write(path, data, profile):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {k: profile[k] for k in ("crs", "transform", "width", "height")}
    prof.update(driver="GTiff", count=len(BANDS), dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **prof) as d:
        d.write(data)
        d.descriptions = tuple(BANDS)


def qc(context, landsat_tif):
    """How the layer relates to the pair's own (target) LST: coverage, scenes, bias and spatial correlation."""
    with rasterio.open(landsat_tif) as s:
        target = s.read(list(s.descriptions).index("LST_K") + 1) - 273.15
    ok = np.isfinite(context[0]) & np.isfinite(target)
    r = float(np.corrcoef(context[0][ok], target[ok])[0, 1]) if ok.sum() > 10 else np.nan
    return dict(coverage=float(np.isfinite(context[0]).mean()), median_n_obs=float(np.median(context[1])),
                context_mean_c=float(np.nanmean(context[0])), target_mean_c=float(np.nanmean(target)),
                r_with_target=r)
