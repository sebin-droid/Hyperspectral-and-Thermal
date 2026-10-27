"""Yearly land cover and surface heat per cell from Earth Engine.

Per cell (the ~7.7 km square around each cell centre) and year:
  - green / built / water / bare area (ha) from Dynamic World V1, the year's most frequent class per 10 m pixel
  - Landsat 8/9 hot-season LST (median of clear pixels): over the cell's built-up pixels, over the whole cell, and
    over a rural ring around the cell (non-built, non-water pixels 1-6 km outside)
  - surface urban heat island = built-up LST minus rural-ring LST. Using the difference removes the region's
    year-to-year weather and climate warming, which affect city and countryside alike.
The hot season is the cell's 3 consecutive warmest months, from its 2016-2025 monthly Landsat LST.
"""
import time

import numpy as np

DW = "GOOGLE/DYNAMICWORLD/V1"
LANDSAT = ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]
# Dynamic World classes: 0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops, 5 shrub & scrub, 6 built, 7 bare, 8 snow
COVER = {"green": [1, 2, 3, 4, 5], "built": [6], "water": [0], "bare": [7]}
QA_BAD_BITS = 0b11111
NODATA = -9999


def regions(ee, lon, lat, cell_m, ring_inner_m, ring_outer_m, max_error=10):
    """(cell, rural ring, outer box containing both)."""
    cell = ee.Geometry.Point([lon, lat]).buffer(cell_m / 2, max_error).bounds(max_error)
    outer = cell.buffer(ring_outer_m, max_error)
    ring = outer.difference(cell.buffer(ring_inner_m, max_error), max_error)
    return cell, ring, outer.bounds(max_error)


def landsat_lst(ee, region, start, end, max_cloud=70):
    col = ee.ImageCollection(LANDSAT[0]).merge(ee.ImageCollection(LANDSAT[1]))
    col = col.filterBounds(region).filterDate(start, end).filter(ee.Filter.lt("CLOUD_COVER", max_cloud))

    def lst(img):
        clear = img.select("QA_PIXEL").bitwiseAnd(QA_BAD_BITS).eq(0)
        out = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("lst").updateMask(clear)
        return ee.Image(out.copyProperties(img, ["system:time_start"]))     # arithmetic drops the timestamp

    return col.map(lst), col.size()


def hot_season(monthly):
    """Start month (1-12) of the 3 consecutive warmest months from 12 monthly means (None = no data)."""
    m = np.array([np.nan if v is None else v for v in monthly], float)
    if np.isnan(m).all():
        return None
    fill = np.where(np.isnan(m), np.nanmean(m), m)
    three = [fill[i] + fill[(i + 1) % 12] + fill[(i + 2) % 12] for i in range(12)]
    return int(np.argmax(three)) + 1


def monthly_lst(ee, cell, years):
    """Mean LST of the cell for each calendar month over `years` (for the hot season)."""
    stack, _ = landsat_lst(ee, cell, f"{years[0]}-01-01", f"{years[-1] + 1}-01-01")
    vals = ee.List.sequence(1, 12).map(lambda m: stack.filter(ee.Filter.calendarRange(m, m, "month")).median()
                                       .reduceRegion(ee.Reducer.mean(), cell, 90, maxPixels=1e9).get("lst", NODATA))
    return [None if v is None or v == NODATA else v for v in vals.getInfo()]


def season_window(year, start_month):
    """(start, end) dates of the 3-month hot season starting in `start_month` of `year` (may run into next year)."""
    end_month = start_month + 3
    end_year = year + (end_month - 1) // 12
    return f"{year}-{start_month:02d}-01", f"{end_year}-{(end_month - 1) % 12 + 1:02d}-01"


def cell_year(ee, cell, ring, outer, year, start_month):
    """ee.Dictionary of one cell-year's areas and LSTs."""
    dw = ee.ImageCollection(DW).filterBounds(outer).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    label = dw.select("label").mode()
    ha = ee.Image.pixelArea().divide(1e4)
    cover = ee.Image.cat([ha.updateMask(label.remap(codes, [1] * len(codes), 0)).rename(name)
                          for name, codes in COVER.items()])
    areas = cover.reduceRegion(ee.Reducer.sum(), cell, 10, maxPixels=1e10)

    s, e = season_window(year, start_month)
    stack, n = landsat_lst(ee, outer, s, e)
    empty = ee.Image.constant(0).rename("lst").updateMask(0)          # season without any Landsat scene
    lst = ee.Image(ee.Algorithms.If(n.gt(0), stack.median(), empty))
    built = label.eq(6)
    rural = label.neq(6).And(label.neq(0))
    # a season without any clear Landsat scene gives an image without bands -> NODATA instead of failing
    mean = lambda img, geom: ee.List([img.reduceRegion(ee.Reducer.mean(), geom, 30, maxPixels=1e10).get("lst", NODATA),
                                      NODATA]).reduce(ee.Reducer.firstNonNull())          # missing or null -> NODATA
    return (areas.set("year", year).set("n_scenes", n).set("dw_images", dw.size())
            .set("lst_built_c", mean(lst.updateMask(built), cell))
            .set("lst_cell_c", mean(lst, cell))
            .set("lst_rural_c", mean(lst.updateMask(rural), ring)))


def cell_series(ee, lon, lat, years, ccfg):
    """List of dicts, one per year, for the cell at (lon, lat)."""
    cell, ring, outer = regions(ee, lon, lat, ccfg["cell_m"], ccfg["ring_inner_m"], ccfg["ring_outer_m"])
    start = hot_season(monthly_lst(ee, cell, years))
    if start is None:
        return [], None
    rows = [_retry(lambda y=y: cell_year(ee, cell, ring, outer, y, start).getInfo()) for y in years]
    rows = [{k: (None if v == NODATA else v) for k, v in r.items()} for r in rows]
    return rows, start


TRANSITIONS = {1: "green_stable", 2: "green_to_built", 3: "built_stable", 4: "water_stable", 5: "water_to_other",
               6: "built_to_green"}


def _period(ee, region, years, start_month):
    """Dynamic World mode label and hot-season median LST over a run of years."""
    label = (ee.ImageCollection(DW).filterBounds(region)
             .filterDate(f"{years[0]}-01-01", f"{years[-1] + 1}-01-01").select("label").mode())
    stacks = [landsat_lst(ee, region, *season_window(y, start_month))[0] for y in years]
    col = stacks[0]
    for s in stacks[1:]:
        col = col.merge(s)
    return label, col.median()


def transition_warming(ee, lon, lat, start_month, early, late, ccfg):
    """Mean LST change (late - early, hot season) per land-cover transition inside the cell, and pixel counts.

    Comparing e.g. green_to_built with green_stable in the same cell and period is a difference-in-differences:
    the regional weather / climate trend affects both and cancels.
    """
    cell, _, _ = regions(ee, lon, lat, ccfg["cell_m"], ccfg["ring_inner_m"], ccfg["ring_outer_m"])
    l0, t0 = _period(ee, cell, early, start_month)
    l1, t1 = _period(ee, cell, late, start_month)
    green = lambda l: l.remap(COVER["green"], [1] * len(COVER["green"]), 0)
    g0, g1, b0, b1, w0, w1 = green(l0), green(l1), l0.eq(6), l1.eq(6), l0.eq(0), l1.eq(0)
    cls = (ee.Image(0).where(g0.And(g1), 1).where(g0.And(b1), 2).where(b0.And(b1), 3)
           .where(w0.And(w1), 4).where(w0.And(w1.Not()), 5).where(b0.And(g1), 6)).rename("cls")
    dlst = t1.subtract(t0).rename("dlst")
    stats = dlst.addBands(cls).updateMask(cls.gt(0)).reduceRegion(
        ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True).group(groupField=1, groupName="cls"),
        cell, 30, maxPixels=1e10)
    out = {}
    for g in _retry(lambda: stats.getInfo()).get("groups", []):
        name = TRANSITIONS[int(g["cls"])]
        out[f"dlst_{name}"], out[f"n_{name}"] = g["mean"], g["count"]
    return out


def _retry(fn, tries=5):
    # one year per request; Earth Engine refuses too many parallel aggregations, so wait and retry
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1 or "concurrent" not in str(e).lower():
                raise
            time.sleep(20 * (k + 1))
