import datetime as dt

import numpy as np
import rasterio
from rasterio.transform import from_origin

from urbanheat.cli import rasterio_bounds
from urbanheat.data import thermal_context as tc


def test_landsat_date():
    assert tc.landsat_date("LC08_129050_20230227") == dt.date(2023, 2, 27)


def test_season_doy_wraps_the_new_year():
    assert tc.season_doy(dt.date(2023, 7, 1), 60) == (122, 242)
    start, end = tc.season_doy(dt.date(2023, 1, 10), 60)
    assert end < start and (start, end) == (315, 70)


def _tif(path, crs="EPSG:32643"):
    t = from_origin(500000, 1400000, 30, 30)
    with rasterio.open(path, "w", driver="GTiff", width=4, height=3, count=7, dtype="float32", crs=crs, transform=t) as d:
        d.write(np.full((7, 3, 4), 300.0, "float32"))
        d.descriptions = ("blue", "green", "red", "nir", "swir1", "swir2", "LST_K")
    return path


def test_pair_grid_and_bounds(tmp_path):
    grid, profile = tc.pair_grid(_tif(tmp_path / "l.tif"))
    assert grid["crsCode"] == "EPSG:32643"
    assert grid["dimensions"] == {"width": 4, "height": 3}
    assert grid["affineTransform"]["scaleY"] == -30
    assert rasterio_bounds(profile) == (500000, 1399910, 500120, 1400000)


def test_write_and_qc(tmp_path):
    landsat = _tif(tmp_path / "l.tif")
    _, profile = tc.pair_grid(landsat)
    ctx = np.stack([np.arange(12, dtype="float32").reshape(3, 4), np.full((3, 4), 5, "float32")])
    ctx[0, 0, 0] = np.nan
    tc.write(tmp_path / "out" / "thermal_context.tif", ctx, profile)
    with rasterio.open(tmp_path / "out" / "thermal_context.tif") as s:
        assert s.descriptions == tuple(tc.BANDS) and s.crs.to_epsg() == 32643
        np.testing.assert_array_equal(s.read(), ctx)
    q = tc.qc(ctx, landsat)
    assert q["coverage"] == 11 / 12 and q["median_n_obs"] == 5
    assert abs(q["target_mean_c"] - 26.85) < 1e-3
