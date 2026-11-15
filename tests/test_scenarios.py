import json

import numpy as np
import pytest
import rasterio
import torch
from rasterio.warp import transform as warp

from conftest import TRANSFORM
from urbanheat.contracts import Scenario
from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient.distill import Prepared
from urbanheat.scenarios import engine


class NdviModel(torch.nn.Module):
    """Prediction = 2 x normalised NDVI, so delta-LST is known exactly."""

    def forward(self, b):
        return 2 * b["ls"][:, 6]


@pytest.fixture
def ds(fake_h5):
    return Prepared(PatchDataset(fake_h5, "test"))


def test_delta_follows_the_lever_in_physical_units(ds):
    pid = ds.pairs[0]
    d = engine.delta_lst(NdviModel(), ds, pid, Scenario("greening", {"NDVI": 0.1}))
    std = ds.norm["ls_std"][6]
    raw = ds.scene(pid)["ls"][6] * std + ds.norm["ls_mean"][6]
    room = ds.scene(pid)["mask"] & (raw < 0.9)
    np.testing.assert_allclose(d[room], 2 * 0.1 / std, rtol=1e-4)
    assert np.isnan(d[~ds.scene(pid)["mask"]]).all()
    assert np.nanmax(d) <= 2 * 0.1 / std + 1e-4                     # NDVI capped at 1


def test_albedo_is_rejected_because_the_cnn_does_not_read_it(ds):
    with pytest.raises(ValueError, match="not an input"):
        engine.delta_lst(NdviModel(), ds, ds.pairs[0], Scenario("cool roofs", {"albedo": 0.1}))


def test_region_limits_the_change_and_geotiff_round_trips(ds, tmp_path):
    x0, y0 = TRANSFORM[2], TRANSFORM[5]
    xs, ys = [x0, x0 + 20 * 60, x0 + 20 * 60, x0], [y0, y0, y0 - 40 * 60, y0 - 40 * 60]   # left half
    lon, lat = warp("EPSG:32643", "EPSG:4326", xs, ys)
    gj = tmp_path / "left.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                  "geometry": {"type": "Polygon", "coordinates": [list(map(list, zip(lon, lat))) + [[lon[0], lat[0]]]]}}]}))
    pid = ds.pairs[0]
    sc = Scenario("park", {"NDVI": 0.1}, region=str(gj))
    d = engine.delta_lst(NdviModel(), ds, pid, sc)
    assert np.nanmax(np.abs(d[:, 22:])) == 0 and np.nanmedian(d[:, :18]) > 0
    out = engine.write_geotiff(tmp_path / "d.tif", d, ds.path, pid, sc)
    with rasterio.open(out) as r:
        assert r.crs.to_epsg() == 32643 and r.transform.c == x0 and r.tags()["scenario"] == "park"
        np.testing.assert_array_equal(r.read(1), d)
