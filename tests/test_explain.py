import numpy as np
import pytest
import rasterio
import torch

from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient import lite
from urbanheat.efficient.distill import Prepared
from urbanheat.explain import xai


@pytest.fixture
def ds(fake_h5):
    return Prepared(PatchDataset(fake_h5, "test"))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return lite.lite_cnn("fusion_cnn", width=8, thermal_width=4).eval()


def test_integrated_gradients_are_complete(ds, model):
    """Attributions add up to prediction(scene) - prediction(baseline), so they are in deg C."""
    s = ds.scene(ds.pairs[0])
    attr = xai.integrated_gradients(model, s, steps=64)
    with torch.no_grad():
        pred = model({**xai._batch(s, "cpu"), "sensor": torch.as_tensor([s["sensor"]])})[0].numpy()
    total = sum(a.sum() for a in attr.values())
    np.testing.assert_allclose(total, pred[s["mask"]].sum() - xai.baseline_sum(model, s), rtol=0.02, atol=0.05)
    assert set(xai.by_group(attr)) == set(xai.GROUPS)


def test_unused_inputs_get_no_attribution(ds):
    m = lite.lite_cnn("fusion_nohs_cnn", width=8, thermal_width=4).eval()
    assert "hs" not in xai.integrated_gradients(m, ds.scene(ds.pairs[0]), steps=4)


def test_direction_finds_a_known_sign():
    """A pixel-wise model hotter where NDVI is low: NDVI direction must be negative, built_share positive."""
    class Linear(torch.nn.Module):
        def forward(self, b):
            return -2 * b["ls"][:, 6] + 3 * b["env"][:, 2]

    rng = np.random.default_rng(0)
    s = dict(hs=np.zeros((122, 20, 20), "float32"), ls=rng.normal(size=(9, 20, 20)).astype("float32"),
             env=rng.normal(size=(3, 20, 20)).astype("float32"), thermal=np.zeros((1, 20, 20), "float32"),
             sensor=0, mask=np.ones((20, 20), bool))
    t = xai.scene_summary(xai.by_group(xai.integrated_gradients(Linear(), s, steps=8)), s).set_index("feature")
    np.testing.assert_allclose(t.loc[["NDVI", "built_share"], "direction"], [-2, 3], rtol=1e-3)   # the slopes


def test_global_importance_table_follows_the_contract(ds, model):
    info = {p: {"climate_group": "tropical"} for p in ds.pairs}
    table, maps = xai.global_importance(model, ds, info, steps=4)
    assert list(table.columns[:4]) == ["feature", "mean_abs_shap", "direction", "climate_group"]
    assert set(table["climate_group"]) == {"all", "tropical"}
    assert (table["mean_abs_shap"] >= 0).all() and set(maps) == set(ds.pairs)


def test_grad_cam_is_a_unit_map_per_branch(ds, model):
    s = ds.scene(ds.pairs[0])
    cam = xai.grad_cam(model, s)
    assert set(cam) == {"surface", "thermal"}
    for c in cam.values():
        assert c.shape == s["mask"].shape and c.min() >= 0 and c.max() <= 1
        assert (c[~s["mask"]] == 0).all()


def test_write_bands_names_each_band(ds, model, tmp_path):
    pid = ds.pairs[0]
    s = ds.scene(pid)
    groups = xai.by_group(xai.integrated_gradients(model, s, steps=4))
    p = xai.write_bands(tmp_path / "a.tif", groups, ds.path, pid, mask=s["mask"], note=xai.CAVEAT)
    with rasterio.open(p) as r:
        assert r.count == len(groups) and list(r.descriptions) == list(groups)
        assert np.isnan(r.read(1)[~s["mask"]]).all() and r.tags()["note"] == xai.CAVEAT


def test_explain_region_adds_up_to_the_region_mean(ds, model):
    s = ds.scene(ds.pairs[0])
    where = np.zeros_like(s["mask"])
    where[10:20, 12:22] = True
    where &= s["mask"]
    groups, base, pred = xai.explain_region(model, s, where, steps=64)
    with torch.no_grad():
        full = model({**xai._batch(s, "cpu"), "sensor": torch.as_tensor([s["sensor"]])})[0].numpy()
    np.testing.assert_allclose(pred, full[where].mean(), atol=1e-4)            # the crop does not change the prediction
    np.testing.assert_allclose(base + sum(groups.values()), pred, atol=0.02)
