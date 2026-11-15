"""What-if engine: apply a contracts.Scenario to a scene, predict, and write delta-LST maps (CONTRACTS sections 3, 5).

Results are modelled estimates of the surface (not air) temperature response, not causal guarantees.
"""
import json

import h5py
import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from urbanheat.contracts import ENV_INPUTS, LEVERS, LS_INPUTS
from urbanheat.models.train import KEYS

# where each lever sits in the model inputs; albedo is a contract lever but not a CNN input
LEVER_CH = {"NDVI": ("ls", LS_INPUTS.index("NDVI")), "MNDWI": ("ls", LS_INPUTS.index("MNDWI")),
            "built_share": ("env", ENV_INPUTS.index("built_share"))}
NOTE = "Modelled estimate of land surface temperature change; not a causal guarantee."


def apply(scene, scenario, norm, where=None):
    """Copy of a normalised scene with the scenario's additive changes made in physical units and clipped to the
    lever's range, on valid pixels (and inside `where` if given). Only the lever channels change.
    shortcut: linked inputs (red/NIR, hyperspectral bands, NDBI) stay as they are; couple levers in one scenario,
    e.g. {"NDVI": +0.1, "built_share": -0.1}, until a physically consistent input model exists."""
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in scene.items()}
    sel = scene["mask"] if where is None else scene["mask"] & where
    for lever, delta in scenario.changes.items():
        if lever not in LEVER_CH:
            raise ValueError(f"{lever} is not an input of the CNN, so the model cannot respond to it")
        k, ch = LEVER_CH[lever]
        mean, std = norm[k + "_mean"][ch], norm[k + "_std"][ch]
        raw = out[k][ch] * std + mean
        out[k][ch] = np.where(sel, ((raw + delta).clip(*LEVERS[lever]) - mean) / std, out[k][ch])
    return out


@torch.no_grad()
def _predict(model, scene, device):
    b = {k: torch.as_tensor(np.asarray(scene[k]))[None].to(device) for k in KEYS}
    return model(b)[0].float().cpu().numpy()


def region_mask(geojson_path, h5_path, pid):
    """(H, W) bool: pixels of pair `pid` inside the polygons of a GeoJSON file (EPSG:4326, as GeoJSON requires)."""
    with h5py.File(h5_path, "r") as f:
        g = f["pairs"][pid]
        shape, transform, crs = g["valid"].shape, Affine(*g.attrs["transform"][:6]), g.attrs["crs"]
    with open(geojson_path) as fh:
        gj = json.load(fh)
    geoms = [ft["geometry"] for ft in gj["features"]] if "features" in gj else [gj.get("geometry", gj)]
    geoms = [transform_geom("EPSG:4326", crs, g) for g in geoms]
    return rasterize(geoms, out_shape=shape, transform=transform, dtype="uint8").astype(bool)


def delta_lst(model, ds, pid, scenario, device="cpu", where=None):
    """(H, W) predicted LST(scenario) - LST(current) in deg C, NaN on invalid pixels.
    The target is an anomaly from the scene median, so change a region (scenario.region and/or a `where` mask)
    and read the result against the unchanged rest; a whole-scene change has no meaningful mean."""
    model.eval()
    s = ds.scene(pid)
    if scenario.region:
        r = region_mask(scenario.region, ds.path, pid)
        where = r if where is None else where & r
    d = _predict(model, apply(s, scenario, ds.norm, where), device) - _predict(model, s, device)
    return np.where(s["mask"], d, np.nan).astype("float32")


def write_geotiff(path, dlst, h5_path, pid, scenario):
    """Single-band float32 GeoTIFF on the pair's grid, scenario and caveat in the tags.
    shortcut: tiled GTiff, not a true COG; convert with `rio cogeo` if the dashboard needs COG."""
    with h5py.File(h5_path, "r") as f:
        g = f["pairs"][pid]
        transform, crs = Affine(*g.attrs["transform"][:6]), g.attrs["crs"]
    with rasterio.open(path, "w", driver="GTiff", height=dlst.shape[0], width=dlst.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(dlst, 1)
        dst.set_band_description(1, "delta_lst_c")
        dst.update_tags(pair_id=pid, scenario=scenario.name, changes=json.dumps(scenario.changes), note=NOTE)
    return path
