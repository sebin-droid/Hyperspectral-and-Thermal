"""Why the heat CNN predicts what it does: integrated gradients per input group, and Grad-CAM per branch.

Integrated gradients (IG) are the Aumann-Shapley value of the input path from a baseline to the scene, i.e. SHAP
for a smooth model. The baseline is the all-zero normalised input: the train mean everywhere (the city median for
the relative channels thermal context and elevation). The attribution target is the summed predicted anomaly over
the valid pixels, so completeness holds:  sum(attr) = sum(pred(scene)) - sum(pred(baseline))  in deg C, and an
input pixel's attribution is its contribution to the predictions around it (receptive field 29 px, ~1.7 km).
"""
import h5py
import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine

from urbanheat.contracts import ENV_INPUTS, LS_INPUTS

INPUT_KEYS = ["hs", "ls", "env", "thermal"]
# input groups a person can reason about: (key, channels)
GROUPS = {
    "hyperspectral": ("hs", slice(None)),
    "landsat bands": ("ls", slice(0, 6)),
    "NDVI": ("ls", [LS_INPUTS.index("NDVI")]),
    "NDBI": ("ls", [LS_INPUTS.index("NDBI")]),
    "MNDWI": ("ls", [LS_INPUTS.index("MNDWI")]),
    "elevation": ("env", [ENV_INPUTS.index("elevation_m")]),
    "slope": ("env", [ENV_INPUTS.index("slope_deg")]),
    "built_share": ("env", [ENV_INPUTS.index("built_share")]),
    "thermal context": ("thermal", [0]),
}
CAVEAT = "Attributions describe what the model learned from data (associations), not physical causes."


def _batch(scene, device):
    return {k: torch.as_tensor(np.asarray(scene[k]))[None].to(device) for k in INPUT_KEYS}


def integrated_gradients(model, scene, device="cpu", steps=32, chunk=8):
    """{key: (C, H, W)} IG attribution in deg C of each input value to the summed prediction over valid pixels.
    Keys the model does not read are left out. Midpoint rule over `steps` points of the path."""
    model.eval()
    x = _batch(scene, device)
    sensor = torch.as_tensor([scene["sensor"]], device=device)
    mask = torch.as_tensor(scene["mask"], device=device)
    grads = {k: torch.zeros_like(v) for k, v in x.items()}
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    for a in alphas.split(chunk):
        b = {k: (a.view(-1, 1, 1, 1) * v).requires_grad_() for k, v in x.items()}
        b["sensor"] = sensor.expand(len(a))
        model(b)[:, mask].sum().backward()
        for k in grads:
            if b[k].grad is not None:
                grads[k] += b[k].grad.sum(0, keepdim=True)
    used = [k for k in grads if grads[k].abs().sum() > 0]
    return {k: (x[k] * grads[k] / steps)[0].detach().cpu().numpy() for k in used}


def by_group(attr):
    """{group: (H, W)} attribution summed over the group's channels (deg C per input pixel)."""
    return {g: attr[k][ch].sum(0) for g, (k, ch) in GROUPS.items() if k in attr}


def baseline_sum(model, scene, device="cpu"):
    """Summed prediction over valid pixels for the baseline input (completeness check: IG total = pred - this)."""
    b = {k: torch.zeros_like(v) for k, v in _batch(scene, device).items()}
    b["sensor"] = torch.as_tensor([scene["sensor"]], device=device)
    with torch.no_grad():
        return float(model(b)[0][torch.as_tensor(scene["mask"], device=device)].sum())


def scene_summary(groups, scene, where=None):
    """Per group: mean |attribution| and mean signed attribution (deg C per pixel) over valid pixels (in `where`),
    and direction: slope of attribution on the normalised input through the baseline, sum(a x) / sum(x^2), i.e. the
    x^2-weighted mean path gradient in deg C per +1 std at a pixel (> 0: a higher value makes the area hotter).
    A centred correlation is not used: it mixes in the spatial pattern and can flip the sign."""
    sel = scene["mask"] if where is None else scene["mask"] & where
    rows = []
    for g, a in groups.items():
        k, ch = GROUPS[g]
        v = a[sel]
        direction = np.nan
        if isinstance(ch, list) and sel.any():
            x = np.asarray(scene[k])[ch[0]][sel]
            if (x * x).sum() > 0:
                direction = float((x * v).sum() / (x * x).sum())
        rows.append(dict(feature=g, mean_abs_shap=float(np.abs(v).mean()), mean_shap=float(v.mean()),
                         direction=direction, n=int(sel.sum())))
    return pd.DataFrame(rows)


def global_importance(model, ds, info_of, device="cpu", steps=32):
    """CONTRACTS section 4 table over all scenes of `ds`: feature, mean_abs_shap, direction, climate_group
    (pixel-weighted means per climate and for "all"). Also returns {pid: {group: (H, W)}}."""
    per, maps = [], {}
    for pid in ds.pairs:
        s = ds.scene(pid)
        maps[pid] = by_group(integrated_gradients(model, s, device, steps))
        per.append(scene_summary(maps[pid], s).assign(pair_id=pid, climate_group=info_of[pid]["climate_group"]))
    per = pd.concat(per, ignore_index=True)

    def agg(d):
        w = d["n"] / d["n"].sum()
        ok = d["direction"].notna()
        return pd.Series(dict(mean_abs_shap=(d["mean_abs_shap"] * w).sum(), mean_shap=(d["mean_shap"] * w).sum(),
                              direction=np.average(d.loc[ok, "direction"], weights=d.loc[ok, "n"]) if ok.any() else np.nan,
                              n=d["n"].sum()))

    cols = ["mean_abs_shap", "mean_shap", "direction", "n"]
    out = pd.concat([per.assign(climate_group="all"), per]).groupby(["climate_group", "feature"])[cols].apply(agg)
    out = out.reset_index().sort_values(["climate_group", "mean_abs_shap"], ascending=[True, False])
    return out[["feature", "mean_abs_shap", "direction", "climate_group", "mean_shap", "n"]], maps


def grad_cam(model, scene, where=None, device="cpu"):
    """{branch: (H, W) in [0, 1]}: Grad-CAM of the surface and thermal branches for the mean predicted anomaly
    over `where` (default: the hottest 10 % of predicted valid pixels). Shows which places the model looked at."""
    model.eval()
    feats = {}
    hooks = [getattr(model, n).register_forward_hook(lambda m, i, o, n=n: feats.__setitem__(n, o))
             for n in ("surface", "thermal") if hasattr(model, n)]
    b = _batch(scene, device)
    b["sensor"] = torch.as_tensor([scene["sensor"]], device=device)
    try:
        pred = model(b)[0]
    finally:
        for h in hooks:
            h.remove()
    mask = torch.as_tensor(scene["mask"], device=device)
    if where is None:
        p = pred.detach()[mask]
        where_t = mask & (pred.detach() >= torch.quantile(p, 0.9))
    else:
        where_t = mask & torch.as_tensor(where, device=device)
    if not where_t.any():
        raise ValueError("no valid pixels in the target region")
    for f in feats.values():
        f.retain_grad()
    pred[where_t].mean().backward()
    out = {}
    for n, f in feats.items():
        w = f.grad[0].mean((1, 2), keepdim=True)                    # channel weights: spatially averaged gradient
        cam = torch.relu((w * f[0]).sum(0)).detach().cpu().numpy()
        cam = np.where(scene["mask"], cam, 0.0)
        out[n] = (cam / cam.max() if cam.max() > 0 else cam).astype("float32")
    return out


def pair_grid(h5_path, pid):
    """(transform, crs, (H, W)) of a pair."""
    with h5py.File(h5_path, "r") as f:
        g = f["pairs"][pid]
        return Affine(*g.attrs["transform"][:6]), g.attrs["crs"], g["valid"].shape


def write_bands(path, bands, h5_path, pid, mask=None, **tags):
    """Multi-band float32 GeoTIFF on the pair's grid, one named band per entry of `bands` ({name: (H, W)}),
    NaN outside `mask`. Used for <pair_id>_attr.tif (CONTRACTS section 4) and prediction maps (section 5)."""
    transform, crs, _ = pair_grid(h5_path, pid)
    arr = np.stack([np.asarray(v, "float32") for v in bands.values()])
    if mask is not None:
        arr[:, ~mask] = np.nan
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=len(arr),
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan, compress="deflate",
                       tiled=True) as dst:
        dst.write(arr)
        for i, name in enumerate(bands, 1):
            dst.set_band_description(i, name)
        dst.update_tags(pair_id=pid, **{k: str(v) for k, v in tags.items()})
    return path
