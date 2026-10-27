"""Spatial harmonisation: aggregate a pair's grid by an integer factor (30 m -> 60 m) on its own CRS."""
import numpy as np
from affine import Affine


def factor(res_m, grid_m):
    """Integer block size taking `res_m` pixels to about `grid_m` (1 = keep)."""
    f = int(round(grid_m / res_m))
    if f < 1 or abs(f * res_m - grid_m) > 0.1 * grid_m:
        raise ValueError(f"{res_m} m does not aggregate to {grid_m} m by an integer factor")
    return f


def _blocks(a, f):
    h, w = a.shape[-2] // f * f, a.shape[-1] // f * f
    a = a[..., :h, :w]
    return a.reshape(a.shape[:-2] + (h // f, f, w // f, f))


def block_mean(a, f, min_frac=0.75):
    """Mean of f x f blocks ignoring NaN; NaN where fewer than `min_frac` of the block is valid."""
    if f == 1:
        return a.astype("float32")
    b = _blocks(a.astype("float32"), f)
    n = np.isfinite(b).sum(axis=(-3, -1))
    with np.errstate(invalid="ignore"):
        m = np.nansum(b, axis=(-3, -1)) / n
    return np.where(n >= min_frac * f * f, m, np.nan).astype("float32")


def block_mode(a, f, nodata=0):
    """Most frequent class of each f x f block of a 2-D categorical layer; ties go to the smallest code."""
    if f == 1:
        return a
    b = _blocks(a, f)
    b = b.transpose(0, 2, 1, 3).reshape(b.shape[0], b.shape[2], f * f)
    out = np.full(b.shape[:2], nodata, a.dtype)
    best = np.zeros(b.shape[:2], int)
    for code in np.unique(b):
        if code == nodata:
            continue
        cnt = (b == code).sum(-1)
        better = cnt > best
        out[better], best[better] = code, cnt[better]
    return out


def transform(t, f):
    """Affine of the aggregated grid."""
    return t * Affine.scale(f, f)
