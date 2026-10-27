"""Spectral harmonisation: every sensor's bands -> one common wavelength grid by linear interpolation."""
import re

import numpy as np


def wavelengths(descriptions):
    """Centre wavelengths (nm) from band descriptions like '381 nm (EMIT b000)' or '434.7 nm (EnMAP b004)'."""
    return np.array([float(re.match(r"\s*([\d.]+)\s*nm", d).group(1)) for d in descriptions])


def interp_matrix(src_wl, tgt_wl, max_gap):
    """(T, S) weights: target band t = weights[t] @ source bands, linear between the two neighbouring source
    bands. A target is covered only if those neighbours are at most `max_gap` nm apart (rows of NaN otherwise)."""
    src_wl = np.asarray(src_wl, float)
    order = np.argsort(src_wl)
    s = src_wl[order]
    w = np.full((len(tgt_wl), len(s)), np.nan)
    for i, t in enumerate(tgt_wl):
        j = np.searchsorted(s, t)
        if j < len(s) and s[j] == t:
            w[i] = 0.0
            w[i, j] = 1.0
        elif 0 < j < len(s) and s[j] - s[j - 1] <= max_gap:
            w[i] = 0.0
            w[i, j - 1] = (s[j] - t) / (s[j] - s[j - 1])
            w[i, j] = (t - s[j - 1]) / (s[j] - s[j - 1])
    out = np.empty_like(w)
    out[:, order] = w
    return out


def common_grid(sensor_wavelengths, wl_min, wl_max, step, max_gap):
    """Target wavelengths on a regular grid that every sensor covers."""
    grid = np.arange(wl_min, wl_max + step / 2, step, dtype=float)
    ok = np.ones(len(grid), bool)
    for wl in sensor_wavelengths:
        ok &= np.isfinite(interp_matrix(wl, grid, max_gap)).all(axis=1)
    return grid[ok]


def apply(weights, cube):
    """(S, H, W) source cube -> (T, H, W). NaN in a used source band gives NaN in the target band."""
    s, h, w = cube.shape
    flat = cube.reshape(s, -1)
    out = np.zeros((weights.shape[0], flat.shape[1]), "float32")
    for i, row in enumerate(weights):
        used = np.nonzero(row)[0]
        out[i] = (row[used, None] * flat[used]).sum(axis=0)
    return out.reshape(-1, h, w)
