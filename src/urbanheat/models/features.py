"""Named per-pixel features from the dataset's pixel table, and the feature sets the baselines compare."""
import h5py
import numpy as np
import pandas as pd

from urbanheat.contracts import ENV_INPUTS, LS_INPUTS, MET_INPUTS, SENSORS
from urbanheat.data.dataset import load_pixels


def _band(hs, wl, nm):
    return hs[:, int(np.argmin(abs(wl - nm)))]


def hs_indices(hs, wl):
    """Physically meaningful hyperspectral features (only possible with narrow bands)."""
    b = lambda nm: _band(hs, wl, nm)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "albedo_hs": hs.mean(axis=1),                                     # broadband brightness (cool roofs)
            "ndre_hs": (b(750) - b(710)) / (b(750) + b(710)),                 # red-edge: vegetation vigour
            "ndwi_hs": (b(860) - b(1240)) / (b(860) + b(1240)),               # leaf / surface water content
            "swir2200_depth_hs": 1 - b(2200) / ((b(2160) + b(2240)) / 2),     # clay / mineral / concrete absorption
        }


def table(path):
    """(X, info): X has every feature column; info has y_abs, y_anom, pair_id, split, sensor, climate_group."""
    px = load_pixels(path)
    with h5py.File(path, "r") as f:
        wl = f.attrs["wavelengths_nm"][:]
        climate = {k: g.attrs["climate_group"] for k, g in f["pairs"].items()}
    X = pd.DataFrame(px["ls"], columns=LS_INPUTS)
    X[ENV_INPUTS] = px["env"]
    X[MET_INPUTS] = px["met"]
    X["lst_context_c"] = px["thermal"][:, 0]
    for k, v in hs_indices(px["hs"], wl).items():
        X[k] = v
    hs_cols = [f"hs_{w:.0f}" for w in wl]
    X = pd.concat([X, pd.DataFrame(px["hs"], columns=hs_cols)], axis=1)

    info = pd.DataFrame({"pair_id": px["pair_id"], "split": px["split"], "y_abs": px["lst_c"],
                         "y_anom": px["lst_anom"]})
    info["sensor"] = info.pair_id.str.split("_").str[0]
    info["climate_group"] = info.pair_id.map(climate)
    for s in SENSORS:
        X[f"is_{s}"] = (info.sensor == s).astype("float32")
    # relative to the pair: the anomaly target removes each scene's level, so do the same for level-like inputs
    med = X.groupby(info.pair_id)[["elevation_m", "lst_context_c"]].transform("median")
    X["elevation_rel_m"] = X.elevation_m - med.elevation_m
    X["thermal_anom_c"] = X.lst_context_c - med.lst_context_c
    return X.astype("float32"), info, hs_cols


def pair_table(path, pair_id):
    """Same features for every valid pixel of one pair (for maps). Returns (X, y_anom, valid mask)."""
    with h5py.File(path, "r") as f:
        wl = f.attrs["wavelengths_nm"][:]
        g = f["pairs"][pair_id]
        v = g["valid"][:].astype(bool)
        hs = g["hs"][:][:, v].T.astype("float32")
        X = pd.DataFrame(g["ls"][:][:, v].T, columns=LS_INPUTS)
        X[ENV_INPUTS] = g["env"][:][:, v].T
        X[MET_INPUTS] = np.tile(g["met"][:], (v.sum(), 1))
        X["lst_context_c"] = g["thermal"][0][v]
        y = g["lst_c"][:][v] - g.attrs["lst_median_c"]
        sensor = g.attrs["sensor"]
    for k, val in hs_indices(hs, wl).items():
        X[k] = val
    X = pd.concat([X, pd.DataFrame(hs, columns=[f"hs_{w:.0f}" for w in wl])], axis=1)
    for s in SENSORS:
        X[f"is_{s}"] = float(s == sensor)
    X["elevation_rel_m"] = X.elevation_m - X.elevation_m.median()
    X["thermal_anom_c"] = X.lst_context_c - X.lst_context_c.median()
    return X.astype("float32"), y, v


SENSOR_COLS = [f"is_{s}" for s in SENSORS]
LANDSAT = LS_INPUTS + ["built_share", "elevation_rel_m", "slope_deg"] + SENSOR_COLS
HS_INDICES = ["albedo_hs", "ndre_hs", "ndwi_hs", "swir2200_depth_hs"]


def feature_sets(hs_cols):
    """Feature sets for the anomaly target, from Landsat only to everything (the hyperspectral ablation)."""
    return {
        "landsat": LANDSAT,
        "landsat+hs_indices": LANDSAT + HS_INDICES,
        "landsat+hs_bands": LANDSAT + HS_INDICES + hs_cols,
        "thermal_only": ["thermal_anom_c"] + SENSOR_COLS,
        "landsat+thermal": LANDSAT + ["thermal_anom_c"],
        "all": LANDSAT + HS_INDICES + hs_cols + ["thermal_anom_c"],
    }


def absolute_set(hs_cols):
    """For the absolute-LST target: everything plus the day's weather and the raw context level."""
    return LANDSAT + HS_INDICES + hs_cols + ["elevation_m", "lst_context_c"] + MET_INPUTS


# The physical factors whose influence the driver analysis ranks (no thermal context: "usually hot" is not a cause)
DRIVERS = ["NDVI", "NDBI", "MNDWI", "built_share", "albedo_hs", "ndre_hs", "ndwi_hs", "swir2200_depth_hs",
           "elevation_rel_m", "slope_deg"] + SENSOR_COLS
