"""Which surface factors drive the LST anomaly, and by how much: exact TreeSHAP from XGBoost (pred_contribs)."""
import numpy as np
import pandas as pd
import xgboost as xgb

# A sensible change of each lever, for effect sizes in deg C ("what if NDVI rose by 0.1?")
STEPS = {"NDVI": 0.1, "NDBI": 0.1, "MNDWI": 0.1, "built_share": 0.1, "albedo_hs": 0.05, "ndre_hs": 0.1,
         "ndwi_hs": 0.1, "swir2200_depth_hs": 0.02, "elevation_rel_m": 10.0, "slope_deg": 5.0}


# Correlated indices share credit between them (e.g. NDVI vs NDBI), so read importance per group of factors
GROUPS = {"vegetation": ["NDVI", "ndre_hs"], "built-up": ["NDBI", "built_share"], "water": ["MNDWI", "ndwi_hs"],
          "brightness (albedo)": ["albedo_hs"], "materials (SWIR 2200)": ["swir2200_depth_hs"],
          "terrain": ["elevation_rel_m", "slope_deg"]}


def contributions(model, X):
    """(N, F) SHAP values in deg C (per-pixel contribution of each feature) and the base value.
    Uses the same trees as model.predict (up to the early-stopping best iteration)."""
    try:
        rng = (0, model.best_iteration + 1)
    except AttributeError:
        rng = (0, 0)
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=rng)
    return pd.DataFrame(c[:, :-1], columns=X.columns, index=X.index), float(c[0, -1])


def group(shap, groups=GROUPS):
    """SHAP summed within each factor group (per pixel)."""
    return pd.DataFrame({g: shap[[c for c in cols if c in shap]].sum(axis=1) for g, cols in groups.items()})


def ranking(shap, groups=None):
    """Mean |SHAP| per feature (deg C), overall or per group (e.g. climate)."""
    if groups is None:
        return shap.abs().mean().sort_values(ascending=False)
    return shap.abs().groupby(np.asarray(groups)).mean().T


def effects(shap, X, features=None, lo=5, hi=95):
    """Effect size per feature: slope of SHAP vs the feature over its central range, scaled to STEPS.

    Positive = more of this factor makes pixels hotter. Read as an association learned by the model,
    not a proven causal effect.
    """
    rows = []
    for f in features or [f for f in STEPS if f in X.columns]:
        x, s = X[f].to_numpy(), shap[f].to_numpy()
        a, b = np.nanpercentile(x, [lo, hi])
        m = np.isfinite(x) & (x >= a) & (x <= b)
        slope = np.polyfit(x[m], s[m], 1)[0] if m.sum() > 100 and b > a else np.nan
        rows.append(dict(feature=f, step=STEPS.get(f, np.nan), effect_c_per_step=slope * STEPS.get(f, np.nan),
                         range_lo=a, range_hi=b, mean_abs_shap=float(np.abs(s).mean())))
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
