import numpy as np
import pandas as pd
import xgboost as xgb

from urbanheat.drivers import attribution as A
from urbanheat.models import baselines as B
from urbanheat.models import features as F


def test_scores():
    y = np.array([1.0, 2, 3, 4])
    s = B.scores(y, y)
    assert s["mae"] == 0 and s["r2"] == 1
    assert abs(B.scores(y, np.full(4, y.mean()))["r2"]) < 1e-12


def test_hs_indices():
    wl = np.arange(460, 2341, 10.0)
    hs = np.tile(np.linspace(0.1, 0.3, len(wl)), (5, 1))
    hs[:, np.argmin(abs(wl - 2200))] *= 0.9                     # a 10 % absorption dip at 2200 nm
    out = F.hs_indices(hs, wl)
    assert set(out) == set(F.HS_INDICES)
    np.testing.assert_allclose(out["albedo_hs"], hs.mean(1))
    assert (out["ndre_hs"] > 0).all() and (out["swir2200_depth_hs"] > 0.09).all()


def test_feature_sets_are_nested():
    sets = F.feature_sets(["hs_500", "hs_600"])
    assert set(sets["landsat"]) < set(sets["landsat+hs_indices"]) < set(sets["landsat+hs_bands"]) < set(sets["all"])
    assert "thermal_anom_c" not in F.DRIVERS and "lst_context_c" not in F.DRIVERS


def test_shap_contributions_add_up_and_effects_have_the_right_sign():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"NDVI": rng.uniform(0, 0.8, 3000), "built_share": rng.uniform(0, 1, 3000)})
    y = -5 * X.NDVI + 3 * X.built_share + rng.normal(0, 0.1, 3000)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3).fit(X, y)
    shap, base = A.contributions(model, X)
    np.testing.assert_allclose(shap.sum(1) + base, model.predict(X), atol=1e-3)
    eff = A.effects(shap, X).set_index("feature").effect_c_per_step
    assert -0.6 < eff["NDVI"] < -0.4                   # -5 deg C per unit NDVI -> about -0.5 per +0.1
    assert 0.2 < eff["built_share"] < 0.4              # +3 per unit -> about +0.3 per +0.1
    g = A.group(shap)
    np.testing.assert_allclose(g["vegetation"], shap.NDVI)


def test_shap_matches_predict_with_early_stopping():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"NDVI": rng.uniform(0, 1, 2000)})
    y = -3 * X.NDVI + rng.normal(0, 0.5, 2000)
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.3, early_stopping_rounds=5)
    model.fit(X[:1500], y[:1500], eval_set=[(X[1500:], y[1500:])], verbose=False)
    assert model.best_iteration < 499
    shap, base = A.contributions(model, X)
    np.testing.assert_allclose(shap.sum(1) + base, model.predict(X), atol=1e-3)
