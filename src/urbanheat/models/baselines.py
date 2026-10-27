"""Pixel baselines (Ridge, Random Forest, XGBoost) with the shared metrics schema (docs/CONTRACTS.md section 2)."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def scores(y, p):
    err = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
                r2=float(1 - (err ** 2).sum() / ss) if ss > 0 else float("nan"), n=int(len(y)))


def evaluate(y, p, info):
    """Overall scores plus per climate group and per sensor."""
    out = scores(y, p)
    for col in ("climate_group", "sensor"):
        out["per_" + col.split("_")[0]] = {g: scores(y[m], p[m]) for g in sorted(info[col].unique())
                                            for m in [(info[col] == g).to_numpy()] if m.sum() > 50}
    return out


def make(kind, seed=42):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1,
                                     random_state=seed)
    if kind == "xgb":
        return xgb.XGBRegressor(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, tree_method="hist", device="cuda",
                                early_stopping_rounds=100, eval_metric="rmse", random_state=seed)
    raise ValueError(kind)


def fit(kind, X, y, info, max_train=None, seed=42):
    """Train on the train split (XGBoost stops early on val). Returns the model and its train time."""
    tr, va = (info.split == "train").to_numpy(), (info.split == "val").to_numpy()
    idx = np.nonzero(tr)[0]
    if max_train and len(idx) > max_train:
        idx = np.random.default_rng(seed).choice(idx, max_train, replace=False)
    model = make(kind, seed)
    t0 = time.time()
    if kind == "xgb":
        model.fit(X.iloc[idx], y[idx], eval_set=[(X[va], y[va])], verbose=False)
    else:
        model.fit(X.iloc[idx], y[idx])
    return model, time.time() - t0


def run(name, kind, X, y, info, runs_dir, target, max_train=None, hardware=""):
    """Fit, evaluate on val and test, save runs/<name>/metrics.json (+ the XGBoost model). Returns metrics."""
    model, secs = fit(kind, X, y, info, max_train)
    out = dict(run_id=name, model=kind, target=target, features=list(X.columns), n_features=X.shape[1],
               train_seconds=round(secs, 1), hardware=hardware)
    for split in ("val", "test"):
        m = (info.split == split).to_numpy()
        t0 = time.time()
        p = model.predict(X[m])
        out[split] = evaluate(y[m], p, info[m])
        out[split]["latency_ms_per_1k_px"] = round(1000 * (time.time() - t0) / m.sum() * 1000, 3)
    d = Path(runs_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    if kind == "xgb":
        model.save_model(d / "model.json")
        out["size_mb"] = round((d / "model.json").stat().st_size / 1e6, 2)
        out["best_iteration"] = int(model.best_iteration)
    (d / "metrics.json").write_text(json.dumps(out, indent=1))
    return out, model
