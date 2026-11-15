import json

import h5py
import numpy as np
import pytest

from urbanheat.contracts import ENV_INPUTS, LS_INPUTS, MET_INPUTS

H = W = 40
TRANSFORM = [60.0, 0.0, 780000.0, 0.0, -60.0, 1440000.0]      # near Bengaluru, EPSG:32643


@pytest.fixture
def fake_h5(tmp_path):
    """Tiny dataset in the urbanheat_v1.h5 layout (CONTRACTS section 1): 4 pairs (train, train, val, test),
    LST = f(NDVI, built_share) + noise, and NDVI outliers like the real file."""
    rng = np.random.default_rng(0)
    path = tmp_path / "fake.h5"
    arrays, px = {k: [] for k in ["hs", "ls", "env", "thermal", "met", "lst", "lst_anom"]}, {"ls": [], "split": []}
    with h5py.File(path, "w") as f:
        gp = f.create_group("pairs")
        for i, split in enumerate(["train", "train", "val", "test"]):
            ndvi = rng.uniform(-0.2, 0.8, (H, W))
            ls = rng.normal(0.2, 0.05, (9, H, W))
            ls[6], ls[7], ls[8] = ndvi, -ndvi, rng.uniform(-0.5, 0.5, (H, W))
            ls[6, 0, :4] = [-478, 1452, 900, -300]
            env = np.stack([rng.normal(900, 5, (H, W)), rng.uniform(0, 5, (H, W)), rng.uniform(0, 1, (H, W))])
            lst = 35 - 5 * ndvi.clip(-1, 1) + 3 * env[2] + rng.normal(0, 0.3, (H, W))
            valid = np.ones((H, W), bool)
            valid[-3:] = False
            d = dict(hs=rng.uniform(0, 0.5, (122, H, W)), ls=ls, env=env, thermal=(lst + rng.normal(0, 1, (H, W)))[None])
            g = gp.create_group(f"EMIT_{i}")
            g["hs"] = d["hs"].astype("float16")
            for k in ["ls", "env", "thermal"]:
                g[k] = d[k].astype("float32")
            g["met"] = rng.normal(0, 1, 6).astype("float32")
            g["landcover"] = np.full((H, W), 50, "uint8")
            g["lst_c"], g["valid"] = lst.astype("float32"), valid.astype("uint8")
            med = float(np.median(lst[valid]))
            g.attrs.update(sensor="EMIT", split=split, climate_group="tropical", cell_id=f"c{i}", lst_median_c=med,
                           transform=TRANSFORM, crs="EPSG:32643")
            px["ls"].append(ls[:, valid].T)
            px["split"].append(np.full(valid.sum(), split))
            if split == "train":
                for k in ["hs", "ls", "env", "thermal"]:
                    arrays[k].append(d[k][:, valid].T)
                arrays["met"].append(g["met"][:][None])
                arrays["lst"].append(lst[valid][:, None])
                arrays["lst_anom"].append((lst[valid] - med)[:, None])
        gx = f.create_group("pixels")
        gx["ls"] = np.concatenate(px["ls"]).astype("float32")
        gx["split"] = np.concatenate(px["split"]).astype("S16")
        gn = f.create_group("norm")
        for k, parts in arrays.items():
            a = np.concatenate(parts)
            gn[k + "_mean"], gn[k + "_std"] = a.mean(0).astype("float32"), (a.std(0) + 1e-6).astype("float32")
        f.attrs.update(version=1, grid_m=60, wavelengths_nm=np.arange(460.0, 460 + 10 * 122, 10), split_column="split",
                       inputs=json.dumps(dict(ls=LS_INPUTS, env=ENV_INPUTS, met=MET_INPUTS, thermal=["lst_context_c"])))
    return path
