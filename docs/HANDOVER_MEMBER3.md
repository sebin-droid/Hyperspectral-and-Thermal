# Handover: efficient model and what-if scenarios

**From:** Muhammed Shahid (GitHub: muhammedshahidsds25-collab)
**Code:** pull request #1, https://github.com/sebin-droid/Hyperspectral-and-Thermal/pull/1 (branch `member3/efficient`, to be merged into `sebin`)
**Files:** this folder, `member3_share/`

---

## 1. What I did, in short

1. **Built a small, fast copy of Member 2's best CNN** (`fusion_cnn`). It is as accurate, 37× smaller, and about 10× faster on a normal laptop CPU.
2. **Built a what-if engine.** You change the land surface in a region (more trees, more buildings), and it returns a map of how much the surface temperature changes (ΔLST, °C) for every pixel.
3. **Exported the models to ONNX**, so the dashboard can run them without PyTorch.

## 2. Results (test split: 56 cities never seen in training)

| Model | MAE / RMSE / R² | Params | MFLOPs per 32×32 patch | CPU ms per patch (ONNX) | Size |
|---|---|---|---|---|---|
| Member 2: `fusion_cnn` (teacher) | 0.98 / 1.40 / 0.780 | 293,241 | 597.6 | 4.63 | 1.21 MB |
| `m3_lite32_ce` (small, no teacher) | 1.04 / 1.46 / 0.761 | 18,985 | 37.3 | 1.18 | 0.11 MB |
| **`m3_lite16_kd` (small, taught by teacher), recommended** | **0.98 / 1.39 / 0.784** | **7,905** | **15.3** | **0.46** | **0.07 MB** |
| `m3_nohs32_kd` (no hyperspectral input) | 0.97 / 1.38 / 0.787 | 15,081 | 29.3 | 0.65 | 0.10 MB |

- The small model keeps **100.6 % of the teacher's R²** with **97.4 % fewer FLOPs**.
- **Learning from the teacher matters.** At the same size, R² is 0.784 with the teacher and 0.761 without it.
- **Hyperspectral input does not help for this target.** The model without it is the most accurate (0.787).
- INT8 quantisation shrinks Member 2's model 3.5× (1.21 MB → 0.34 MB) with no accuracy loss (R² 0.7796 → 0.7794). It is slower than FP32 ONNX on the laptop CPU, and the small models are already under 0.1 MB, so I recommend FP32 ONNX.
- CPU timings are from an Intel laptop (no GPU) at batch 32. Training ran on a Kaggle GPU, 60 epochs max with early stopping. Each model was trained once (one seed).

### What-if results (central 32×32 block of each test city changed, the rest unchanged)

| Scenario | Small model | Member 2's model | Observed (Member 1, satellite change data) |
|---|---|---|---|
| green → built | **+0.20 °C** | −0.04 °C (wrong sign) | +0.36 °C (95 % CI 0.29 – 0.44) |
| built → green | **−0.15 °C** | +0.18 °C (wrong sign) | −0.42 °C (95 % CI −0.49 – −0.38) |
| greening (NDVI +0.1, NDBI −0.06) | −0.09 °C | −0.00 °C | – |

- The small model gets the same direction as the real data, at about half the size. Member 2's model gets the direction wrong for both changes.
- The effect stays local: pixels outside the changed block move by less than 0.005 °C.
- By climate, the effect is largest in arid cities (green → built +0.35 °C, built → green −0.26 °C) and smallest in tropical ones.

## 3. What's in this folder

| Folder | Content | Who needs it |
|---|---|---|
| `1_models/` | 5 trained models (`model.pt`, `config.json`, `metrics.json`, `history.json`) | Member 4 (XAI), Member 2 |
| `2_models_onnx/` | The same 5 models as ONNX | Member 4 (dashboard) |
| `3_scenario_maps/` | ΔLST GeoTIFFs for city `EMIT_243`: `built_to_green`, `green_to_built`, `greening`, `ndvi_only`, `densify` | Member 4 (dashboard, GenAI text) |
| `4_results/` | `model_comparison.csv` (the table above, slide 17) and the updated slides | everyone |
| `5_notebooks/` | `01_efficient.ipynb` (model comparison), `02_scenarios.ipynb` (what-if results) | everyone |

## 4. How to use it (Member 4)

### Important: input preparation

My models expect **NDVI, NDBI and MNDWI clipped to [−1, 1] before normalising**. The h5 file has outliers (NDVI from −478 to 1452) from `harmonize/build.py`, and they made the original normalisation wrong. `Prepared` fixes this for you. **Always wrap the dataset:**

```python
from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient import distill, lite

ds = distill.Prepared(PatchDataset(H5, "test"))   # clips indices, uses corrected train stats
model = lite.load("member3_share/1_models/m3_lite16_kd")   # or runs/m3_lite16_kd after the merge
scene = ds.scene("EMIT_243")                      # dict of inputs, same as Member 2's
```

Member 2's models still use the plain `PatchDataset`. Don't feed them `Prepared` data.

### Predict one city

```python
from urbanheat.models import train
lst_anomaly = train.predict_scene(model, ds, "EMIT_243", "cpu")   # (H, W), deg C above/below the city median
```

### ONNX (dashboard, no PyTorch)

- Input names: `hs`, `ls`, `env`, `thermal`, `sensor`.
- Shapes: `(B, C, H, W)` float32 for the first four, and `(B,)` int64 for `sensor`. The batch size and the patch size can vary.
- Output: `lst`, shape `(B, H, W)`, the LST anomaly in °C.
- Inputs must be prepared exactly as `Prepared` does: clipped, then normalised.

### Make a new what-if map

```python
from urbanheat.contracts import Scenario
from urbanheat.scenarios import engine

sc = Scenario("built_to_green", {"NDVI": 0.34, "NDBI": -0.20, "built_share": -0.30}, region="district.geojson")
dlst = engine.delta_lst(model, ds, "EMIT_243", sc)              # (H, W) deg C, NaN outside valid pixels
engine.write_geotiff("EMIT_243_built_to_green.tif", dlst, ds.path, "EMIT_243", sc)
```

- **Levers:**
  - `NDVI`, `NDBI` and `MNDWI`: −1 to 1
  - `built_share`: 0 to 1
  - Changes are added in real units and then clipped to the lever's range.
- **`albedo` is rejected.** It's in the contract, but it isn't a model input.
- **Region:** a GeoJSON file in EPSG:4326, or pass `where=` with a boolean `(H, W)` mask.
- **Always change a region, not the whole city.** The target is an anomaly from the city median, so a whole-city change is meaningless.
- **Recommended scenarios** (typical changes taken from Member 1's data):
  - built → green: `{"NDVI": +0.34, "NDBI": -0.20, "built_share": -0.30}`
  - green → built: `{"NDVI": -0.34, "NDBI": +0.20, "built_share": +0.30}`
  - greening: `{"NDVI": +0.1, "NDBI": -0.06}`
- **GeoTIFF:** one float32 band named `delta_lst_c`, on the city's own CRS. The tags hold the scenario and the caveat note.

## 5. Limits (please show these in the dashboard and the report)

- **Modelled estimate, not proof of cause.** Every map carries this note.
- **The effect is about half the observed size.** The thermal-context input (last season's LST) and the linked inputs (red/NIR bands, hyperspectral) stay unchanged in a scenario, which damps the response.
- **Surface temperature, not air temperature.**
- **Single seed.** Repeated runs, to show the spread, are not done yet.
- **Scenario maps exist only for `EMIT_243`.** Make others with the code above.

## 6. Ideas where others can help

- **Member 4 (XAI):** run SHAP / attributions on `m3_lite16_kd` and compare with Member 2's model. Mine responds to NDVI and NDBI in the right direction; Member 2's barely responds.
- **Member 4 (dashboard):** use the ONNX model for live what-if on a drawn polygon. It takes about 0.5 ms per 32×32 patch on CPU.
- **Member 4 (GenAI):** the GeoTIFF tags (`scenario`, `changes`, `note`) can feed a text summary.
- **Member 1:** fix the NDVI/NDBI outliers in `harmonize/build.py` by clipping the indices after the 60 m block mean. The real fix belongs there, not in my wrapper.
- **Member 2:** retraining `fusion_cnn` with clipped indices should fix its wrong-sign scenario response. Also consider dropping the hyperspectral branch.

## 7. Code map (in the pull request)

| File | What it does |
|---|---|
| `src/urbanheat/efficient/lite.py` | `lite_cnn` (small CNN with depthwise-separable blocks), `load(run_dir)` loads any run, including Member 2's |
| `src/urbanheat/efficient/distill.py` | `Prepared` (index clipping, teacher blending), `run()` trains a student |
| `src/urbanheat/efficient/benchmark.py` | `measure` (params, FLOPs, size, latency), `export_onnx`, `quantize_int8`, `onnx_run` |
| `src/urbanheat/scenarios/engine.py` | `apply`, `delta_lst`, `region_mask`, `write_geotiff` |
| `notebooks/member3/00_kaggle_train.ipynb` | Trains all 5 small models on Kaggle (GPU) |
| `tests/test_efficient.py`, `tests/test_scenarios.py` | Tests (37 pass in total) |

**Contract changes in the pull request:**
- `NDBI` added to `LEVERS` in `contracts.py`.
- `fusion_nohs_cnn` added to `cnn.VARIANTS`.
