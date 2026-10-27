# Team contracts

These formats let the four members work at the same time without waiting on each other.
The code version is [`src/urbanheat/contracts.py`](../src/urbanheat/contracts.py). Change either only by team agreement,
in a pull request that all owners review.

## 1. Harmonised dataset — produced by `urbanheat.harmonize` (Members 1 + 2)

One HDF5 file `data/processed/urbanheat_v<N>.h5`, all 422 pairs on a common grid:

| Path | Shape / type | Meaning |
|---|---|---|
| `/meta` attrs | | `wavelengths_nm`, `grid_m`, input names, `version`, source index checksum |
| `/norm/<key>_mean`, `_std` | float32 | statistics from the **train split only** |
| `/pairs/<pair_id>/hs` | (C, H, W) float16 | reflectance resampled to the common wavelength grid |
| `/pairs/<pair_id>/ls` | (9, H, W) float32 | `LS_INPUTS` |
| `/pairs/<pair_id>/env` | (3, H, W) float32 | `ENV_INPUTS` |
| `/pairs/<pair_id>/met` | (6,) float32 | `MET_INPUTS` (ERA5 at the Landsat overpass) |
| `/pairs/<pair_id>/landcover` | (H, W) uint8 | ESA WorldCover class |
| `/pairs/<pair_id>/lst_c` | (H, W) float32 | Landsat LST, deg C (NaN where invalid) |
| `/pairs/<pair_id>/valid` | (H, W) uint8 | 1 = usable pixel |
| `/pairs/<pair_id>` attrs | | sensor, cell_id, climate_group, date, split, transform, crs |

`pair_id` = `<SENSOR>_<pair_no>`, e.g. `EMIT_12`. Splits come from Member 1's `split` column (by cell).

## 1b. Thermal context layer — produced by `urbanheat thermal-context` (Member 1 work)

`<derived_dir>/thermal_context/<rel_dir>/thermal_context.tif`, on the pair's own grid (same CRS, transform, size):

| Band | Meaning |
|---|---|
| 1 `lst_context_c` | median Landsat 8/9 LST (°C) of clear pixels in the same season (±60 days of year, 2013–2025) |
| 2 `n_obs` | number of clear observations behind the median |

Scenes within ±16 days of the pair's Landsat date are excluded, so the target never leaks into the input.
`thermal_context/summary.csv` holds per-pair QC: `n_scenes, coverage, median_n_obs, context_mean_c,
target_mean_c, r_with_target`. Settings: `thermal_context` in `configs/default.yaml`.

## 2. Model — produced by Member 2, consumed by Members 3 and 4

Every model implements `HeatModel.predict(batch) -> (B, H, W) LST in °C` and is saved as
`runs/<run_id>/` containing `model.pt` (or `model.onnx`), `config.yaml`, `metrics.json` and `model_card.md`.

`metrics.json` (one schema for every model, so Member 3's comparison table is automatic):

```json
{"run_id": "...", "model": "fusion_cnn", "target": "anomaly", "split": "test",
 "mae": 0.0, "rmse": 0.0, "r2": 0.0, "per_climate": {"arid": {"mae": 0.0}},
 "params": 0, "flops": 0, "size_mb": 0.0, "latency_ms": 0.0, "peak_mem_mb": 0.0, "hardware": "RTX 3050 6 GB"}
```

## 3. Scenario — defined by Member 3, used by Member 4's app

`Scenario(name, changes={lever: additive change}, region=geojson)`. Allowed levers: `NDVI`, `built_share`,
`albedo`, `MNDWI`. Output: `ΔLST = predict(changed) − predict(current)` per pixel, as a GeoTIFF, with the
note that it is a modelled estimate, not a causal guarantee.

## 4. Explanations — produced by Member 4 (and Member 2 for the factor analysis)

- `runs/<run_id>/explain/global_importance.csv`: `feature, mean_abs_shap, direction, climate_group`
- `runs/<run_id>/explain/<pair_id>_attr.tif`: one band per input group (attribution per pixel)

## 5. Map outputs — for the dashboard

Cloud-optimised GeoTIFF, EPSG of the pair, band 1 = predicted LST (°C), band 2 = uncertainty (°C) if available.
