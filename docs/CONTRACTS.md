# Team contracts

These formats let the four members work at the same time without waiting on each other.
The code version is [`src/urbanheat/contracts.py`](../src/urbanheat/contracts.py). Change either only by team agreement,
in a pull request that all owners review.

## 1. Harmonised dataset — produced by `urbanheat.harmonize` (Members 1 + 2)

One HDF5 file `<processed_dir>/urbanheat_v<N>.h5` (`urbanheat harmonize`), all 422 pairs, each on its own
CRS at 60 m (30 m pairs are 2×2-averaged), on one common wavelength grid (122 bands, 460–2340 nm, the
10 nm wavelengths every sensor covers). Read it with `urbanheat.data.dataset.PatchDataset` / `load_pixels`.

| Path | Shape / type | Meaning |
|---|---|---|
| file attrs | | `wavelengths_nm`, `grid_m`, `version`, `split_column`, `index_md5`, `inputs` (JSON of channel names) |
| `/norm/<key>_mean`, `_std` | float32 | `hs, ls, env, thermal, met, lst, lst_anom`; **train split only** |
| `/pairs/<pair_id>/hs` | (122, H, W) float16 | reflectance (0–1) on the common wavelength grid |
| `/pairs/<pair_id>/ls` | (9, H, W) float32 | `LS_INPUTS` (Landsat bands + NDVI, NDBI, MNDWI) |
| `/pairs/<pair_id>/env` | (3, H, W) float32 | `ENV_INPUTS` |
| `/pairs/<pair_id>/thermal` | (1, H, W) float32 | thermal context, °C (section 1b) |
| `/pairs/<pair_id>/met` | (6,) float32 | `MET_INPUTS` (ERA5 at the Landsat overpass) |
| `/pairs/<pair_id>/landcover` | (H, W) uint8 | ESA WorldCover class (2×2 mode) |
| `/pairs/<pair_id>/lst_c` | (H, W) float32 | Landsat LST of clear pixels, °C (target) |
| `/pairs/<pair_id>/valid` | (H, W) uint8 | 1 = every input and the target are present |
| `/pairs/<pair_id>` attrs | | sensor, cell_id, climate_group, profile, split, block_1deg, landsat_scene, landsat_date, `lst_median_c`, n_valid, transform, crs |
| `/pixels/*` | (N, …) | random valid pixels per pair for RF / XGBoost, incl. `lst_c` and `lst_anom` |

`pair_id` = `<SENSOR>_<pair_no>`, e.g. `EMIT_12`. Splits come from Member 1's `split` column (by cell).
Anomaly target = `lst_c − lst_median_c` (the pair's median over valid pixels): the surface effect without the day's weather.

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

## 4b. Land-cover and heat change — produced by `urbanheat change` (Member 1 work), for Member 4's dashboard

`<derived_dir>/change/cell_year.csv`, one row per city (cell) and year 2016–2025:
`cell_id, climate_group, year, hot_season_start_month, green, built, water, bare` (ha, Dynamic World yearly mode),
`n_scenes, dw_images, lst_built_c, lst_cell_c, lst_rural_c` (hot-season Landsat median, °C; empty = no clear scene),
`suhi_c` (= built − rural ring 1–6 km outside).

`<derived_dir>/change/transitions.csv`, one row per city: `dlst_<transition>` = hot-season LST change
(last 3 years − first 3 years, °C) and `n_<transition>` = 30 m pixels, for `green_stable, green_to_built,
built_stable, built_to_green, water_stable, water_to_other`. Extra warming of a change =
`dlst_green_to_built − dlst_green_stable` etc.

## 5. Map outputs — for the dashboard

Cloud-optimised GeoTIFF, EPSG of the pair, band 1 = predicted LST (°C), band 2 = uncertainty (°C) if available.
