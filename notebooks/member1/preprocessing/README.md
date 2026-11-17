# Member 1: satellite data collection & preprocessing

These notebooks built the dataset every other member uses: **422 hyperspectral + Landsat 8/9 pairs in 281 locations**
(Hyperion 133, EMIT 258, EnMAP 31), all passed through the same quality control (QC v2.3), each with 12 extra layers on
the same pixel grid, one index file and a train / val / test split by location.

The full dataset (the rasters, about 3.8 GB) is kept on Google Drive and shared with the team members only; ask Member 1
for access. Small result files and figures are in [`results/member1/`](../../../results/member1/README.md).

All notebooks ran in **Google Colab** with Google Drive (`/content/drive/MyDrive/MEMBER1___PREPROCESSING`) and
Google Earth Engine (project `hyperspectral-thermal-511303`).

## Run order

| # | Notebook | What it does | Main output |
|---|---|---|---|
| 1 | [01_pair_screening](01_pair_screening.ipynb) | Global screen of 0.07° (≈ 7.7 km) cells: settlement (GHSL SMOD 2020), climate (Köppen–Geiger 1991–2020, Beck et al.), land-cover fit against reference tiles (ESA WorldCover); footprints of every Hyperion (USGS M2M), EMIT (NASA earthaccess) and EnMAP (DLR STAC) scene; Landsat 8/9 match within ±15 days and ≥ 85 % clear | `screening/ranked_pairs.csv` (8,990 candidate pairs), `handover_pairs.csv` (first 200) |
| 2 | [02_hyperion_landsat](02_hyperion_landsat.ipynb) | Hyperion L1T download (USGS M2M), clip to the cell, bad-band removal (242 → 137), **6S atmospheric correction** (MODIS MAIAC aerosol + water vapour, SRTM height), Landsat on the Hyperion grid, alignment, QC v1 → v2.3, handover | 133 PASS pairs of 283 |
| 3 | [03_emit_landsat](03_emit_landsat.ipynb) | EMIT L2A reflectance from Earth Engine (`NASA/EMIT/L2A/RFL`), cell only, with Landsat averaged onto the 60 m grid; gap bands removed (285 → 232); cloud / shadow / haze masks; QC v2.3; handover | 258 PASS pairs of 400 |
| 4 | [04_enmap_landsat](04_enmap_landsat.ipynb) | EnMAP L2A from DLR (STAC + window read, DLR login in Colab Secrets), 47 noisy bands removed (224 → 177), defective pixels filled, Landsat on the EnMAP grid, neighbouring Landsat scene where the screened one ended inside the cell; QC v2.3; handover | 31 PASS pairs of 40 |
| 5 | [05_final_dataset](05_final_dataset.ipynb) | For all 422 pairs: `aux_layers.tif` (NDVI, NDBI, NDWI, MNDWI, hyperspectral NDVI, Landsat clear mask, elevation, slope, aspect, GHSL built-up share, WorldCover, Dynamic World), `dataset_index.csv`, GHSL change check, split by cell, EDA, README | `FINAL-DATASET/` |

Notebooks 1 and 2 keep their printed outputs (account names replaced by `<your-username>`). In notebook 2, cells
06–12 were run later in Colab; their code is included with each cell's result written above it. Notebooks 3–5 are
the cleaned final versions; their results are summarised in [`results/member1/README.md`](../../../results/member1/README.md).

## Credentials

Never stored in the notebooks. USGS (username + application token) and NASA Earthdata are typed into hidden prompts;
the DLR account is read from Colab Secrets (`DLR_USER`, `DLR_PASS`).

## Quality control (QC v2.3, identical for all three sensors)

A pair passes only if all eight checks pass:

| Check | Rule for PASS |
|---|---|
| Joint coverage | ≥ 50 % of the cell valid in the hyperspectral image **and** clear in Landsat |
| NIR ratio | hyperspectral / Landsat NIR between 0.85 and 1.15 |
| Spatial agreement | NDVI correlation r ≥ 0.7 after alignment (Hyperion: the better of NDVI and SWIR1) |
| Alignment | no unresolved shift (search ±8 px; shifts > 3 px confirmed by 2 of 3 signals) |
| Aerosol | AOD ≤ 0.8 |
| Sun elevation | ≥ 25° |
| Landsat clear | ≥ 85 % |
| Haze | hyperspectral NDVI / Landsat NDVI over vegetation ≥ 0.80 |

Flags only (the pair still passes): blue, red or SWIR1 ratio outside 0.85–1.15; `haze_mild` (0.80–0.85).
All thresholds are design choices, documented in each notebook's "Design choices" section.
