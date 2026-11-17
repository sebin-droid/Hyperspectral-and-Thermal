FINAL DATASET: hyperspectral + Landsat 8/9 pairs for urban heat analysis (Member 1 -> Member 2)
=================================================================================================
Made 2026-10-24 by notebook FINAL_DATASET.ipynb. 422 pairs, 281 cells, QC v2.3 for all.

  EMIT       258 pairs   162 cells  60 m  232 bands  2023-2026   split train/val/test 202/27/29
  ENMAP       31 pairs    23 cells  30 m  177 bands  2023-2026   split train/val/test 21/4/6
  HYPERION   133 pairs   106 cells  30 m  137 bands  2013-2016   split train/val/test 98/14/21

WHERE THINGS ARE (paths relative to MEMBER1___PREPROCESSING/; column rel_dir of dataset_index.csv)
  <SENSOR>-LANDSAT/PASSED-PAIRS/<N>/
      <sensor>_reflectance.tif  hyperspectral surface reflectance x 10000 (int16), nodata -32768; wavelength in each band description
      landsat_<id>.tif          Landsat 8/9 on the SAME grid (float32, NaN = no data): blue, green, red, nir, swir1, swir2 (reflectance),
                                LST_K (land surface temperature, kelvin), band 8 = QA_PIXEL (Hyperion) or clear_frac (EMIT, EnMAP)
      aux_layers.tif            12 layers on the SAME grid (float32, NaN = no data):
                                1 NDVI  2 NDBI  3 NDWI (McFeeters)  4 MNDWI  (Landsat, clear pixels only)
                                5 NDVI_hs (from the hyperspectral bands)  6 landsat_clear (1 = clear, same rule as the QC)
                                7 elevation_m  8 slope_deg  9 aspect_deg (SRTM)  10 built_share (GHSL, 0-1, epoch nearest the scene year, <= 2020)
                                11 landcover_worldcover (ESA WorldCover 2021 class code)  12 landcover_dw (Dynamic World, scene date +/- 6 months)
      pair_info.json            scene IDs, dates, angles, atmosphere, climate, all QC numbers
  <SENSOR>-LANDSAT/PASSED-PAIRS/README.txt, handover_pass.csv, <sensor>_bands.json   per-sensor details and band lists
  FINAL-DATASET/dataset_index.csv          one row per pair: files, dates, cell, climate, QC numbers, layer statistics, split
  FINAL-DATASET/dataset_index_columns.csv  meaning of every column
  FINAL-DATASET/eda/                       figures and tables

HOW TO LOAD ONE PAIR (Python)
  import pandas as pd, numpy as np, rasterio
  BASE = "/content/drive/MyDrive/MEMBER1___PREPROCESSING"
  idx = pd.read_csv(f"{BASE}/FINAL-DATASET/dataset_index.csv"); r = idx.iloc[0]; d = f"{BASE}/{r['rel_dir']}"
  with rasterio.open(f"{d}/{r['reflectance_file']}") as s: hs = s.read().astype("float32"); hs[hs == -32768] = np.nan; hs /= 10000
  with rasterio.open(f"{d}/{r['landsat_file']}") as s: ls = s.read()          # ls[6] = LST in kelvin
  with rasterio.open(f"{d}/{r['aux_file']}") as s: aux = s.read(); names = s.descriptions
  valid = np.isfinite(hs).all(axis=0) & (aux[names.index("landsat_clear")] == 1)   # pixels usable for training

SPLIT (column split)
  Assigned by CELL (a stable hash of cell_id, about 70/15/15 % of cells): all dates of a cell are in the same split, so the test set has
  only places the model never saw. Use this column; do not split by pair or by pixel. For a stricter spatial test use block_1deg.

THINGS TO KNOW
  - Resolution differs: Hyperion and EnMAP 30 m, EMIT 60 m (lat/lon grid). Band sets differ per sensor (see <sensor>_bands.json);
    resample spectrally to a common set or train per sensor.
  - Scene years: EMIT 2023-2026; ENMAP 2023-2026; HYPERION 2013-2016. Hyperion cells are mostly NOT Kerala-like (dense city / semi-arid);
    EMIT and EnMAP give the humid tropical cells. Filter on climate_group or profile if needed.
  - Flags (still PASS): flag_blue / flag_red / flag_swir1 = band ratio to Landsat outside 0.85-1.15; haze_mild = slight haze.
    Use them to drop or down-weight pairs for the cleanest set: 44 haze_mild, 116 flag_swir1.
    flag_blue is very common (321 of 422): the blue band differs between the sensors' atmospheric
    corrections; it is not a reason to drop a pair.
  - SRTM is a surface model: over forests and dense buildings elevation_m includes part of the canopy / building height.
  - Split shares are by cell and random, so they differ a little from 70/15/15 (321/45/56 pairs);
    for cross-validation group by cell_id (e.g. sklearn GroupKFold).
  - Known sensor differences: Hyperion SWIR reads lower than Landsat (about 0.88); EnMAP SWIR higher (about 1.16-1.19);
    EnMAP NIR depends on the viewing angle (BRDF; across_offnadir).
  - LST is Landsat Collection 2 Level-2 surface temperature. Time between the Landsat and the hyperspectral scene
    (|landsat_dt_days|): EMIT median 2.21 d, max 15 d; ENMAP median 1.97 d, max 14 d; HYPERION median 3.15 d, max 14 d.
  - All EMIT and EnMAP pairs are tropical; the arid, temperate and cold pairs are all Hyperion (2013-2016).
  - Built-up change: 0 pairs flagged built_growth_large (cell changed between the scene and 2020).
  - Dynamic World starts in June 2015: landcover_dw is empty for most Hyperion pairs; use landcover_worldcover (2021) there.
  - PASS means "passed these checks", not perfect data.
