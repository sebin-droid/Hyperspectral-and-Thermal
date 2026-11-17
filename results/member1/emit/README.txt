PASSED-PAIRS: EMIT + Landsat 8/9 pairs that passed the quality rules (EMIT QC v2.3)
=========================================================================
Pairs: 258 in 162 cells (pair numbers as in the master archive EMIT-LANDSAT/pairs/; gaps = not passed)
Climate groups (Koppen, Beck et al.): {'tropical': 258}

Each pair folder contains
  emit_reflectance.tif   EMIT L2A surface reflectance x 10000 (int16), nodata -32768 (masked or no data),
                         232 bands, wavelength in each band description
  landsat_<id>.tif       Landsat 8/9 on the SAME 60 m EMIT grid (float32, NaN = no data):
                         blue, green, red, nir, swir1, swir2 (surface reflectance), LST_K (kelvin), clear_frac (share of clear 30 m pixels)
  pair_info.json         scene IDs, dates, atmosphere values, Koppen code / climate group, QC numbers (final_qc)
handover_pass.csv lists every pair with cell, dates, climate and QC numbers. emit_bands.json is the band configuration.

How the pairs were made
  - EMIT L2A reflectance from Earth Engine (NASA/EMIT/L2A/RFL), used as delivered (no extra atmospheric correction)
  - water-vapour gap bands 127-141 and 187-212 dropped with a 3-band margin (232 of 285 kept)
  - EMIT pixel valid if cloud_flag == 0 and >= 80% of kept bands > 0
  - residual cloud and cloud shadow masked by comparison with Landsat (5x5 median difference > 0.06 and factor
    1.5, relative to the pair's own median EMIT/Landsat ratio, buffered 2 px);
    where Landsat is not clear, EMIT blue > 0.35 masked
  - Landsat averaged onto the EMIT 60 m grid over clear pixels
  - EMIT shifted by whole pixels only where the NDVI correlation with Landsat clearly improved (search +-8 px;
    applied_shift_yx); shifts > 3 px kept only if >= 2 of 3 independent signals agree

PASS rules (my design choices, not from the literature)
  joint coverage (EMIT valid and Landsat clear) >= 0.5, NIR ratio within [0.85, 1.15], AOD <= 0.8,
  sun elevation >= 25.0 deg, NDVI correlation after alignment >= 0.7, no unresolved shift,
  Landsat clear fraction (where Landsat has data) >= 0.85, large shifts confirmed,
  haze: NDVI gain (mean EMIT NDVI / mean Landsat NDVI over Landsat NDVI > 0.3) >= 0.8.
  FLAGS only: blue_ok / red_ok / swir1_ok (ratio outside 0.85-1.15): blue 219, red 101, SWIR1 52 of 258 pairs;
  haze_mild (NDVI gain 0.8-0.85): 31 pairs. Drop or down-weight these if you need the cleanest set.

Known caveats
  - Several scenes share a cell: split training and validation BY cell_id, not by pair.
  - The EMIT pairs are humid tropical cells by design (profile humid_coastal); the Hyperion pairs cover the other climates.
  - EMIT reads brighter than Landsat in blue (median ratio ~1.36) and red (~1.1); which sensor is closer to the truth is not established.
  - EMIT position offsets of up to 6 px (360 m) were found and corrected (large_shift = True: [27, 28, 29, 248]); cause not established.
  - EMIT pairs are 60 m; the Hyperion pairs are 30 m.
  - 'humid_coastal' is a climate/land-cover profile; the cells are not necessarily on the coast.
  - PASS means "passed these checks", not perfect data.
