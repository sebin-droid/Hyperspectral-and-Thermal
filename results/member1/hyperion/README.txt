PASSED-PAIRS: Hyperion + Landsat 8/9 pairs that passed the quality rules (QC v2.3)
==================================================================================
Pairs: 133 in 106 cells (pair numbers as in the master archive HYPERION-LANDSAT/pairs/; gaps = not passed)
Profiles: {'dense_city': 99, 'semi_arid_city': 33, 'humid_coastal': 1} | climate groups (Koppen): {'temperate': 53, 'arid': 41, 'tropical': 21, 'cold': 18}

Each pair folder contains
  hyperion_reflectance.tif  surface reflectance x 10000 (int16), nodata -32768 (masked or no data), 137 bands,
                            wavelength + band number in each band description, 30 m
  landsat_<id>.tif          Landsat 8/9 on the SAME 30 m grid: blue, green, red, nir, swir1, swir2 (surface reflectance), LST_K (kelvin), QA_PIXEL
  pair_info.json            scene IDs, dates, atmosphere inputs, QC numbers (final_qc), koppen_code, climate_group
handover_pass.csv lists every pair with profile, climate group, cell, dates and QC numbers. hyperion_bands.json is the band configuration.

How the Hyperion reflectance was made
  - L1T GeoTIFF (L1Gst only where L1T was unavailable), clipped to a 0.07 deg cell
  - 6S atmospheric correction per scene (uniform atmosphere; AOD and water vapour from MODIS MAIAC +/-3 days, else +/-15; SRTM elevation)
  - bad / noisy bands removed: [51, 110, 132, 184, 185, 186]; plus < 450 nm, 930-980 nm, 1100-1180 nm and the water-vapour regions
  - pixel invalid if more than 20% of bands have reflectance <= 0 (swath-edge wedge)
  - haze/cloud mask: 5x5 median of (Hyperion blue - Landsat blue) > 0.06 AND Hyperion blue / Landsat blue > 1.5
    where Landsat is clear; Hyperion blue > 0.35 where Landsat is not clear; grown by 1 px
  - cloud-shadow mask: 5x5 median of (Hyperion NIR - k x Landsat NIR) < -0.06 AND Hyperion NIR < k x Landsat NIR / 1.5
    (k = the pair's median NIR ratio), Landsat NIR > 0.1; grown by 2 px (same rule as the EMIT pairs)
  - Hyperion shifted by whole pixels only where the correlation with Landsat (NDVI or SWIR1) clearly improved (search +/-8 px)

PASS rules (my design choices, not from the literature)
  Hyperion coverage >= 0.5 and joint coverage (Hyperion valid AND Landsat clear) >= 0.5; NIR ratio within [0.85, 1.15];
  agreement = max(NDVI r, SWIR1 r) >= 0.7; no unresolved shift; AOD <= 0.8; sun elevation >= 25.0 deg;
  NDVI level: mean Hyperion NDVI / mean Landsat NDVI over Landsat NDVI > 0.3 >= 0.8
  (catches whole-scene haze and red bias; same rule as the EMIT pairs).
  FLAGS only: blue_ok / red_ok / swir1_ok (ratio outside 0.85-1.15): blue 84, red 31, SWIR1 48 of 133 pairs;
  haze_mild (NDVI gain 0.8-0.85): 12 pairs. Drop or down-weight these if you need the cleanest set.

Known caveats
  - Hyperion SWIR reads lower than Landsat (median SWIR1 ratio about 0.88), more so in later years (2016); cause not established.
  - Use QA_PIXEL (bits 1-4 = 0) as the Landsat clear mask, and Hyperion nodata (-32768) as the Hyperion mask.
  - Several dates can share a cell: split training and validation BY cell_id, not by pair.
  - Only one humid_coastal pair; dense_city is a general urban profile (no climate gate). Filter on climate_group if needed.
  - Cells were selected with WorldCover 2021 / GHSL 2020; the Hyperion scenes are from 2013-2017.
  - Pairs with a shift larger than 3 px: none.
  - PASS means "passed these checks", not perfect data. Hyperion pairs are 30 m; EMIT pairs are 60 m.
