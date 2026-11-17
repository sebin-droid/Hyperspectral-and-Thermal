PASSED-PAIRS: EnMAP + Landsat 8/9 pairs that passed the quality rules (QC v2.3, same rules as the Hyperion and EMIT pairs)
=========================================================================
Pairs: 31 in 23 cells (pair numbers as in ENMAP-LANDSAT/pairs/; gaps = not passed)
Climate groups (Koppen, Beck et al.): {'tropical': 31}

Each pair folder contains
  enmap_reflectance.tif  EnMAP L2A surface reflectance x 10000 (int16), nodata -32768, 177 bands, 30 m,
                         wavelength and EnMAP band number in each band description
  landsat_<id>.tif       Landsat 8/9 on the SAME 30 m EnMAP grid (float32, NaN = no data, bilinear):
                         blue, green, red, nir, swir1, swir2 (surface reflectance), LST_K (kelvin), clear_frac (share of clear Landsat pixels)
  pair_info.json         scene IDs, dates, sun and viewing angles, AOD, EnMAP processor version, Koppen code / climate group, QC numbers
handover_pass.csv lists every pair with cell, dates, angles, climate and QC numbers. enmap_bands.json is the band configuration.

How the pairs were made
  - EnMAP L2A surface reflectance (DLR, processor version in pair_info), read for the 7.7 km cell only; reflectance = value x 0.0001
  - bands: 224 -> 177 kept; dropped: oxygen 764 nm, water vapour 920-1004 and 1116-1152 nm, empty bands
    1343-1390 nm, the edges of the 1400 / 1900 nm gaps, 418-429 nm and 2415-2445 nm (noisy; data-driven, EN-05)
  - defective detector pixels (EnMAP defective_pixel_mask; not repaired by DLR) filled with the mean of good 3x3 (then 5x5) neighbours
    in the same band (test error about 5 %, equal to normal neighbour differences); share per pair: defective_filled_frac
  - EnMAP masks applied: cloud, cloud shadow, cirrus, snow. EnMAP's haze flag is recorded but NOT applied (it marked clear areas)
  - residual cloud and shadow masked by comparison with Landsat (5x5 median difference > 0.06 and factor 1.5,
    relative to the pair's own median ratio, buffered 2 px)
  - Landsat 8/9 L2 resampled bilinearly onto the EnMAP 30 m grid; where the screening's Landsat scene ended inside the cell, the
    neighbouring scene (same overpass, or within +/-15 days) was used instead
  - EnMAP shifted by whole pixels only where the NDVI correlation with Landsat clearly improved (search +-8 px)

PASS rules (design choices, not from the literature; identical for Hyperion, EMIT and EnMAP)
  joint coverage >= 0.5, NIR ratio within [0.85, 1.15], NDVI r after alignment >= 0.7, AOD <= 0.8,
  sun elevation >= 25.0 deg, Landsat clear >= 0.85, no unresolved shift, large shifts confirmed,
  NDVI gain (mean EnMAP NDVI / mean Landsat NDVI over Landsat NDVI > 0.3) >= 0.8.
  FLAGS only: blue_ok / red_ok / swir1_ok (ratio outside 0.85-1.15): blue 18, red 10, SWIR1 16 of 31; haze_mild: 1.

Known caveats
  - EnMAP is a pointing satellite (off-nadir up to about 25 deg). The EnMAP/Landsat NIR ratio rises with the signed across-track
    angle (Spearman about 0.6): most likely directional reflectance (BRDF), not bad data. Angles are in pair_info for a later correction.
  - EnMAP SWIR reads about 15-20 % higher than Landsat SWIR; which sensor is closer to the truth is not established.
  - Several scenes share a cell: split training and validation BY cell_id, not by pair.
  - All EnMAP pairs are humid tropical cells (profile humid_coastal). Large shifts: none.
  - PASS means "passed these checks", not perfect data.
