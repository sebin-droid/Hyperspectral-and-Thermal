# Urban heat knowledge base

Source text for the dashboard assistant (retrieval-augmented generation, `urbanheat.app.assistant`). Each `##`
section is one retrievable passage, so keep sections short and self-contained.

## Land surface temperature (LST)

LST is the radiative temperature of the ground, roofs, roads and canopy, measured from space in the thermal
infrared (Landsat 8/9 band 10, Collection 2 Level-2 `ST_B10`). It is not air temperature: on a sunny afternoon a
dark roof can be 20–30 °C hotter than the air above it, while air temperature varies much less across a city.
LST tells where surfaces store and re-emit solar heat. It is a proxy for heat exposure, not a measure of health risk.

## LST anomaly: what the model predicts

The models predict the LST anomaly: a pixel's LST minus the median LST of its city scene on that date, in °C.
A value of +3 °C means the pixel is 3 °C hotter than the typical pixel of the same city at the same time. Using
the anomaly removes weather and season, which differ between dates, and keeps the within-city pattern that
land cover controls. An anomaly is relative: changing a whole city by the same amount does not move it.

## Surface urban heat island (SUHI)

The surface urban heat island is the LST difference between built-up land and its rural surroundings. In this
project it is measured as built-up LST minus the LST of a rural ring 1–6 km outside the city (Member 1's change
data, `suhi_c`). Causes: impervious materials that absorb and store heat, little evaporation from sealed
surfaces, less vegetation and shade, and heat from traffic and buildings.

## NDVI: vegetation

NDVI = (NIR − red) / (NIR + red), from −1 to 1. Dense green vegetation is about 0.6–0.9, sparse grass 0.2–0.4,
bare soil and built-up surfaces about 0–0.2, water below 0. Vegetation cools the surface by evapotranspiration
and shade, so higher NDVI usually means lower LST. The cooling is weaker in dry seasons, when plants close their
stomata.

## NDBI: built-up surfaces

NDBI = (SWIR1 − NIR) / (SWIR1 + NIR), from −1 to 1. High values mark built-up and bare surfaces (concrete,
asphalt, roofs, dry soil); vegetation and water give negative values. A high NDBI value usually means a hotter
surface, because sealed materials heat quickly and do not evaporate water. NDBI also responds to bare dry soil,
so it is not a pure measure of buildings.

## MNDWI: water

MNDWI = (green − SWIR1) / (green + SWIR1), from −1 to 1. Positive values mark open water. Water bodies have a low
daytime LST because water has a high heat capacity and evaporates; they cool their immediate surroundings.

## Built share

`built_share` is the fraction (0–1) of a 60 m pixel covered by built-up land in ESA WorldCover. It measures how
much of the pixel is buildings and pavement. More built share usually means a hotter surface.

## Thermal context

The thermal context input is the median LST of other Landsat dates in the same season (±60 days of the year),
leaving out ±16 days around the target date, so the target is never seen. It tells the model where a city is
usually hot. It is the strongest single input (alone it explains about 73 % of the anomaly variance with a
linear model) but it explains little about why a place is hot: it carries the effect of land cover from earlier
dates.

## Elevation and slope

Elevation and slope come from the Copernicus GLO-30 DEM. Elevation is given relative to the city median. Higher
ground is often cooler (about −6.5 °C per km in air temperature), and slopes facing the sun heat more. In flat
cities they matter little.

## Hyperspectral data

The hyperspectral sensors (EMIT on the ISS, EnMAP, Hyperion) measure reflectance in about 200–285 narrow bands
from 380 to 2500 nm, harmonised in this project to 122 bands on a 10 nm grid at 60 m. They can separate
materials (roof types, asphalt, soils, vegetation stress) that broad Landsat bands cannot. In this project they
did not improve the LST anomaly prediction significantly over Landsat bands plus indices.

## The project models

Member 2's fusion CNN (`fusion_cnn`) combines a surface branch (hyperspectral, Landsat bands and indices, terrain,
built share) and a thermal branch (thermal context) by concatenating their feature maps, with a per-sensor
embedding. It predicts the anomaly for every pixel; test R² 0.78 on 56 cities never seen in training. Member 3's
distilled small model `m3_lite16_kd` keeps the same structure with depthwise-separable convolutions: 7,905
parameters (37× fewer), 97 % fewer FLOPs, test R² 0.784, about 0.5 ms per 32×32 patch on a laptop CPU.

## Accuracy of the predictions

On the test cities the small model's mean absolute error is about 1.0 °C and RMSE 1.4 °C for the LST anomaly.
It is most accurate in temperate and tropical cities and least accurate in cold cities (few training examples).
Accuracy also varies by sensor: EnMAP scenes are predicted best, Hyperion scenes worst.

## What drives urban heat in this data

Member 2's factor analysis (XGBoost with exact TreeSHAP on the pixel table) found built-up factors (NDBI, built
share) to account for about half of the explained variation of the LST anomaly, followed by vegetation (NDVI),
water and terrain. Correlated indices share credit, so factors are read in groups.

## Observed effect of land-cover change

Member 1's change data (Dynamic World land cover and hot-season Landsat LST, 2016–2025) show that pixels changing
from green to built warmed by about 0.36 °C more than unchanged green pixels (95 % CI 0.29–0.44), and pixels
changing from built to green cooled by about 0.42 °C compared with unchanged built pixels (95 % CI 0.38–0.49).

## What-if scenarios

The scenario engine changes surface factors (levers NDVI, NDBI, MNDWI, built share) inside a region, predicts
again, and maps ΔLST = prediction(changed) − prediction(current) per pixel. Typical changes taken from the
observed data: built → green NDVI +0.34, NDBI −0.20, built share −0.30; greening NDVI +0.1, NDBI −0.06. The
small model predicts −0.15 °C for built → green and +0.20 °C for green → built, the same direction as the
observed change and about half its size.

## Limits of the scenario results

Scenario maps are modelled estimates, not proof of cause. The response is damped (about half the observed size)
because the thermal context and the linked inputs (red/NIR bands, hyperspectral bands) stay unchanged. The model
predicts surface, not air, temperature. Each model was trained once (one seed). Albedo is a contract lever but
not a model input, so it cannot be tested.

## Explanations: integrated gradients and Grad-CAM

The dashboard explains the CNN with integrated gradients, which attribute the predicted anomaly to each input
group in °C (positive: the input makes the place hotter than the city median; negative: cooler). They add up to
the prediction minus the prediction for an average input. Grad-CAM highlights the places each branch (surface,
thermal) relied on for a region's prediction. Both describe what the model learned, which can include
correlations that are not causes.

## Mitigation options for surface heat

Common ways to lower surface temperature: trees and green space (shade and evaporation), green roofs, cool
(high-albedo) roofs and pavements, protecting water bodies and wetlands, permeable surfaces that hold moisture,
and keeping ventilation corridors open. Trees give the largest daytime surface cooling per area; cool roofs act
on roofs only. Effects depend on climate: vegetation cools most where water is available.
