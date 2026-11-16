# Urban heat knowledge base

Source text for the dashboard assistant (`urbanheat.app.assistant`). Each `##` section is one passage. The first
two sentences of a section are the plain answer shown to users, so they must make sense to anyone (no jargon);
details for experts follow.

## Is this the air temperature? (land surface temperature, LST)

No. The maps show how hot the ground, roofs and roads are, measured by a satellite from space, not the air
temperature you feel or read on a thermometer. On a sunny afternoon a dark roof can be 20–30 °C hotter than the air
above it, so these numbers are often much higher than the weather forecast. Details: this is land surface
temperature (LST) from the thermal band of Landsat 8/9 (Collection 2 Level-2, band 10). It shows where surfaces
soak up and give off the sun's heat; it is a sign of heat exposure, not a measure of health risk.

## Hot and cool spots: what does "hotter than the city's typical" mean?

Each place is compared with the typical (middle) ground temperature of its own city on the same day. "+3 °C" means
that spot is 3 °C hotter than most of the city; "−3 °C" means 3 °C cooler. Details: the model predicts this
difference (the LST anomaly: pixel LST minus the scene median), which removes the weather of the day and keeps the
pattern that land cover controls. Because it is relative, changing a whole city by the same amount does not move it.

## Why are cities hotter than the countryside? (urban heat island)

Concrete, asphalt and roofs soak up sunlight and stay hot, while plants and wet soil cool themselves by giving off
water. So built-up areas with little greenery get hotter than the surrounding countryside. Details: the surface
urban heat island is measured here as built-up LST minus the LST of a rural ring 1–6 km outside the city. Traffic,
air conditioners and the lack of shade add to it.

## What does NDVI (greenery) mean?

NDVI is a number from −1 to 1 that tells how green and leafy a place is. Dense trees are about 0.6–0.9, grass
0.2–0.4, bare soil and buildings close to 0, and water below 0. Greener places are usually cooler, because plants
give shade and cool themselves by releasing water. Details: NDVI = (NIR − red) / (NIR + red); the cooling is weaker
in dry seasons, when plants save water.

## What does NDBI (built-up surfaces) mean?

NDBI is a number from −1 to 1 that is high for buildings, roads and bare dry ground, and low for plants and water. A
high NDBI usually means a hotter surface, because sealed materials heat up fast and cannot cool by evaporation.
Details: NDBI = (SWIR1 − NIR) / (SWIR1 + NIR); it also rises on bare dry soil, so it is not a pure measure of
buildings.

## What does MNDWI (water) mean?

MNDWI is a number from −1 to 1 that is above 0 where there is open water. Lakes, rivers and ponds stay cooler by day
and cool the land right next to them. Details: MNDWI = (green − SWIR1) / (green + SWIR1); water heats slowly and
loses heat by evaporation.

## What is "buildings and pavement" (built share)?

It is the share of the ground covered by buildings, roads and other built surfaces, from 0 (none) to 1 (fully built).
More built ground usually means a hotter surface. Details: `built_share` is the built-up fraction of each 60 m pixel
in the ESA WorldCover map.

## What is "heat seen on other days"?

It is how hot the same spot was in other satellite images of the same season. A place that was hot on other days
is usually hot again, so this is the model's strongest clue, but it does not say why the place is hot. Details: the
thermal context input is the median LST of other Landsat dates within ±60 days of the year, leaving out ±16 days
around the image date, so the answer is never seen. Alone it explains about 73 % of the differences.

## Height and slope

Higher ground is often a little cooler, and slopes facing the sun heat more. In flat cities they matter little.
Details: elevation and slope come from the Copernicus GLO-30 terrain model; elevation is given relative to the city.

## What are the detailed spectrum (hyperspectral) satellites?

Some satellites (EMIT on the space station, EnMAP, Hyperion) split sunlight into hundreds of narrow colours, which
can tell materials apart, such as roof types, asphalt, soils or stressed plants. In this project they did not make
the heat prediction clearly better than a normal Landsat satellite. Details: the images are harmonised to 122
bands from 420 to 2400 nm at 60 m.

## How does the model work?

A small neural network looks at a satellite picture of a place (greenery, buildings, water, materials, height and
past heat) and estimates how much hotter or cooler each 60 m patch of ground is than the city's typical value. It
learned this from 321 image pairs and was tested on 56 places it had never seen. Details: Member 3's distilled
model `m3_lite16_kd` (7,905 parameters, depthwise-separable two-branch CNN) keeps the accuracy of Member 2's fusion
CNN (test R² 0.78) with 37× fewer parameters and runs in about 0.5 ms per 32 × 32 patch on a laptop.

## How accurate is the model?

On places it never saw, the model is off by about 1 °C on average and explains about 78 % of the differences in
ground temperature within a city. It is most accurate in temperate and tropical cities and least accurate in cold
cities, where it had few examples. Details: test MAE 0.98 °C, RMSE 1.39 °C, R² 0.784; by satellite, EnMAP images are
predicted best and Hyperion images worst.

## What makes places hot in this data?

Buildings, roads and bare ground explain about half of the differences, followed by the lack of greenery, water and
height. These factors are linked (where there are buildings there are fewer plants), so they are best read together.
Details: Member 2's factor analysis with XGBoost and exact TreeSHAP on the pixel table; built-up factors (NDBI,
built share) account for about half of the explained variation.

## What happened when land really changed?

When green land was built on between 2016 and 2025, its ground warmed by about 0.4 °C more than green land that
stayed green. When built land turned green, it cooled by about 0.4 °C compared with land that stayed built.
Details: Member 1's change data (Dynamic World land cover and hot-season Landsat LST): green → built +0.36 °C
(95 % CI 0.29–0.44), built → green −0.42 °C (95 % CI 0.38–0.49).

## What do the "what if" buttons do?

They change the greenery and buildings inside the circle you chose and ask the model to estimate the ground
temperature again; the map then shows how much it would change. "Turn built-up land into a park" cools test places by
about 0.15 °C on average in this model, and "build on green land" warms them by about 0.2 °C. Details: the scenario
engine adds NDVI +0.34, NDBI −0.20, built share −0.30 for built → park (opposite for building on green land) and
NDVI +0.1, NDBI −0.06 for street trees, then maps ΔLST = prediction(changed) − prediction(current).

## Can I trust the "what if" results?

Treat them as a rough, cautious estimate, not a promise. In this model the effect is about half of what was seen
when land really changed, because the model leans on past heat, which a what-if cannot change. Details: the thermal
context and the linked inputs (red/NIR bands, hyperspectral bands) stay fixed in a scenario, which damps the response;
the model shows surface, not air, temperature; it was trained once (one seed); albedo is not a model input.

## How does the dashboard explain a spot?

It starts from an average place and shows how much each factor (greenery, buildings, water, past heat and so on)
pushes the spot's temperature up or down, in °C; together they add up to the model's estimate. These are the
model's reasons, learned from data, which can include patterns that are not real causes. Details: integrated
gradients for the mean prediction of the circle, from the train-mean input; Grad-CAM shows where each branch looked.

## How can a hot place be cooled?

Trees and parks cool the most, through shade and evaporation; green roofs, light-coloured (cool) roofs and
pavements, keeping lakes and wetlands, and ground that lets water soak in also help. Cooling from plants is
strongest where they have enough water. Details: cool roofs act only on roofs; ventilation corridors help air
temperature more than surface temperature.
