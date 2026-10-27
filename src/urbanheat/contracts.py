"""Interfaces shared between members (see docs/CONTRACTS.md). Change only by team agreement."""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# Keys of a model input batch. Every array is float32, shape (B, C, H, W) unless noted, already normalised.
HS = "hs"            # hyperspectral reflectance on the common wavelength grid
LS = "ls"            # Landsat blue, green, red, nir, swir1, swir2, NDVI, NDBI, MNDWI
ENV = "env"          # elevation_m, slope_deg, built_share
MET = "met"          # (B, 6) weather at the Landsat overpass
SENSOR = "sensor"    # (B,) int: index into SENSORS
MASK = "mask"        # (B, H, W) bool, valid pixels

LS_INPUTS = ["blue", "green", "red", "nir", "swir1", "swir2", "NDVI", "NDBI", "MNDWI"]
ENV_INPUTS = ["elevation_m", "slope_deg", "built_share"]
MET_INPUTS = ["air_temp_c", "rel_humidity", "wind_speed_kmh", "solar_rad_wm2", "cloud_cover_pct", "air_temp_prev24h_c"]
SENSORS = ["EMIT", "ENMAP", "HYPERION"]

# Surface factors a scenario may change (mitigation levers); everything else is held fixed.
LEVERS = {"NDVI": (-1.0, 1.0), "built_share": (0.0, 1.0), "albedo": (0.0, 1.0), "MNDWI": (-1.0, 1.0)}


class HeatModel(Protocol):
    """What Member 2 trains, Member 3 compresses and Member 4 explains and serves."""

    name: str
    inputs: list[str]          # subset of HS, LS, ENV, MET, SENSOR the model reads
    target: str                # "anomaly" or "absolute" (deg C)

    def predict(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """(B, H, W) predicted LST in deg C; invalid pixels may hold any value."""
        ...


@dataclass
class Scenario:
    """A what-if change to surface factors, e.g. Scenario("greening", {"NDVI": +0.1}, region="district.geojson")."""

    name: str
    changes: dict[str, float]               # lever -> additive change
    region: str | None = None               # GeoJSON path limiting where it applies; None = whole scene
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.changes) - set(LEVERS)
        if unknown:
            raise ValueError(f"not a mitigation lever: {sorted(unknown)}; allowed {sorted(LEVERS)}")
