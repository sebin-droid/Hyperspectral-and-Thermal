"""ERA5 weather at the Landsat overpass (Open-Meteo historical archive, no account needed).

Reflectance says what the surface is, not how hot the day was: most LST variance is between scenes. Only cell
coordinates and dates are sent. Raw hourly series are cached, so each pair is requested once.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation", "cloud_cover"]
LANDSAT_LOCAL_SOLAR_TIME = 10.2          # Landsat 8/9 equator crossing ~10:11


def overpass_utc(date, lon):
    return pd.Timestamp(date) + pd.Timedelta(hours=(LANDSAT_LOCAL_SOLAR_TIME - lon / 15.0) % 24)


def _request(lat, lon, date, session):
    d = pd.Timestamp(date)
    q = dict(latitude=lat, longitude=lon, hourly=",".join(HOURLY), timezone="GMT",
             start_date=str((d - pd.Timedelta(days=2)).date()), end_date=str((d + pd.Timedelta(days=1)).date()))
    for k in range(6):
        try:
            r = session.get("https://archive-api.open-meteo.com/v1/archive", params=q, timeout=60)
            if r.status_code == 200:
                return r.json()["hourly"]
        except requests.exceptions.RequestException:
            pass
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"Open-Meteo failed for {lat}, {lon}, {date}")


def at(hourly, when):
    """MET_INPUTS at `when` (UTC): interpolated hourly values + mean air temperature of the previous 24 h."""
    t = pd.to_datetime(hourly["time"])
    sec = (t - t[0]).total_seconds().to_numpy()
    x = (when - t[0]).total_seconds()
    v = lambda k: float(np.interp(x, sec, np.asarray(hourly[k], "float64")))
    prev = (t > when - pd.Timedelta(hours=24)) & (t <= when)
    return [v("temperature_2m"), v("relative_humidity_2m"), v("wind_speed_10m"), v("shortwave_radiation"),
            v("cloud_cover"), float(np.mean(np.asarray(hourly["temperature_2m"], "float64")[prev]))]


def for_pairs(pairs, cache_file):
    """{pair_id: MET_INPUTS} for rows with pair_id, cell_lat, cell_lon, landsat_date."""
    cache_file = Path(cache_file)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    s = requests.Session()
    out = {}
    for p in pairs.itertuples():
        if p.pair_id not in cache:
            cache[p.pair_id] = _request(p.cell_lat, p.cell_lon, p.landsat_date, s)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache))
            time.sleep(0.3)
        out[p.pair_id] = at(cache[p.pair_id], overpass_utc(p.landsat_date, p.cell_lon))
    return out
