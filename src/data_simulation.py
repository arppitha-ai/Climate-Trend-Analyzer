"""Generate a realistic, deliberately messy synthetic monthly climate dataset."""
import numpy as np
import pandas as pd

from .config import END, REGIONS, SEED, START


def simulate_climate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="MS")
    n = len(dates)
    t = np.arange(n) / 12.0                      # time in years
    month = dates.month.to_numpy()

    # Global-style series shared by all regions
    co2 = 338 + 1.5 * t + 0.012 * t**2 + 3 * np.sin(2 * np.pi * (month - 4) / 12) + rng.normal(0, 0.3, n)
    sea = 2.8 * t + 0.02 * t**2 + rng.normal(0, 3, n)          # mm relative to 1980

    frames = []
    for region, p in REGIONS.items():
        # Temperature = base + seasonal cycle + long-term warming + noise
        temp = (p["base"] + p["amp"] * np.sin(2 * np.pi * (month - 2) / 12)
                + p["warming"] * t + rng.normal(0, 0.6, n))
        # Rainfall = monsoon bump (circular distance to peak month) * trend * gamma noise
        d = np.minimum(np.abs(month - p["peak"]), 12 - np.abs(month - p["peak"]))
        rain = (5 + p["rain"] * np.exp(-0.5 * (d / 1.5) ** 2)) * (1 + p["rain_trend"] * t) * rng.gamma(4, 0.25, n)
        humidity = np.clip(45 + 0.08 * rain + rng.normal(0, 3, n), 20, 100)

        # Inject known anomalies (heat waves / cold snaps and extreme rain events)
        injected = np.zeros(n, dtype=bool)
        i_t = rng.choice(n, 10, replace=False)
        temp[i_t] += rng.choice([-1, 1], 10) * rng.uniform(3.5, 5.0, 10)
        i_r = rng.choice(n, 6, replace=False)
        rain[i_r] += rng.uniform(200, 350, 6)
        injected[i_t] = True
        injected[i_r] = True

        frames.append(pd.DataFrame({
            "date": dates, "region": region, "temperature_c": temp, "rainfall_mm": rain,
            "humidity_pct": humidity, "co2_ppm": co2, "sea_level_mm": sea,
            "injected_anomaly": injected}))

    df = pd.concat(frames, ignore_index=True)

    # Make it messy like real data: missing values, sentinel errors, duplicates
    df.loc[rng.choice(len(df), int(0.02 * len(df)), replace=False), "temperature_c"] = np.nan
    df.loc[rng.choice(len(df), int(0.01 * len(df)), replace=False), "rainfall_mm"] = np.nan
    df.loc[rng.choice(len(df), 5, replace=False), "temperature_c"] = -99.0
    df = pd.concat([df, df.sample(8, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
