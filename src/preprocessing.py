"""Data cleaning: dates, duplicates, impossible values, missing values."""
import numpy as np
import pandas as pd

NUMERIC = ["temperature_c", "rainfall_mm", "humidity_pct", "co2_ppm", "sea_level_mm"]


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts of common data-quality problems."""
    return pd.DataFrame({
        "missing": df[NUMERIC].isna().sum(),
        "impossible": pd.Series({
            "temperature_c": int((~df["temperature_c"].between(-60, 60) & df["temperature_c"].notna()).sum()),
            "rainfall_mm": int((df["rainfall_mm"] < 0).sum()),
            "humidity_pct": int((~df["humidity_pct"].between(0, 100) & df["humidity_pct"].notna()).sum()),
            "co2_ppm": 0, "sea_level_mm": 0}),
    }).assign(duplicates_in_file=int(df.duplicated(["date", "region"]).sum()))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date", "region"])
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    # Impossible physical values -> NaN
    df.loc[~df["temperature_c"].between(-60, 60), "temperature_c"] = np.nan
    df.loc[df["rainfall_mm"] < 0, "rainfall_mm"] = np.nan
    df.loc[~df["humidity_pct"].between(0, 100), "humidity_pct"] = np.nan

    # Time-aware gap filling inside each region
    df[NUMERIC] = df.groupby("region")[NUMERIC].transform(lambda s: s.interpolate(limit_direction="both"))
    return df
