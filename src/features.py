"""Feature engineering: calendar fields, rolling means, climatology and anomalies."""
import pandas as pd

from .config import BASELINE

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    df["decade"] = (df["year"] // 10) * 10
    df["season"] = df["month"].map(SEASONS)

    g = df.groupby("region")
    df["temp_roll12"] = g["temperature_c"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    df["rain_roll12"] = g["rainfall_mm"].transform(lambda s: s.rolling(12, min_periods=12).mean())

    # Anomaly = value minus the average for that region+month in the baseline period
    base = df[df["year"].between(*BASELINE)].groupby(["region", "month"])[["temperature_c", "rainfall_mm"]].mean()
    base.columns = ["temp_clim", "rain_clim"]
    df = df.join(base, on=["region", "month"])
    df["temp_anomaly"] = df["temperature_c"] - df["temp_clim"]
    df["rain_anomaly"] = df["rainfall_mm"] - df["rain_clim"]
    return df
