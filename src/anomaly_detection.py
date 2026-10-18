"""Anomaly detection: seasonal robust z-score + Isolation Forest, with evaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score - resistant to the outliers we are trying to find."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def _detrend(s: pd.Series) -> pd.Series:
    x = np.arange(len(s))
    return s - np.polyval(np.polyfit(x, s.values, 1), x)


def detect_anomalies(df: pd.DataFrame, temp_thresh: float = 4.0, rain_thresh: float = 5.0,
                     contamination: float = 0.012, seed: int = 42) -> pd.DataFrame:
    df = df.sort_values(["region", "date"]).copy()
    # Remove the warming trend so a slowly warming climate is not flagged as anomalous
    df["temp_resid"] = df.groupby("region")["temp_anomaly"].transform(_detrend)
    df["temp_z"] = df.groupby("region")["temp_resid"].transform(robust_z)
    # Rainfall is highly seasonal, so compare each month only with the same month in other years
    df["rain_z"] = df.groupby(["region", "month"])["rainfall_mm"].transform(robust_z)
    df["flag_zscore"] = (df["temp_z"].abs() > temp_thresh) | (df["rain_z"] > rain_thresh)

    df["flag_iforest"] = False
    for region, idx in df.groupby("region").groups.items():
        X = df.loc[idx, ["temp_resid", "rain_z", "humidity_pct"]]
        pred = IsolationForest(contamination=contamination, random_state=seed).fit_predict(X)
        df.loc[idx, "flag_iforest"] = pred == -1
    df["flag_any"] = df["flag_zscore"] | df["flag_iforest"]
    df["anomaly_type"] = np.select(
        [df["temp_z"] > temp_thresh, df["temp_z"] < -temp_thresh, df["rain_z"] > rain_thresh],
        ["Heat anomaly", "Cold anomaly", "Extreme rainfall"], default="Multivariate outlier")
    return df


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Precision/recall against the anomalies we injected in the simulation."""
    rows = []
    for col in ["flag_zscore", "flag_iforest", "flag_any"]:
        tp = int((df[col] & df["injected_anomaly"]).sum())
        fp = int((df[col] & ~df["injected_anomaly"]).sum())
        fn = int((~df[col] & df["injected_anomaly"]).sum())
        rows.append(dict(method=col, true_positives=tp, false_positives=fp, false_negatives=fn,
                         precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3)))
    return pd.DataFrame(rows)
