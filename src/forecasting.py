"""Forecasting with Holt-Winters, validated against a seasonal-naive baseline."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _model(series: pd.Series):
    return ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()


def forecast_region(series: pd.Series, horizon: int = 60, test_size: int = 36) -> dict:
    series = series.asfreq("MS")
    train, test = series.iloc[:-test_size], series.iloc[-test_size:]
    pred = _model(train).forecast(test_size)
    naive = series.shift(12).iloc[-test_size:]               # "same month last year"
    metrics = {
        "MAE": float(np.mean(np.abs(test - pred))),
        "RMSE": float(np.sqrt(np.mean((test - pred) ** 2))),
        "MAE_seasonal_naive": float(np.mean(np.abs(test - naive))),
    }
    final = _model(series)
    fc = final.forecast(horizon)
    sigma = float(np.std(final.resid))
    return {"history": series, "forecast": fc, "lower": fc - 1.96 * sigma, "upper": fc + 1.96 * sigma,
            "metrics": metrics}
