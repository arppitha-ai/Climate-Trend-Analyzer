"""All charts. Each function saves a PNG into outputs/figures/."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import FIG_DIR

sns.set_theme(style="whitegrid")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def plot_temperature_trend(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    for region, g in df.groupby("region"):
        line, = ax.plot(g["date"], g["temperature_c"], alpha=0.25, lw=0.8)
        ax.plot(g["date"], g["temp_roll12"], color=line.get_color(), lw=2, label=f"{region} (12-mo avg)")
    ax.set(title="Monthly Temperature with 12-Month Rolling Mean", xlabel="Year", ylabel="Temperature (°C)")
    ax.legend()
    _save(fig, "01_temperature_trend.png")


def plot_rainfall_trend(df):
    annual = df.groupby(["region", "year"])["rainfall_mm"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for region, g in annual.groupby("region"):
        line, = ax.plot(g["year"], g["rainfall_mm"], marker="o", ms=3, alpha=0.6, label=region)
        lr = stats.linregress(g["year"], g["rainfall_mm"])
        ax.plot(g["year"], lr.intercept + lr.slope * g["year"], "--", color=line.get_color())
    ax.set(title="Annual Rainfall Totals with Linear Trend", xlabel="Year", ylabel="Rainfall (mm/year)")
    ax.legend()
    _save(fig, "02_rainfall_trend.png")


def plot_seasonal_pattern(df):
    clim = df.groupby(["region", "month"])[["temperature_c", "rainfall_mm"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(data=clim, x="month", y="temperature_c", hue="region", marker="o", ax=axes[0])
    sns.lineplot(data=clim, x="month", y="rainfall_mm", hue="region", marker="o", ax=axes[1])
    axes[0].set(title="Average Monthly Temperature", ylabel="°C")
    axes[1].set(title="Average Monthly Rainfall", ylabel="mm")
    for ax in axes:
        ax.set_xticks(range(1, 13))
    _save(fig, "03_seasonal_pattern.png")


def plot_heatmap(df, region):
    pivot = df[df["region"] == region].pivot_table(index="month", columns="year", values="temp_anomaly")
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(pivot, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Anomaly (°C)"})
    ax.set_title(f"Temperature Anomaly Heatmap (month x year) - {region}")
    _save(fig, "04_temperature_heatmap.png")


def plot_anomalies(df):
    regions = df["region"].unique()
    fig, axes = plt.subplots(len(regions), 1, figsize=(13, 3.6 * len(regions)), sharex=True)
    for ax, region in zip(np.atleast_1d(axes), regions):
        g = df[df["region"] == region]
        ax.plot(g["date"], g["temp_resid"], color="steelblue", lw=0.8)
        f = g[g["flag_any"]]
        ax.scatter(f["date"], f["temp_resid"], color="red", s=28, zorder=3, label="Flagged anomaly")
        ax.set(title=region, ylabel="Detrended anomaly (°C)")
        ax.legend(loc="upper left")
    _save(fig, "05_anomaly_detection.png")


def plot_decade_comparison(decades):
    ax = decades.plot(kind="bar", figsize=(10, 5))
    ax.set(title="Mean Temperature Anomaly by Decade (vs 1981-2010)", xlabel="Decade", ylabel="Anomaly (°C)")
    ax.axhline(0, color="black", lw=0.8)
    _save(ax.figure, "06_yearly_comparison.png")


def plot_co2_vs_temp(annual):
    lr = stats.linregress(annual["co2_ppm"], annual["temperature_c"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(annual["co2_ppm"], annual["temperature_c"], c=annual.index, cmap="viridis")
    xs = np.linspace(annual["co2_ppm"].min(), annual["co2_ppm"].max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, "r--", label=f"r = {lr.rvalue:.2f}")
    fig.colorbar(sc, label="Year")
    ax.set(title="CO₂ vs Mean Temperature (annual)", xlabel="CO₂ (ppm)", ylabel="Temperature (°C)")
    ax.legend()
    _save(fig, "07_co2_vs_temperature.png")


def plot_forecast(result, region):
    hist = result["history"].iloc[-180:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist.index, hist.values, label="Observed")
    ax.plot(result["forecast"].index, result["forecast"].values, color="red", label="Forecast (5 yrs)")
    ax.fill_between(result["forecast"].index, result["lower"], result["upper"], color="red", alpha=0.15, label="95% band")
    m = result["metrics"]
    ax.set(title=f"Temperature Forecast - {region} (holdout MAE {m['MAE']:.2f}°C vs naive {m['MAE_seasonal_naive']:.2f}°C)",
           ylabel="Temperature (°C)")
    ax.legend()
    _save(fig, f"08_forecast_{region.lower().replace(' ', '_')}.png")
