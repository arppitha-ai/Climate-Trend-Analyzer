"""Run the full Climate Trend Analyzer pipeline:  python main.py"""
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows

import pandas as pd

from src import (anomaly_detection, config, data_simulation, features, forecasting,
                 preprocessing, trend_analysis)
from src import visualization as viz


def main():
    for d in (config.RAW_DIR, config.PROC_DIR, config.FIG_DIR, config.TAB_DIR, config.REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/7] Simulating climate dataset ...")
    raw = data_simulation.simulate_climate()
    raw.to_csv(config.RAW_DIR / "climate_raw.csv", index=False)
    print(f"      rows={len(raw)}  regions={raw['region'].nunique()}")

    print("[2/7] Data quality report (before cleaning) ...")
    qr = preprocessing.quality_report(raw)
    qr.to_csv(config.TAB_DIR / "data_quality_report.csv")
    print(qr.to_string())

    print("[3/7] Cleaning + feature engineering ...")
    df = features.add_features(preprocessing.clean(raw))
    df.to_csv(config.PROC_DIR / "climate_clean.csv", index=False)
    print(f"      clean rows={len(df)}  missing after cleaning={int(df[preprocessing.NUMERIC].isna().sum().sum())}")

    print("[4/7] Trend analysis ...")
    trends = pd.concat([trend_analysis.annual_trends(df, "temperature_c"),
                        trend_analysis.annual_trends(df, "rainfall_mm", agg="sum")], ignore_index=True)
    trends.to_csv(config.TAB_DIR / "trend_summary.csv", index=False)
    print(trends[["region", "variable", "slope_per_decade", "r_squared", "mk_p_value", "significant"]].to_string(index=False))
    decades = trend_analysis.decade_means(df)
    decades.to_csv(config.TAB_DIR / "decade_anomalies.csv")
    annual = trend_analysis.national_annual(df)
    corr = trend_analysis.correlation_table(annual)
    corr.to_csv(config.TAB_DIR / "correlations.csv")

    print("[5/7] Anomaly detection ...")
    det = anomaly_detection.detect_anomalies(df)
    ev = anomaly_detection.evaluate(det)
    ev.to_csv(config.TAB_DIR / "anomaly_evaluation.csv", index=False)
    print(ev.to_string(index=False))
    cols = ["date", "region", "temperature_c", "rainfall_mm", "temp_z", "rain_z", "anomaly_type"]
    anomalies = det[det["flag_any"]][cols].round({"temperature_c": 2, "rainfall_mm": 2, "temp_z": 2, "rain_z": 2}).sort_values("date")
    anomalies.to_csv(config.TAB_DIR / "anomalies.csv", index=False)
    print(f"      flagged {len(anomalies)} anomalous months")

    print("[6/7] Forecasting ...")
    fc_rows = []
    for region in df["region"].unique():
        res = forecasting.forecast_region(df[df["region"] == region].set_index("date")["temperature_c"])
        viz.plot_forecast(res, region)
        fc_rows.append({"region": region, **{k: round(v, 3) for k, v in res["metrics"].items()}})
        res["forecast"].rename("forecast_c").to_csv(config.TAB_DIR / f"forecast_{region.lower().replace(' ', '_')}.csv")
    fc = pd.DataFrame(fc_rows)
    fc.to_csv(config.TAB_DIR / "forecast_metrics.csv", index=False)
    print(fc.to_string(index=False))

    print("[7/7] Charts + report ...")
    viz.plot_temperature_trend(df)
    viz.plot_rainfall_trend(df)
    viz.plot_seasonal_pattern(df)
    viz.plot_heatmap(df, df["region"].iloc[0])
    viz.plot_anomalies(det)
    viz.plot_decade_comparison(decades)
    viz.plot_co2_vs_temp(annual)

    t = trends[trends["variable"] == "temperature_c"].sort_values("slope_per_decade", ascending=False)
    report = f"""# Climate Trend Analyzer - Summary Report

## Key insights
- Fastest warming region: **{t.iloc[0]['region']}** at **{t.iloc[0]['slope_per_decade']} °C/decade**.
- All regional temperature trends significant (Mann-Kendall p<0.05): **{bool(t['significant'].all())}**.
- National CO2 vs temperature correlation: **{corr.loc['co2_ppm', 'temperature_c']}**.
- Anomaly detection recall (combined methods): **{ev.loc[ev['method'] == 'flag_any', 'recall'].iloc[0]}**.

## Trend table
```
{trends.to_string(index=False)}
```

## Forecast accuracy (36-month holdout)
```
{fc.to_string(index=False)}
```
"""
    (config.REPORT_DIR / "climate_summary.md").write_text(report, encoding="utf-8")
    print("\nDone. See outputs/figures, outputs/tables and reports/climate_summary.md")


if __name__ == "__main__":
    main()
