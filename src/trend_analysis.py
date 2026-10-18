"""Long-term trend statistics: OLS, Mann-Kendall (Kendall tau), Sen's slope, correlations."""
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def annual_trends(df: pd.DataFrame, col: str, agg: str = "mean") -> pd.DataFrame:
    rows = []
    for region, g in df.groupby("region"):
        a = g.groupby("year")[col].agg(agg)
        lr = stats.linregress(a.index, a.values)
        tau, p_mk = stats.kendalltau(a.index, a.values)   # Mann-Kendall equivalent for time trends
        sen = stats.theilslopes(a.values, a.index)[0]      # robust slope
        rows.append(dict(region=region, variable=col, slope_per_decade=round(lr.slope * 10, 3),
                         r_squared=round(lr.rvalue ** 2, 3), p_value=round(lr.pvalue, 6),
                         kendall_tau=round(tau, 3), mk_p_value=round(p_mk, 6),
                         sen_slope_per_decade=round(sen * 10, 3), significant=bool(p_mk < 0.05)))
    return pd.DataFrame(rows)


def decade_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["decade", "region"])["temp_anomaly"].mean().unstack("region").round(2)


def national_annual(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("year").agg(temperature_c=("temperature_c", "mean"), rainfall_mm=("rainfall_mm", "mean"),
                                  co2_ppm=("co2_ppm", "mean"), sea_level_mm=("sea_level_mm", "mean"))


def correlation_table(annual: pd.DataFrame) -> pd.DataFrame:
    return annual.corr().round(3)


def decompose(df: pd.DataFrame, region: str):
    s = df[df["region"] == region].set_index("date")["temperature_c"].asfreq("MS")
    return seasonal_decompose(s, model="additive", period=12)
