"""Interactive dashboard:  streamlit run app/streamlit_app.py  (run main.py first)"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "climate_clean.csv"
ANOM = ROOT / "outputs" / "tables" / "anomalies.csv"
TREND = ROOT / "outputs" / "tables" / "trend_summary.csv"

st.set_page_config(page_title="Climate Trend Analyzer", layout="wide")
px.defaults.color_discrete_sequence = px.colors.qualitative.Bold
st.title("🌍 Climate Trend Analyzer")

if not DATA.exists():
    st.error("Processed data not found. Run `python main.py` first.")
    st.stop()

df = pd.read_csv(DATA, parse_dates=["date"])
anom = pd.read_csv(ANOM, parse_dates=["date"])
trend = pd.read_csv(TREND)

regions = st.sidebar.multiselect("Regions", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
years = st.sidebar.slider("Year range", int(df["year"].min()), int(df["year"].max()),
                          (int(df["year"].min()), int(df["year"].max())))
d = df[df["region"].isin(regions) & df["year"].between(*years)]

c1, c2, c3 = st.columns(3)
t = trend[(trend["variable"] == "temperature_c") & trend["region"].isin(regions)]
if not t.empty:
    c1.metric("Avg warming (°C/decade)", f"{t['slope_per_decade'].mean():.2f}")
c2.metric("Latest 12-mo avg temp (°C)", f"{d.sort_values('date').groupby('region')['temp_roll12'].last().mean():.2f}")
c3.metric("Anomalies flagged", int(anom[anom["region"].isin(regions) & anom["date"].dt.year.between(*years)].shape[0]))

st.plotly_chart(px.line(d, x="date", y="temp_roll12", color="region", title="Temperature trend (12-month average)"),
                use_container_width=True)
st.plotly_chart(px.line(d.groupby(["year", "region"])["rainfall_mm"].sum().reset_index(), x="year", y="rainfall_mm",
                        color="region", title="Annual rainfall (mm)"), use_container_width=True)
st.plotly_chart(px.density_heatmap(d.groupby(["year", "month"])["temp_anomaly"].mean().reset_index(), x="year", y="month",
                                   z="temp_anomaly", histfunc="avg", color_continuous_scale="RdBu_r",
                                   title="Temperature anomaly: month vs year"), use_container_width=True)
st.subheader("Detected anomalies")
st.dataframe(anom[anom["region"].isin(regions)], use_container_width=True)
st.subheader("Trend statistics")
st.dataframe(trend, use_container_width=True)
