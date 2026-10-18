"""Central configuration: paths, random seed, simulated regions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SEED = 42
START, END = "1980-01-01", "2024-12-01"
BASELINE = (1981, 2010)          # climatology baseline period (like WMO normals)

# Each region: base temp (C), seasonal amplitude (C), warming (C/year),
# peak monsoon rainfall (mm), peak month, rainfall trend (fraction/year)
REGIONS = {
    "North Plains":    dict(base=24.0, amp=10.0, warming=0.028, rain=180, peak=7, rain_trend=0.002),
    "Coastal South":   dict(base=27.5, amp=2.5,  warming=0.020, rain=320, peak=6, rain_trend=0.004),
    "Central Plateau": dict(base=25.5, amp=6.0,  warming=0.024, rain=150, peak=8, rain_trend=-0.003),
}
