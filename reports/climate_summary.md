# Climate Trend Analyzer - Summary Report

## Key insights
- Fastest warming region: **Central Plateau** at **0.284 °C/decade**.
- All regional temperature trends significant (Mann-Kendall p<0.05): **True**.
- National CO2 vs temperature correlation: **0.903**.
- Anomaly detection recall (combined methods): **0.875**.

## Trend table
```
         region      variable  slope_per_decade  r_squared  p_value  kendall_tau  mk_p_value  sen_slope_per_decade  significant
Central Plateau temperature_c             0.284      0.675 0.000000        0.648    0.000000                 0.279         True
  Coastal South temperature_c             0.192      0.509 0.000000        0.479    0.000004                 0.193         True
   North Plains temperature_c             0.241      0.530 0.000000        0.547    0.000000                 0.229         True
Central Plateau   rainfall_mm           -24.691      0.054 0.124845       -0.182    0.078269               -30.576        False
  Coastal South   rainfall_mm            61.164      0.074 0.071453        0.202    0.050411                67.564        False
   North Plains   rainfall_mm            32.339      0.043 0.172100        0.137    0.183388                34.973        False
```

## Forecast accuracy (36-month holdout)
```
         region   MAE  RMSE  MAE_seasonal_naive
Central Plateau 0.623 0.891               0.929
  Coastal South 0.510 0.657               0.789
   North Plains 0.449 0.563               0.697
```
