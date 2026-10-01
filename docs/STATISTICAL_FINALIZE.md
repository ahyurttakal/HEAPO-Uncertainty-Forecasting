# Statistical finalization

The `finalize` stage calculates the pre-specified statistical summaries from
cached prediction outputs. It does not retrain models or regenerate forecasts.

Run from the repository root:

```powershell
py -3.11 -m heapo_forecasting.cli finalize --config .\configs\local.yaml
```

## Main outputs

- `outputs/heapo_analysis/final_statistical_report.md`
- `outputs/heapo_analysis/statistical_finalize.json`
- `outputs/heapo_analysis/metrics/household_cluster_metric_intervals.parquet`
- `outputs/heapo_analysis/metrics/nested_household_day_metric_intervals.parquet`
- `outputs/heapo_analysis/metrics/household_cluster_paired_tests.parquet`
- `outputs/heapo_analysis/metrics/nested_household_day_paired_tests.parquet`
- `outputs/heapo_analysis/metrics/common_index_household_cluster_metric_intervals.parquet`
- `outputs/heapo_analysis/metrics/common_index_nested_household_day_metric_intervals.parquet`
- `outputs/heapo_analysis/metrics/common_index_household_cluster_tft_paired_tests.parquet`
- `outputs/heapo_analysis/metrics/common_index_nested_household_day_tft_paired_tests.parquet`
- `outputs/heapo_analysis/portfolios/portfolio_calibration_sensitivity.csv`

The primary inference resamples complete households. Nested household/local-day
resampling is a sensitivity analysis and does not increase the number of
independent households. Holm-adjusted significance is called confirmatory only
for strata containing at least 20 independent households.
