# HEAPO statistical finalization (v5)

This cumulative patch retains the portfolio memory fixes and adds a cached-result
`finalize` stage. It never trains or predicts again.

## Run

From the project root on Windows PowerShell:

```powershell
py -3.11 -m heapo_forecasting.cli finalize --config .\configs\local.yaml
```

## New outputs

- `outputs/heapo_main/final_statistical_report.md`
- `outputs/heapo_main/statistical_finalize.json`
- `outputs/heapo_main/metrics/household_cluster_metric_intervals.parquet`
- `outputs/heapo_main/metrics/nested_household_day_metric_intervals.parquet`
- `outputs/heapo_main/metrics/household_cluster_paired_tests.parquet`
- `outputs/heapo_main/metrics/nested_household_day_paired_tests.parquet`
- `outputs/heapo_main/metrics/common_index_household_cluster_metric_intervals.parquet`
- `outputs/heapo_main/metrics/common_index_nested_household_day_metric_intervals.parquet`
- `outputs/heapo_main/metrics/common_index_household_cluster_tft_paired_tests.parquet`
- `outputs/heapo_main/metrics/common_index_nested_household_day_tft_paired_tests.parquet`
- `outputs/heapo_main/portfolios/portfolio_calibration_sensitivity.parquet`
- `outputs/heapo_main/portfolios/portfolio_calibration_sensitivity.csv`

Primary inference resamples complete households. Nested household/local-day
resampling is a sensitivity analysis. Holm significance is labelled
confirmatory only for strata with at least 20 independent households.

