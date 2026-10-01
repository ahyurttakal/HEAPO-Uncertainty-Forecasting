# HEAPO leakage-resistant analysis protocol

This document defines the information boundaries and reporting rules for the
public analysis. Results, checkpoints, or cached artifacts from another
configuration must not be mixed with a current run.

## Information boundaries

| Partition | Permitted use | Required target boundary |
| --- | --- | --- |
| Development households, training segment | Encoder fitting and model estimation | Every target is at or before `train_end` |
| Household-disjoint validation homes | LightGBM early stopping and model selection only | Origin and every target are at or before `train_end` |
| Seen-household calibration | Conformal, cold-condition, and portfolio thresholds only | Origin is after `train_end`; every target is at or before `calibration_end` |
| Seen-household test | Locked within-household evaluation | Origin is after `calibration_end` |
| Unseen-household test | Locked household-generalisation evaluation | Origin is after `calibration_end` |

All 96 target timestamps must remain within the split assigned to the origin.
Rows that cross a split boundary are excluded. Predictive preprocessing is fit
on training rows only. TFT early stopping uses only the final pre-specified
portion of its training segment; calibration and test targets are excluded.

## Missingness and short gaps

Forecast origins and scored targets must be observed in the source meter data.
The default `data.short_gap_steps: 1` permits interpolation of one isolated
15-minute gap only when constructing lagged and rolling history features. It
does not convert an imputed point into a valid forecast origin or scored
target, and longer gaps are left missing.

## Weather provenance

The primary operational analysis uses only weather information available at
the forecast origin. Realised target-time weather is reserved for explicitly
labelled oracle or ex-post analyses and cannot select a model or calibrate an
operational interval.

## Interval reporting

Figure 3 and all manuscript-wide coverage statements use the
`common_tft_tabular` and `macro_model_stratum_horizon_cells` row in
`metrics/calibration_index_audit.*`. The audit reports raw and calibrated PICP,
absolute deviation from nominal coverage, interval width, Winkler score, and
WIS. Dense-index results are labelled sensitivity estimates and cannot replace
the common-index values in the manuscript.

## Operational diagnostic

Median and conformal-upper-bound signals are deployable forecast scenarios.
Perfect foresight is a non-deployable diagnostic: realised future load is
passed through the same source selection, destination, capacity, acceptance,
flexibility, and energy-conservation constraints. This separates losses due to
forecast error from limitations of the dispatch formulation.

The sensitivity grid crosses 0.90, 0.95, and 0.99 peak-threshold quantiles with
5%–20% flexible fractions and 1–3 h shifting windows. The pre-specified primary
scenario is a 0.95 threshold, 20% flexible fraction, and 3 h window.

## Fail-closed checks

- `splits/leakage_audit.json` verifies household, time, horizon, clock-stratum,
  and complete-trajectory boundaries.
- `splits/leakage_audit_boundaries.csv` records the corresponding split-level
  boundaries.
- `metrics/calibration_index_audit.json` prevents common/dense PICP mixing.
- `portfolios/portfolio_protocol.json` records portfolio decision rules and
  labels perfect foresight as non-deployable.
- `validation_report.json` must report `PASS` before manuscript export.

## Local execution

Copy `configs/local.windows.example.yaml` to `configs/local.yaml`, set
`data.root` to the local HEAPO folder, and run:

```powershell
py -3.11 -m pip install -e ".[tft,manuscript]"
.\scripts\run_all.cmd .\configs\local.yaml
```
