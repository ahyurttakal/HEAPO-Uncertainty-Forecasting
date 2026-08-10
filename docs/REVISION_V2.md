# HEAPO forecasting revision v2

This revision addresses the limitations found in the first completed run.

## Corrected

- MASE is based on a 96-step daily-naive error from full, continuous,
  pre-training household panels. Held-out-home fallback scales are derived only
  from training homes of the same heat-pump type.
- Unseen validation and unseen test homes now use the same chronological
  calibration/test boundaries as the seen cohort.
- Persistence, daily seasonal, weekly seasonal, Elastic Net, LightGBM, and TFT
  intervals are conformally calibrated by a hierarchical
  model/type/horizon/cold rule.
- Raw and calibrated interval metrics are separate artifacts.
- MAE and PICP uncertainty uses household-day block bootstrap; model differences
  use matched forecast rows and paired block bootstrap.
- TFT reuses an existing checkpoint, generates forecasts on a saved common
  evaluation-origin index, and is compared with every tabular model on exactly
  those rows.
- Portfolio sizes are limited to feasible `[4, 5, 10, 25]` combinations.
  Corrections with fewer than 100 distinct calibration origins are explicitly
  invalid and excluded from peak-control claims.
- Load shifting conserves energy, respects destination capacity, prevents
  cascaded shifts, and rejects any action that would increase the realised
  forecast-window peak.
- `suite` produces isolated operational and oracle-weather runs and a direct
  metric-difference table.

## Windows execution

From the repository root:

```powershell
py -3.11 -m pip install -r .\requirements.txt
py -3.11 -m pip install -r .\requirements-tft.txt
py -3.11 -m pip install -e .
py -3.11 -m heapo_forecasting.cli full --config .\configs\local.yaml
```

For the operational/oracle sensitivity analysis:

```powershell
py -3.11 -m heapo_forecasting.cli suite --config .\configs\local.yaml
```

Old result files are not deleted. The revised incompatible tables use `v2`
artifact names, and the existing TFT checkpoint is reused automatically.
