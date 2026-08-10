# Study-to-code mapping

| Study component | Implementation | Main outputs |
| --- | --- | --- |
| Separate-meter ASHP/GSHP audit | `audit.py`, `io.py` | `household_audit`, exclusion counts |
| UTC/local-time and gap handling | `io.py`, `features.py` | processed household panels |
| Lag/calendar/weather/static features | `features.py` | supervised origin–horizon shards |
| Household and temporal validation | `splits.py`, `experiment.py` | split manifest, rolling metrics |
| Persistence and seasonal naive | `models.py` | tidy baseline forecasts |
| Elastic Net and LightGBM | `models.py` | fitted bundles and forecasts |
| q10/q50/q90 + conformal intervals | `conformal.py` | raw and calibrated intervals for every model, hierarchical corrections |
| TFT | `tft_model.py` | reusable checkpoint, exact common-origin quantile forecasts |
| Point/probabilistic/peak metrics | `metrics.py` | forecast, cold, and peak tables |
| ASHP/GSHP transfer | `experiment.py` | transfer metrics |
| Extreme-cold test | `experiment.py` | extreme-cold metrics |
| SHAP by heat-pump type | `explain.py` | importance tables and figures |
| Synthetic portfolios | `portfolio.py` | memberships and aggregate forecasts |
| Portfolio interval calibration | `portfolio.py` | residual corrections and q10/q90 |
| Peak decisions and load shifting | `portfolio.py` | scenario detail and summary |
| Block-bootstrap uncertainty and paired tests | `statistics.py` | metric confidence intervals and paired MAE deltas |
| Operational/oracle comparison | `experiment.py` (`suite`) | scenario metrics and deltas |
| Post-run integrity checks | `reporting.py` | JSON and Markdown validation reports |

## Leakage controls encoded in the implementation

1. Raw time is kept in UTC; Zurich time is used only to derive calendar fields.
2. Windows never cross a long target-data gap or an energy-consultant visit.
3. Held-out household IDs contribute no target rows to model fitting. Held-out
   validation origins are restricted to the calibration period and held-out
   test origins to the final chronological test period.
4. Operational future weather uses the previous day's same-quarter observation;
   fallback persistence uses only the latest exact hourly observation at origin.
5. Realised future temperature is retained only as
   `evaluation_temperature_c`; the model feature selector explicitly excludes it.
6. Conformal corrections are estimated on a chronological calibration period
   and use a model/type/horizon/cold hierarchy with documented small-cell
   fallbacks. The same procedure is applied to learned models and baselines.
7. Aggregate uncertainty is recalibrated from coincident aggregate residuals;
   household marginal quantiles are never presented as portfolio quantiles.
8. MASE denominators come from full pre-training panels, respect segment and
   timestamp continuity, and never use evaluation targets.
9. Model confidence intervals and comparisons resample household-day blocks;
   rows from the same daily trajectory are not treated as independent.
10. TFT and tabular comparisons use the exact same household/origin/horizon
    index. A TFT checkpoint without predictions is resumed at evaluation.

## Interpretation boundary

The load-shifting module is a grid-oriented scenario analysis. HEAPO does not
contain indoor temperature, so the code does not claim thermal-comfort-optimal
control. Energy is conserved within each forecast window and results are
reported as peak/load-factor/intervention outcomes only. A proposed shift is
rejected if destination capacity is exceeded or the realised window peak would
increase; rejected actions and energy-balance error are reported.
