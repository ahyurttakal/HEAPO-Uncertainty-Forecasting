# Results schema

All generated artifacts are written below `outputs/<run_name>/`. Parquet is
used when PyArrow is available; selected summaries are also exported as CSV,
JSON, Markdown, PNG, PDF, TIFF, or XLSX.

| Location | Purpose | Representative artifacts |
| --- | --- | --- |
| `audit/` | Inclusion and data-quality checks | household audit and exclusion counts |
| `panel/` | Regular, gap-aware household panels | panel files and panel manifest |
| `splits/` | Household and chronological partitions | `split_manifest.json` |
| `supervised/` | Origin–horizon model rows | sharded tables and v2 manifest |
| `models/` | Fitted tabular models and training sample | LightGBM/joblib bundle, samples |
| `predictions/` | Tidy household forecasts | true load, q10/q50/q90, model, split, horizon |
| `metrics/` | Model evaluation and inference | MAE/MASE/CRPS/PICP, bootstrap and paired tests |
| `figures/` | Pipeline diagnostics | SHAP and calibration figures |
| `portfolios/` | Virtual portfolios and aggregate uncertainty | memberships, corrections, forecasts, summaries |
| `peak_management/` | Constrained load-shifting analysis | detail and scenario summary |
| `tft/` | TFT training/evaluation | common index, checkpoint, predictions, metrics |
| `manuscript_outputs/` | Journal-facing deliverables | four tables, six figures, captions, checksums |

## Validation gates

`validation_report.md` and its JSON companion summarize finite MASE values,
prediction-interval calibration, calibrated baselines, paired bootstrap
availability, portfolio support, peak-shifting constraints, and TFT/common
index consistency. A `PASS` result is required before manuscript export. A
`REVIEW` result must be resolved or explicitly justified before reporting.

## Interpretation

Prediction and metric rows remain household-level or origin-level until the
documented aggregation step. Portfolio cells marked `calibration_valid=false`
are unsupported and must not be used for peak-management claims. Oracle
weather outputs are sensitivity results and should remain isolated from the
operational run.
