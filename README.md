# HEAPO Uncertainty-Aware Forecasting

Reproducible Python pipeline for probabilistic, multi-horizon forecasting of
air-source (ASHP) and ground-source (GSHP) heat-pump electricity demand using
the [HEAPO dataset](https://github.com/tbrumue/heapo). The workflow covers data
audit, leakage-resistant feature construction, household and temporal
validation, calibrated prediction intervals, portfolio aggregation,
peak-management scenarios, statistical finalization, and manuscript outputs.

> This repository contains analysis software only. HEAPO data and generated
> results are deliberately excluded from version control.

## Study scope

- Forecast horizons: 15 min, 1 h, 6 h, and 24 h.
- Models: persistence, daily/weekly seasonal naive, Elastic Net, LightGBM, and
  Temporal Fusion Transformer (TFT).
- Uncertainty: q10/q50/q90 forecasts and hierarchical conformal calibration.
- Validation: held-out households, chronological testing, rolling origins, and
  a common TFT/tabular evaluation index.
- Inference: household-cluster bootstrap as the primary analysis and nested
  household/day bootstrap as a sensitivity analysis, with Holm correction.
- Operations: dependence-aware virtual portfolios and energy-conserving
  peak-shifting scenarios.

The methodological implementation map is available in
[`docs/METHODS_MAPPING.md`](docs/METHODS_MAPPING.md).

## Repository layout

```text
.
├── .github/workflows/       GitHub Actions continuous integration
├── configs/                 default and local configuration examples
├── docs/                    methods, data, outputs, and reproducibility notes
├── scripts/                 download, pipeline, finalization, and figure tools
├── src/heapo_forecasting/   installable Python package
├── tests/                   data-free unit and adapter integration tests
├── CITATION.cff             software citation metadata
├── LICENSE                  MIT license
├── pyproject.toml           build and dependency metadata
└── requirements*.txt        reproducible installation entry points
```

## Requirements

- Python 3.11 is recommended; Python 3.12 is supported.
- Core analysis: at least 16 GB RAM is recommended.
- Full TFT and portfolio analyses are substantially more demanding. CPU-only
  TFT training is supported but can take many hours.

## Installation

### Windows PowerShell

```powershell
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -e .
py -3.11 -m pip install -e ".[tft,manuscript]"
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tft,manuscript]"
```

Equivalent requirements files are provided for environments that do not use
optional dependency groups:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-tft.txt
python -m pip install -r requirements-manuscript.txt
python -m pip install -e .
```

## Data

Download and extract HEAPO separately. The configured root must contain:

```text
heapo_data/
├── meta_data/
├── reports/
├── smart_meter_data/
└── weather_data/
```

An optional downloader is included:

```bash
python scripts/download_heapo.py --output-dir data
```

See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) before using or
redistributing the dataset.

## Configuration

Create a local override; `configs/local.yaml` is ignored by Git.

```powershell
Copy-Item .\configs\local.windows.example.yaml .\configs\local.yaml
```

Edit only the local data root and, if desired, the run name:

```yaml
project:
  run_name: heapo_main

data:
  root: "C:/path/to/heapo_data"
  weather_scenario: operational
```

The local file is merged recursively over `configs/default.yaml`. The default
configuration is also bundled into the installed Python package, so commands
work from editable installs and built wheels.

## Run the analysis

Start with the audit and inspect the eligibility counts:

```powershell
py -3.11 -m heapo_forecasting.cli audit --config .\configs\local.yaml
```

The stages can then be run separately:

```powershell
py -3.11 -m heapo_forecasting.cli prepare --config .\configs\local.yaml
py -3.11 -m heapo_forecasting.cli pilot --config .\configs\local.yaml
py -3.11 -m heapo_forecasting.cli full --config .\configs\local.yaml
py -3.11 -m heapo_forecasting.cli finalize --config .\configs\local.yaml
```

Or run the complete cached/restartable workflow:

```powershell
.\scripts\run_all.ps1 -Config .\configs\local.yaml
```

Linux/macOS:

```bash
./scripts/run_all.sh configs/local.yaml
```

`full` includes TFT when `models.tft.enabled: true`. Completed compatible
intermediate artifacts and TFT checkpoints are reused. `finalize` performs the
pre-specified bootstrap analyses from cached predictions and does not retrain
models.

To compare operational weather with an oracle realised-weather upper bound:

```powershell
py -3.11 -m heapo_forecasting.cli suite --config .\configs\local.yaml
```

Oracle results are isolated in a separate run directory and must not be
reported as the primary operational result.

## Manuscript tables and figures

After `full` and `finalize` complete:

```powershell
py -3.11 .\scripts\make_manuscript_outputs.py `
  --run-dir .\outputs\heapo_main `
  --dpi 600 --tiff
```

This writes four main-paper tables and six publication figures beneath
`outputs/heapo_main/manuscript_outputs/`. Figure captions are saved separately
in `MANUSCRIPT_CAPTIONS.md`; no global caption is drawn above the figure files.

## Main outputs

```text
outputs/<run_name>/
├── audit/                 eligibility and variable availability
├── panel/                 processed household panels and manifest
├── splits/                household IDs and chronological cutoffs
├── models/                fitted models and calibration objects
├── predictions/           tidy q10/q50/q90 household forecasts
├── metrics/               point, interval, cold, peak, and test statistics
├── portfolios/            memberships and calibrated aggregate forecasts
├── peak_management/       load-shifting detail and summaries
├── tft/                   TFT data metadata, checkpoint, and predictions
├── manuscript_outputs/    four tables, six figures, captions, and manifest
├── run_manifest.json      environment, seed, config, and input fingerprints
├── validation_report.md   post-run integrity checks
└── final_statistical_report.md
```

The output contract is documented in
[`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md).

## Tests

The test suite creates a small synthetic HEAPO-like structure and requires no
downloaded data:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

GitHub Actions runs these checks on Python 3.11 and 3.12 and also builds the
wheel/source distribution.

## Reproducibility and resource controls

The pipeline records the configuration, seed, package versions, and input file
fingerprints. UTC is retained for model rows; Europe/Zurich time is derived
only for calendar variables. Operational future weather uses causal
information, long gaps are not interpolated, and held-out households never
contribute training targets.

The full dataset can create very large origin–horizon tables. The defaults use
bounded, stratified training samples and daily evaluation origins. Change
`sampling.max_train_rows`, origin strides, portfolio repetitions, TFT batch
size, or TFT row limits only as a documented sensitivity/resource decision.
See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation and license

If you use the dataset, cite the HEAPO data authors according to the original
repository and data record. If you use this software, use the metadata in
[`CITATION.cff`](CITATION.cff). The code is released under the
[`MIT License`](LICENSE); no HEAPO data are included.

## Contributing

Bug reports and focused pull requests are welcome. Do not commit HEAPO data,
model checkpoints, generated results, or personal paths. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).
