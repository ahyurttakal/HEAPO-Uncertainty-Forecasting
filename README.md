# HEAPO Uncertainty-Aware Forecasting

[![CI](https://github.com/ahyurttakal/HEAPO-Uncertainty-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/ahyurttakal/HEAPO-Uncertainty-Forecasting/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible Python pipeline for leakage-resistant probabilistic forecasting of
air-source (ASHP) and ground-source (GSHP) heat-pump electricity demand using
the [HEAPO dataset](https://github.com/tbrumue/heapo).

The workflow covers data-quality auditing, causal feature construction,
household- and time-disjoint evaluation, conformal prediction intervals,
portfolio aggregation, peak-management diagnostics, statistical inference,
and publication figures. Curated aggregate results are included under
[`results/`](results/); raw HEAPO data, household-level predictions, fitted
models, and checkpoints are not distributed.

## Study scope

- Forecast horizons: 15 min, 1 h, 6 h, and 24 h.
- Models: persistence, daily and weekly seasonal baselines, Elastic Net,
  LightGBM, and Temporal Fusion Transformer (TFT).
- Uncertainty: q10/q50/q90 forecasts with hierarchical conformal calibration.
- Validation: chronological splits, unseen households, rolling origins, and a
  common TFT/tabular evaluation index.
- Inference: household-cluster bootstrap as the primary analysis and nested
  household/day bootstrap as a sensitivity analysis.
- Operations: dependence-aware virtual portfolios, energy-conserving load
  shifting, sensitivity analysis, and a perfect-foresight diagnostic.

See the [analysis protocol](docs/ANALYSIS_PROTOCOL.md),
[methods map](docs/METHODS_MAPPING.md), and
[probabilistic metrics](docs/PROBABILISTIC_METRICS.md).

## Repository layout

```text
.
├── .github/workflows/       continuous integration
├── configs/                 default and local configuration examples
├── docs/                    methods, data, outputs, and reproducibility notes
├── results/                 curated aggregate tables, figures, and audits
├── scripts/                 pipeline and manuscript-output entry points
├── src/heapo_forecasting/   installable Python package
├── tests/                   data-free unit and integration tests
├── CITATION.cff             software citation metadata
├── LICENSE                  MIT license
└── pyproject.toml           build and dependency metadata
```

## Requirements and installation

Python 3.11 is recommended; Python 3.12 is supported. At least 16 GB RAM is
recommended for the core analysis. TFT training and portfolio experiments are
more demanding.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[tft,manuscript]"
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tft,manuscript]"
```

## Data and configuration

Download and extract HEAPO separately. The configured data root must contain:

```text
heapo_data/
├── meta_data/
├── reports/
├── smart_meter_data/
└── weather_data/
```

An optional downloader is available:

```bash
python scripts/download_heapo.py --output-dir data
```

Create a private local configuration. It is ignored by Git:

```powershell
Copy-Item .\configs\local.windows.example.yaml .\configs\local.yaml
```

Set the actual dataset location in `configs/local.yaml`:

```yaml
project:
  run_name: heapo_analysis

data:
  root: "C:/path/to/heapo_data"
  weather_scenario: operational
```

See [data availability](docs/DATA_AVAILABILITY.md) before redistributing any
source data.

## Run the analysis

Audit the data first:

```powershell
py -3.11 -m heapo_forecasting.cli audit --config .\configs\local.yaml
```

Run the complete workflow on Windows without changing the system-wide
PowerShell execution policy:

```powershell
.\scripts\run_all.cmd .\configs\local.yaml
```

The equivalent Linux/macOS command is:

```bash
./scripts/run_all.sh configs/local.yaml
```

Individual stages are also available:

```powershell
py -3.11 -m heapo_forecasting.cli prepare --config .\configs\local.yaml
py -3.11 -m heapo_forecasting.cli pilot --config .\configs\local.yaml
py -3.11 -m heapo_forecasting.cli full --config .\configs\local.yaml
py -3.11 -m heapo_forecasting.cli finalize --config .\configs\local.yaml
```

The default run directory is `outputs/heapo_analysis/`. Existing artifacts
must match the current schema, configuration signature, and input
fingerprints; incompatible cached outputs are rejected.

## Publication outputs

After `full` and `finalize` complete:

```powershell
py -3.11 .\scripts\make_manuscript_outputs.py `
  --run-dir .\outputs\heapo_analysis `
  --dpi 600 --tiff
```

The command creates four main-paper tables and six publication figures in
`outputs/heapo_analysis/manuscript_outputs/`. Figure captions are saved in a
separate Markdown file; no overall title or caption is embedded in the figure.

The generated output contract is documented in
[results schema](docs/RESULTS_SCHEMA.md). The aggregate reference outputs
bundled with this repository are described in [results/README.md](results/README.md).

## Leakage safeguards

- Training, model-selection validation, conformal calibration, seen testing,
  and unseen-household testing have explicit and disjoint roles.
- LightGBM validation origins and their complete target trajectories remain
  inside the development period.
- TFT early stopping uses only a pre-specified final portion of its training
  segment; conformal calibration and test targets are excluded.
- Every forecast origin and scored target must be genuinely observed.
- All 96 target steps must remain within the partition assigned to an origin.
- Predictive preprocessing is fitted on training rows only.
- Operational future-weather features use only origin-available information;
  realised target-time weather is restricted to labelled oracle analyses.
- Portfolio decisions are frozen from forecasts. Realised load is used only
  for execution availability and ex-post scoring.
- Machine-readable audits must pass before manuscript outputs are created.

## Tests

The tests construct synthetic HEAPO-like data and do not require the original
dataset:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python -m build
```

GitHub Actions runs the checks on Python 3.11 and 3.12.

## Citation, license, and contributing

If you use the data, cite the HEAPO authors according to the original data
record. If you use this software, cite the metadata in [CITATION.cff](CITATION.cff).
The software is released under the [MIT License](LICENSE).

Bug reports and focused pull requests are welcome. Do not commit HEAPO data,
household-level outputs, model checkpoints, secrets, or personal filesystem
paths. See [CONTRIBUTING.md](CONTRIBUTING.md).
