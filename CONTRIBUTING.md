# Contributing

Thank you for improving the HEAPO forecasting pipeline. Contributions should
be small enough to review, methodologically explicit, and reproducible.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

Install `.[tft]` only when changing TFT code and `.[manuscript]` when changing
publication-output code.

## Pull requests

1. Open an issue for changes that alter study design, data inclusion, splits,
   estimands, uncertainty calibration, or peak-management constraints.
2. Add or update a synthetic-data test for behavioral changes.
3. Run the unit tests and source compilation locally.
4. Describe the scientific and computational impact in the pull request.
5. Keep generated data, outputs, logs, checkpoints, and local paths out of the
   commit.

## Data and privacy

Do not upload HEAPO data to this repository. Do not include household-level
records, local absolute paths, credentials, or generated prediction tables in
issues or pull requests. Use minimal synthetic fixtures when reporting bugs.

## Reporting results

Do not mix the operational-weather analysis with the oracle realised-weather
upper bound. Changes to statistical thresholds, forecast origins, portfolio
sampling, or resource-control parameters must be documented in the run config
and manuscript methods.
