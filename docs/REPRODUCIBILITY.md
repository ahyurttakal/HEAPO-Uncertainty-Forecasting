# Reproducibility protocol

## Reference environment

- Python: 3.11 recommended; 3.12 supported
- Time basis: UTC model rows with Europe/Zurich calendar features
- Random seed: `2026` in the default configuration
- Main weather setting: `operational`
- Oracle weather: sensitivity analysis only

Install the core, TFT, and manuscript dependency groups from `pyproject.toml`
or the corresponding requirements files. Keep the configuration used for each
reported run outside Git while retaining a sanitized copy with the manuscript
archive.

## Execution order

1. `audit` checks variable availability and pre-specified household inclusion.
2. `prepare` constructs regular household panels, splits, and supervised
   origin–horizon shards.
3. `pilot` fits baselines, Elastic Net, and LightGBM, calibrates intervals, and
   writes primary household predictions.
4. `full` reuses compatible cached artifacts, adds transfer, rolling-origin,
   extreme-cold, portfolio, peak-management, and optional TFT analyses.
5. `finalize` runs cached-result bootstrap inference and portfolio calibration
   sensitivity checks.
6. `scripts/make_manuscript_outputs.py` generates four tables and six figures
   without retraining.

The `scripts/run_all.*` wrappers execute steps 4–6 because `full` invokes or
reuses all preceding stages.

## Leakage controls

- Held-out household target observations are never used for model fitting.
- Temporal calibration and test periods follow fixed chronological cutoffs.
- Feature windows do not cross long target gaps or intervention boundaries.
- Operational future weather is causal; realised future temperature is kept
  only as an evaluation field and excluded by the model feature selector.
- MASE scales use continuous pre-training household panels and respect segment
  boundaries.
- Conformal corrections are fitted on calibration data and applied separately
  from raw intervals.
- Portfolio intervals are recalibrated from coincident aggregate residuals;
  marginal household quantiles are not summed as aggregate quantiles.

## Statistical analysis

Primary confidence intervals and paired comparisons resample complete
households. Nested household/local-day resampling is a sensitivity analysis.
Holm adjustment is applied within pre-specified comparison families, and
confirmatory labels require the configured minimum number of independent
households.

## Computational determinism

The run manifest records the configuration, seed, platform, Python/package
versions, and input fingerprints. Exact numerical equality across operating
systems, BLAS libraries, CPU instruction sets, and GPU kernels is not
guaranteed. Report substantive tolerances and preserve the manifest alongside
all results.

## Resource controls

The row caps and origin strides are part of the computational design. Record
any changes to:

- `sampling.max_train_rows`
- `sampling.train_origin_stride`
- `sampling.eval_origin_stride`
- TFT row, origin, batch, and epoch limits
- portfolio sizes and repetitions
- bootstrap repetitions

When memory is limited, reduce these controls in that order rather than
changing household/test membership or statistical definitions.
