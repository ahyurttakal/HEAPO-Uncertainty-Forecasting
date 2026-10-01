# Curated reference results

This directory contains the aggregate, publication-facing outputs from the
reference analysis. It is included so readers can inspect the reported
findings without downloading the restricted raw data or reproducing the full
compute-intensive workflow.

## Contents

- `figures/`: six main-paper figures in PNG and vector PDF formats.
- `tables/`: four rounded main-paper tables in CSV format.
- `audits/`: aggregate cohort exclusions, calibration-index consistency, and
  portfolio sensitivity summaries.
- `reports/`: validation status, statistical finalization summary, and figure
  captions.

These files contain no raw meter readings, household identifiers,
household-level predictions, fitted models, or checkpoints. Values should be
interpreted together with the analysis protocol and the limitations of the
small unseen GSHP cohort.

To regenerate the directory, run the full pipeline and then execute
`scripts/make_manuscript_outputs.py`. Generated outputs remain under
`outputs/<run_name>/` and are ignored by Git unless deliberately curated.
