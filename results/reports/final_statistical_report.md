# HEAPO final statistical validation

Status: COMPLETE

## Resampling design

- Primary inference resamples complete households with replacement.
- Sensitivity inference resamples households and then local calendar days within each sampled household occurrence.
- MAE and PICP remain row-weighted within each bootstrap replicate.
- Paired tests use identical household/origin/horizon observations for both models.
- Holm correction controls family-wise error separately for every pre-specified candidate/comparator family.
- Holm-significant results are labelled confirmatory only when at least 20 independent households are present.

## Independent household counts

| Split | HP type | Households |
|---|---:|---:|
| seen_test | ASHP | 29 |
| seen_test | GSHP | 12 |
| unseen_test | ASHP | 9 |
| unseen_test | GSHP | 4 |

## Paired-test summary

- household_cluster_paired_tests: 24/64 Holm-significant; candidate better=22, candidate worse=2; confirmatory with sufficient households=7.
- nested_household_day_paired_tests: 15/64 Holm-significant; candidate better=15, candidate worse=0; confirmatory with sufficient households=6.
- common_index_household_cluster_tft_paired_tests: 18/64 Holm-significant; candidate better=3, candidate worse=15; confirmatory with sufficient households=6.
- common_index_nested_household_day_tft_paired_tests: 6/64 Holm-significant; candidate better=1, candidate worse=5; confirmatory with sufficient households=4.

## Portfolio calibration sensitivity

Configured minimum independent origins: 100.

Excluded portfolio cells at the configured threshold:
- ASHP, size 5: calibration_n=58 (independent origins below 100).
- ASHP, size 25: calibration_n=12 (independent origins below 100).
- GSHP, size 10: calibration_n=42 (independent origins below 100).
- mixed, size 25: calibration_n=12 (independent origins below 100).

## Interpretation guardrail

Household-level results are the primary inferential evidence. The nested household/day results are a sensitivity analysis and do not increase the number of independent households.
