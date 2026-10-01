# HEAPO run validation

Overall status: PASS

- [x] leakage_boundary_audit: Household, temporal, horizon, and clock-stratum boundaries passed.
- [x] conformal_temperature_provenance: Operational conformal regimes use origin-available weather; realised target-time temperature is evaluation-only.
- [x] finite_mase: 160/160 metric cells have a finite MASE value.
- [x] baseline_intervals_calibrated: Seasonal and persistence baselines have non-degenerate calibrated intervals.
- [x] paired_block_bootstrap: Household-day paired bootstrap comparison table is present.
- [x] figure3_calibration_index_consistency: Common-index macro PICP=0.829286; the actual Figure 3 cell source and manuscript audit agree exactly.
- [x] interval_coverage: Common-index macro mean absolute PICP deviation from 0.80 is 0.0485; interval width and WIS are audited.
- [x] portfolio_calibration_flagged: 8784000/12282240 aggregate rows pass the minimum calibration sample rule.
- [x] portfolio_operational_sensitivity: Peak thresholds, flexibility fractions and shift windows are exported in peak_management_sensitivity.
- [x] peak_shift_constraints: Negative reductions are retained; minimum peak reduction=-4.547. Maximum energy error=2.27374e-13.
- [x] perfect_foresight_benchmark: Oracle dispatch is present, energy-conserving, and has no forecast classification errors.
- [x] tft_common_index: TFT and tabular models are evaluated on the same origin index.
- [x] tft_early_stopping_independence: TFT early-stopping targets follow fitting targets and remain inside the training segment.
- [x] portfolio_origin_only_decisions: Portfolio decisions use forecasts only; realised demand is restricted to ex-post execution/scoring.
