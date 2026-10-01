# HEAPO main-paper figure and table captions

## Tables

**Table 1.** Study population and household-level data partitions. The seen-test row reports independent households contributing to the final within-household evaluation; the confirmatory threshold was 20 households.

**Table 2.** Common-index model summary across the four operational forecast horizons. Metrics were weighted by the number of common forecast-origin observations. Lower MAE, approximate CRPS, interval width and weighted interval score (WIS) indicate better performance conditional on calibration; nominal prediction-interval coverage was 0.80.

**Table 3.** Confirmatory paired MAE comparisons from the primary household-cluster bootstrap and the nested household/local-day sensitivity analysis. Delta MAE was calculated as candidate minus comparator; negative values favour the candidate. Holm correction was applied within each pre-specified candidate/comparator family.

**Table 4.** Leakage-resistant portfolio peak-management outcomes under the pre-specified 20% flexibility and 3 h forward-shifting scenario. Forecast-driven source, destination and acceptance decisions were frozen from origin-available information; realised demand was used only for ex post scoring. The non-deployable perfect-foresight row applies the identical dispatch constraints to realised future load and separates forecast loss from dispatch limitations. Values were weighted by forecast origins across calibration-valid portfolio strata.

## Figures

**Figure 1.** HEAPO study population and leakage-resistant evaluation design. Model parameters were fitted using development-household targets ending by the training cutoff; household-disjoint validation origins and their complete 96-step targets also ended by that cutoff and were used only for LightGBM early-stopping/model-selection monitoring. The final unseen-test group was opened only for locked generalization evaluation. Primary inference resampled complete households; the sensitivity analysis additionally resampled local calendar days within each sampled household occurrence. Confirmatory inference required at least 20 independent households.

**Figure 2.** Multi-horizon MAE on the common TFT/tabular evaluation index. Lines show model-specific MAE at 15 min, 1 h, 6 h and 24 h; thin error bars show 95% household-cluster bootstrap confidence intervals. Sample sizes (independent households/common-index forecast origins per horizon) were: Seen households — ASHP: n = 29 households and 348 common-index forecast origins per horizon; Seen households — GSHP: n = 12 households and 144 common-index forecast origins per horizon; Unseen households — ASHP: n = 9 households and 108 common-index forecast origins per horizon; Unseen households — GSHP: n = 4 households and 48 common-index forecast origins per horizon.

**Figure 3.** Prediction-interval coverage on the common evaluation index. Cell values are PICP estimates and colours encode deviation from the nominal 0.80 coverage; blue and red represent over- and under-coverage, respectively. Household counts are shown in the panel identifiers. The heatmap cells and text are rendered as vector elements in the PDF.

**Figure 4.** Confirmatory paired MAE effects in the seen-ASHP stratum. Points show candidate-minus-comparator MAE differences and horizontal lines show 95% household-cluster bootstrap confidence intervals. Filled points remained confirmatory in the nested household/local-day sensitivity analysis; open points were significant only in the primary design.

**Figure 5.** Global LightGBM feature importance for ASHP and GSHP observations. Only the 12 highest-ranked features are shown. Importance was quantified as the mean absolute SHAP value and averaged over a seeded sample of held-out seen- and unseen-test origin–horizon observations within each heat-pump type (maximum 5,000 observations per type) for the LightGBM mean model; features were then ranked by their mean importance across heat-pump types.

**Figure 6.** Peak-reduction and false-intervention trade-off for calibration-valid virtual portfolios under 20% flexible load and a 3 h forward-shifting window. Median forecasts and conformal upper bounds were used as deployable decision signals; the star denotes the non-deployable perfect-foresight diagnostic under identical dispatch constraints. Values were aggregated across seen and unseen cohorts and valid portfolio sizes using forecast origins as weights. Horizontal and vertical error bars show 95% portfolio-ID cluster-bootstrap confidence intervals. The portfolio-calibration sensitivity analysis is reported in Supplementary Table S1.
