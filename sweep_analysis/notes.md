# Notes

- Both PCA-based selections (Config B and Config C) beat the full features (Config A)
    - All lines sit above the dashed (A) line (54.7 all features, 47.3 language features only), often by one to four points. This is in agreement with Ng et al. (2025).
- Weighted PCA (C) is not distinguishable from standard PCA (B) at the resolution of these experiments => Null result, matches the predictions of Dillon's Jaccard analysis (weights that change few features).
    - Weighting versus uniform PCA shows no consistent effect. Averaging the heatmap values across the seven `N` values we have Table 1. The signs flip with N, with the input mode and with the variant
    - The noise floor seems large. At `N = 300`, `w_finest` differs from `uniform` by only 6 of 300 features, however the all features score changes by −2.9 points.
    - `w_finest` looks flatter (more stable) across `N` than `uniform` (all features range about 1.3 points vs about 4.3). That being said, this comes from seven points, so we should treat it as a hypothesis, not a finding.

## Table 1
| Variant              | Mean change, all features | Mean change, language features only |
| -------------------- | ------------------------- | ----------------------------------- |
| w_finest             | +0.4                      | 0.0                                 |
| w_finest_adminmedian | −0.3                      | -0.7                                |
| w_finest_adminzero   | +0.7                      | -0.3                                |
| w_level3_a05         | +0.4                      | -0.5                                |
| w_level3_a1          | +0.7                      | -0.6                                |
