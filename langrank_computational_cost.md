# Computational Cost of LangRank

- Choosing Transfer Languages for Cross-Lingual Learning (LangRank paper): https://aclanthology.org/P19-1301/
- LangRank repo: https://github.com/neulab/langrank
- URIELPlus-LangRank repo (Lee Language Lab): https://github.com/LeeLanguageLab/URIELPlus-LangRank

## Summary

Running the full LangRank pipeline, which includes computing URIEL+ linguistic distances and then training and evaluating rankers for all four tasks (DEP, EL, MT, POS), took **10,474.94s (~2h 54m)** end to end. Distance computation alone accounts for **10,448.29s, or 99.7%** of the total runtime; the CSV-update step and the actual ranking experiments together add up to less than 27 seconds. This pipeline needs to run three times (base URIEL+, URIEL+ with PCA, and URIEL+ with phylogenetically-informed PCA), bringing the total estimated cost to **~9 hours**. We believe this is a manageable cost and recommend proceeding with LangRank.

## 1. Distance Calculation

This is the dominant cost in the pipeline.

| Step | Time (s) | Time | % of stage |
|---|---:|---:|---:|
| Initialize URIEL+ | 0.71 | 0.7s | 0.0% |
| Reset | 0.61 | 0.6s | 0.0% |
| Set cache | 0.00 | 0.0s | 0.0% |
| Integrate databases | 151.35 | 2m 31s | 1.4% |
| Softimpute imputation | 1,646.38 | 27m 26s | 15.8% |
| Load mappings | 0.01 | 0.0s | 0.0% |
| Calculate distances - DEP | 1,406.96 | 23m 27s | 13.5% |
| Calculate distances - EL | 699.82 | 11m 40s | 6.7% |
| Calculate distances - MT | 4,512.07 | 1h 15m 12s | 43.2% |
| Calculate distances - POS | 2,030.39 | 33m 50s | 19.4% |
| **Stage total** | **10,448.29** | **2h 54m 8s** | **100%** |

**Takeaways:**
- Distance calculation for the **MT** dataset is the single most expensive step (43.2% of this stage, ~1h 15m), consistent with MT having the largest number of language pairs of the four tasks.
- **Softimpute imputation** (27m 26s) and database integration (2m 31s) together account for ~15% of the stage but only need to run once per configuration, not once per dataset.
- The four `new_distance` loops (DEP, EL, MT, POS) together take **8,649.24s (~2h 24m)**, i.e. the vast amount of total pipeline time.

## 2. Replacing Distances in Experiment CSVs

Fast - negligible relative to the rest of the pipeline.

| Step | Time (s) |
|---|---:|
| Update dep experiment csv | 0.04 |
| Update el experiment csv | 0.02 |
| Update mt experiment csv | 0.10 |
| Update pos experiment csv | 0.04 |
| **Stage total** | **0.21** |

## 3. LangRank Experiments

Ranker training/evaluation (leave-one-language-out cross-validation) across all four tasks, in both feature-set modes (`all` and `lang`).

| Experiment | Time (s) |
|---|---:|
| DEP - All features | 3.36 |
| DEP - Lang features | 2.66 |
| EL - All features | 1.27 |
| EL - Lang features | 0.72 |
| MT - All features | 7.00 |
| MT - Lang features | 5.79 |
| POS - All features | 3.14 |
| POS - Lang features | 2.50 |
| **Stage total** | **26.44** |

## 4. Total Cost

| Stage | Time (s) | % of total |
|---|---:|---:|
| Distance calculation | 10,448.29 | 99.75% |
| Replacing distances | 0.21 | 0.00% |
| LangRank experiments (all 8 runs) | 26.44 | 0.25% |
| **Total** | **10,474.94 (~2h 54m 35s)** | **100%** |

## 5. Recommendation

Distance computation, not model training, is what makes LangRank expensive, and it scales with the number of URIEL+ configurations tested rather than with the ranking experiments themselves. Since distances must be recomputed for each of the three planned configurations (base URIEL+, URIEL+ with PCA, and URIEL+ with phylogenetically-informed PCA), the full cost comes to roughly **9 hours**.

**We believe this cost is manageable and recommend proceeding with LangRank.**
