# Milestone 2 Plan: Phylogenetically-Weighted PCA for URIEL+ Typological Feature Reduction

All work uses **URIEL+ v1.3.2**.

## 1. Target / Task

URIEL+ represents each language with 828 binary typological features. We want to reduce this to a smaller subset that keeps the information that enables URIEL+ vectors to be used for language embeddings and URIEL+ distances to be used for signals of language similarity in tasks such as Machine Translation, Entity Linking, Dependency Parsing, and Part-of-Speech Tagging.

Linguistic work such as Dunn et al., 2005 and Jäger and Wahle, 2021 has shown that phylogenetically similar languages tend to be typologically similar. This has a side effect for Principal Component Analysis (PCA): Overly-represented families (Indo-European, Niger-Congo, and Austronesian) contribute many near-redundant rows, which dominate the covariance structure and therefore the loadings used for feature selection. Our hypothesis is that **down-weighting phylogenetically redundant languages produces a better feature subset than ordinary PCA**, evaluated by how well the resulting typological distances help LangRank (Lin et al., 2019) rank transfer languages for four NLP tasks: Machine Translation, Entity Linking, Dependency Parsing, and Part-of-Speech Tagging.

The baseline is York et al. (2025)'s PCA-loading feature selection on URIEL+. The baseline authors have confirmed that their original pipeline selected features on data still containing the `-1` missing-value sentinel, and that this was a mistake. Our baseline reproduction fixes this by imputing before selecting (Section 4), so our baseline will not precisely reproduce their originally published numbers; this is an expected, intentional difference, not an error in our reproduction.

This is an **unsupervised representation-reduction task** (PCA, imputation) evaluated via a **downstream supervised ranking task** (LangRank's `LGBMRanker`). Sections 5 and 6 describe how these two pieces fit together.

## 2. Language Set: Source-Integrated Languages Only, not Glottolog Dialects

URIEL+ v1.3.2 with all linguistic data sources integrated contains 26,881 languages, but 18,709 of these languages were added via `integrate_glottolog` with the aim of supplying dialect targets for the BFS genetic-imputation step (Section 4): a dialect's row exists mainly to receive a value copied from its parent language. Therefore, many of the dialects' cells are literal duplicates of the parent's row rather than independent typological evidence.

**Decision: we skip `integrate_glottolog`**, using `integrate_custom_databases` with every other source instead, our working matrix contains 8,172 languages (Section 6, Setup Code). Reasons:
- It avoids a form of redundancy that is not well captured by our phylogenetic weighting scheme, since dialects are mechanically copied rows rather than independently evolved relatives, and including them would not obviously advantage either PCA arm fairly.
- LangRank's 150 evaluation languages are standard languages, not Glottolog dialects, so the dialects contribute nothing directly to evaluation.
- It keeps our PCA step comparable in scale to York et al. (2025), who worked with the original 8,172-language URIEL+.

## 3. Scope: Typological Features Only

LangRank's ranker uses eight distance feature categories: `GENETIC`, `SYNTACTIC`, `FEATURAL`, `PHONOLOGICAL`, `INVENTORY`, `GEOGRAPHIC`, `MORPHOLOGICAL`, `SCRIPT`. Our PCA reduction applies only to the **typological** features (the `S_`, `P_`, `INV_`, `M_`-prefixed columns), which correspond to `SYNTACTIC`, `PHONOLOGICAL`, `INVENTORY`, `MORPHOLOGICAL`, and `FEATURAL` (`FEATURAL` distances include all features regardless of category). `GENETIC`, `GEOGRAPHIC`, and `SCRIPT` are computed identically across all three configurations (Section 6) and are not touched by our method.

## 4. Missing-Value Handling

URIEL+'s built-in `softimpute_imputation()` handles the full pipeline internally: it aggregates (using whatever aggregation mode is configured), runs BFS genetic imputation, converts the `-1` missing-value sentinel to `NaN`, and fills remaining missing values with SoftImpute. We do not need to call these steps separately (Section 6, Setup Code).

1. **Union aggregation**: an OR operation across sources (any source reporting 1 results in the value of the feature being 1, while all sources reporting 0 result in a value of 0).
2. **BFS genetic imputation** (`fill_with_base_lang = True`, the library default): propagates known values from parent languages down to dialect children still missing them, run internally as part of aggregation.
3. **SoftImpute**, run on the full matrix, on whatever remains missing after step 2. All URIEL+ imputation methods, including SoftImpute, output **binary** values (0 or 1), consistent with the rest of URIEL+'s typological data. There is no separate rounding/thresholding step needed on our end.

Imputation is a **controlled variable**: all three configurations (Section 6) use the identical imputed matrix, so any downstream difference is attributable to the PCA step, not to how missing values were filled.

**Missingness**, for the 8,172-language typological matrix, after union aggregation and BFS imputation, before SoftImpute: 87.34% missing. This reflects the larger, more comprehensive v1.3.2 language set relative to the original URIEL+, not a regression in data quality.

**Limitations:**
- A union value of 0 means "no source reported a 1," not "confirmed absent."
- At ~87.34% sparsity, imputed values partly reflect column-level priors and low-rank structure rather than language-specific evidence.
- Missingness is not random: poorly documented languages and families are missing far more data than well-documented ones.

## 5. Method: Phylogenetically-Weighted PCA

### 5.1 Phylogenetic Weights

Source: `lang_fam_geo.csv`, keyed by Glottocode (matching our matrix rows). Every one of the 8,172 source-integrated languages has a phylogenetic vector and an ordered family chain. Therefore, there are no languages to drop or assign default weights to for this reason. Each language's chain runs broad to narrow, for example:

| Language | Chain (abridged) | Finest-grained family |
|---|---|---|
| Swahili | Atlantic-Congo, ..., Sabaki-Swahili, Swahili (G.40), Mombasa-Lamu-Inland Swahili | `Mombasa-Lamu-Inland Swahili` |
| Zulu | Atlantic-Congo, ..., Nguni (S.40), Nuclear Nguni, Southern Ndebele-Lowland | `Southern Ndebele-Lowland` |
| Wolof | Atlantic-Congo, North-Central Atlantic, Wolof-BKK, Wolofic | `Wolofic` |

**Finest-grained family** = the last element of a language's chain (split the string on `", "`). The weight for a language is:

```
w_i = 1 / (number of languages sharing the same finest-grained label)
```

A language with many close relatives gets a low weight (it is redundant). A language with few or none gets a high weight.

**Remaining edge case:** isolates (no relatives at the finest level) get weight 1. Languages whose chain is short use their last available label as the finest-grained level; the granularity of "finest-grained" therefore varies with how deeply a language has been classified, which is a noted limitation rather than something we correct for.

Weights are normalized to sum to 1, and their distribution (min, median, max, and the most heavily down-weighted labels) is reported in the final write-up.

### 5.2 Computation

Let X be the imputed N_lang × 828 matrix (typological features only, Section 3) and w the weight vector.

1. Weighted mean: μ_w = Σ w_i x_i
2. Center: X̃ = X − μ_w (no standardization, matching the baseline)
3. Weighted covariance: C_w = X̃ᵀ diag(w) X̃
4. Eigendecompose C_w with `np.linalg.eigh` (828 × 828, negligible cost)
5. Keep components up to 95% explained variance (as in the baseline)
6. Feature score = max over components of |loading|, as in the baseline's `pca.ipynb`. Keep the top N original features.

`sklearn.PCA` does not support sample weights, so the eigendecomposition is done directly with NumPy/SciPy.

### 5.3 Choosing N

Rather than sweeping the full N = 100 to 700 range, we select one working value of N for the main comparison.

Before running any LangRank evaluation, we compute both the weighted and unweighted PCA feature rankings on the same imputed matrix and measure the **Jaccard overlap** of the selected feature sets across a range of N. This is cheap (minutes) and tells us where the two methods actually diverge:
- If overlap is very high (say ≥0.95) at a candidate N, the two arms select nearly the same features and any LangRank difference will likely be negligible; a smaller N should be tried.
- N = 400 (half of 828, inside York et al. (2025)'s original sweep range) is our starting candidate, changed based on the overlap results.

The chosen N and the overlap table are reported regardless of which N is ultimately used. This overlap analysis is our **model-selection / validation step** (Section 7). It is done without touching LangRank or NDCG at all, so N is not tuned against the test metric.

## 6. Experimental Configurations and Setup

| Config | Typological features | Pipeline | Purpose |
|---|---|---|---|
| A. Base URIEL+ | Full 828 | Imputed matrix, no reduction | Upper Reference |
| B. Regular PCA | Top N | Impute first, then unweighted PCA loading selection | Baseline |
| C. Phylo-weighted PCA | Top N | Impute first, then weighted PCA loading selection | Our Method |

B vs. C isolates the effect of phylogenetic weighting, since both use the same imputed data and the same selection rule. A shows how much information any reduction gives up. `GENETIC`, `GEOGRAPHIC`, and `SCRIPT` features (Section 3) are identical across all three configurations.

**Setup Code** (data integration and imputation, shared by all three configurations):

```python
# Data generation (already executed; outputs archived in the root zip).
from urielplus import urielplus

u = urielplus.URIELPlus()
u.reset()

# Configuration
u.set_cache(True)
u.set_aggregation('U')

# Integrate databases, excluding Glottolog to keep only the 8,172 source-integrated
# languages (Glottolog integration adds ~19,000 dialect rows used only
# as BFS targets for genetic imputation; see Section 2).
u.integrate_custom_databases("UPDATED_SAPHON", "BDPROTO", "GRAMBANK", "APICS", "EWAVE")

# Aggregates (union), runs BFS genetic imputation (fill_with_base_lang,
# default True), converts -1 to NaN, and fills remaining missing values
# with SoftImpute. Output is binary, matching URIEL+'s data convention.
u.softimpute_imputation()
```

```python
# Extraction (run on the archived output to load the imputed matrix,
# language list, feature list, and source metadata for downstream
# weighting + PCA).
npz_file = np.load('features.npz', allow_pickle=True)

print("Arrays in the NPZ file:", npz_file.files)

features = npz_file['feats']
languages = npz_file['langs']
sources = npz_file['sources']
data = npz_file['data']
```

**Controls Held Constant across Configurations:** the distance metric (angular distance, URIEL+'s default), LangRank version and tasks, the 150-language evaluation set, the SoftImpute output, the language set (8,172 languages, Section 2), and the non-typological feature categories (Section 3).

## 7. Train / Validation / Test Strategy

PCA and SoftImpute are **unsupervised**: there is no label to fit against, so "training" here means fitting the weighted/unweighted covariance eigendecomposition and the imputer on data, not learning from a labeled training split. Our actual train/validation/test structure is:

- **Fit ("Train")**: PCA (weighted and unweighted) and SoftImpute are fit on the full 8,172-language matrix, including the 150 languages later used for LangRank evaluation. This is a **transductive** setup: the PCA basis "sees" the evaluation languages during fitting, which is standard for representation-learning work on typological databases, but is noted as a limitation rather than presented as a clean separation.
- **Validation**: the Jaccard-overlap analysis (Section 5.3) selects N without any reference to LangRank or NDCG, so N is not tuned against the test metric.
- **Test**: LangRank's own evaluation protocol, run once per finalized configuration (A, B, C). LangRank trains an `LGBMRanker` on a feature vector that includes our (possibly reduced) typological distances alongside `GENETIC`, `GEOGRAPHIC`, `SCRIPT`, and task-specific features (entity overlap, dataset sizes), using **leave-one-target-language-out cross-validation** (`LeaveOneGroupOut` over `Target lang`): for each fold, all rows for one target language are held out, the ranker is trained on the rest, and NDCG@3 is computed on the held-out target language's ranking of candidate source languages. This is repeated for every target language and averaged. We inherit this protocol as-is for each of the four LangRank tasks (Machine Translation, Entity Linking, Dependency Parsing, POS Tagging) rather than redefining a split ourselves. LangRank's `LGBMRanker` is run with a fixed `random_state`, so results for a given configuration are deterministic and reported as a single run rather than an average over seeds.

## 8. Primary metric

**NDCG@3** (Normalized Discounted Cumulative Gain at rank 3), as computed by LangRank's existing evaluation code. It measures whether our distance-based features help the ranker surface the correct top-3 source languages for transfer learning, relative to the empirically best source language for a given target language and task. Reported per task and averaged across the four tasks, for each of configurations A, B, and C.

Secondary, non-downstream diagnostics (Useful for interpreting results):
- Jaccard overlap between weighted and unweighted PCA feature sets at the chosen N (Section 5.3).
- Feature-type composition of the selected sets (`S_` / `M_` / `INV_` / `P_`), as in the baseline's `selection_type_analysis.ipynb`.
- NDCG@3 broken out by within-family vs. cross-family transfer pairs, to check whether gains (if any) hold when phylogeny alone would not already identify the correct transfer language.

## 9. Feasibility and compute plan

**Feasibility**: all required tooling exists and has been verified. `softimpute_imputation()` runs in a couple of minutes even across URIEL+'s full 26,881-language set, so it is not a bottleneck even though we restrict to 8,172 languages. The weighted PCA eigendecomposition is a single 828×828 eigendecomposition, on the order of seconds. LangRank's evaluation code (Entity Linking shown above; Machine Translation, Dependency Parsing, and POS Tagging follow the same structure) is already adapted to the current URIEL+ version and Glottocode mapping.

**Compute Budget (9 Hours)**: one LangRank run per configuration takes at most about 3 hours, almost all of it spent computing distances, across the four tasks.

| Item | Time |
|---|---|
| Config A (base URIEL+, 828 typological features) | ≤ 3 h |
| Config B (regular PCA, single N) | ≤ 3 h |
| Config C (phylo-weighted PCA, single N) | ≤ 3 h |
| **Total** | **≤ 9 h** |

Preprocessing (database integration without Glottolog, `softimpute_imputation()`, weight computation, and the PCA eigendecompositions) takes minutes, not hours, and is excluded from this budget.

## 10. Risks

- At ~87.34% sparsity, imputed values may mostly reflect column priors, limiting how much any downstream method can improve on the full-feature reference (Config A).
- Phylogenetic weighting could over-emphasize poorly documented isolates or small clades.
- Weighted and unweighted PCA may select highly overlapping feature sets at the chosen N, yielding no measurable LangRank difference; the Jaccard overlap check (Section 5.3) is designed to catch this before spending compute.
- Phylogeny-informed selection could favor features that mostly encode "which family is this" at the expense of areally shared features, which could hurt cross-family transfer specifically; the within-/cross-family NDCG breakdown (Section 8) is designed to catch this.
- The transductive fit (Section 7) means PCA has indirect access to the evaluation languages' typological data during fitting; results should be interpreted with this in mind rather than as a fully held-out test.
- Our baseline reproduction (impute-then-select) will not exactly match York et al.'s originally published numbers, since their pipeline selected features on data still containing `-1` values, which the authors have confirmed was unintended.

## 11. References

Dunn, M., Terrill, A., Reesink, G., Foley, R. A., & Levinson, S. C. (2005). Structural phylogenetics and the reconstruction of ancient language history. Science, 309(5743), 2072–2075. https://doi.org/10.1126/science.1114615

Jäger, G., & Wahle, J. (2021). Phylogenetic typology. *Frontiers in Psychology*, 12, Article 682132. https://doi.org/10.3389/fpsyg.2021.682132

Lin, Y.-H., Chen, C.-Y., Lee, J., Li, Z., Zhang, Y., Xia, M., Rijhwani, S., He, J., Zhang, Z., Ma, X., Anastasopoulos, A., Littell, P., & Neubig, G. (2019). Choosing transfer languages for cross-lingual learning. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 3125–3135). Association for Computational Linguistics. https://doi.org/10.18653/v1/P19-1301

Ng, Y. H., Hoang, P. H., & Lee, E.-S. A. (2025). Less is more: The effectiveness of compact typological language representations. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 25805–25816). Association for Computational Linguistics.