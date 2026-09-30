# Data Card: URIEL+ Language Vectors and LangRank Transfer Rankings

**Team 25 - URIEL Underdogs** · CSCI 3052U

---

## Overview

This project investigates whether incorporating genealogical relationships into PCA produces more informative low-dimensional representations of URIEL+ typological data than standard PCA. Three representations are compared: the original URIEL+ typological vectors, vectors reduced with standard PCA, and vectors reduced with phylogenetically-informed PCA. These are evaluated on two tasks: a typological-data imputation task and the downstream LangRank cross-lingual transfer task. Three datasets are used:

- **URIEL+** v1.3.1 (Khan et al., 2025): per-language typological, phylogenetic, geographic, and script feature vectors covering approximately 26,000 languages across 828 features.
- **Glottolog** (Hammarström et al., 2026): language family tree and coordinates; provides the phylogenetic backbone for URIEL+.
- **LangRank experiment tables** (Lin et al., 2019): one row per (target, transfer) language pair with measured NLP task performance across four tasks: dependency parsing (DEP), entity linking (EL), machine translation (MT), and POS tagging.

---

## Licenses

| Source | License |
|---|---|
| URIEL+ (code + database) | CC BY-SA 4.0: attribution required; derived vectors must share alike |
| Bundled Grambank, APiCS | CC BY 4.0 |
| Bundled eWAVE | CC BY 3.0 |
| Glottolog | CC BY 4.0 |
| LangRank code | BSD-3-Clause |
| URIELPlus-LangRank CSVs | No license on file: acceptable for coursework; do not redistribute publicly |

---

## Data Description

**URIEL+ typological matrix**: binary features ({0, 1}) merged from multiple linguistic databases. Missing values are represented as -1. Features are grouped into syntactic, morphological, inventory, phonological, phylogenetic, geographic, and script categories. Coverage is sparse, with the majority of typological cells missing, particularly for syntactic and script features.

**LangRank experiment tables**: each row is a (task language, transfer language) pair. Columns include word/entity overlap, dataset sizes, and URIEL distance scores (genetic, syntactic, featural, phonological, inventory, geographic, morphological, script). The target variable is `Accuracy` for DEP, EL, and POS, and `BLEU` for MT. These distance columns are what our experiments replace.

---

## Cleaning, Filtering & Preprocessing

**Upstream (inherited):**
- Language codes mapped to Glottocodes; ambiguous macrolanguage codes fixed by hand.
- MT pairs involving ambiguous macrolanguages removed.
- URIEL+ sources merged by taking the max per feature; missing values filled with SoftImpute.
- Relevance labels assigned per task language: top-10 transfer languages ranked 10 to 1, all others 0.

**Our pipeline:**
1. Replace the -1 missing values and standardise features before PCA.
2. Fit plain PCA and phylogenetically-informed PCA.
3. Recompute pairwise distances from reduced vectors and update the experiment CSVs.

**Known issues:**
- `dep_updated.csv` has stray columns and is missing morphological/script distances; it will be regenerated rather than reused.
- POS has rows with no `Accuracy` value; these rows receive a relevance score of 0.

---

## Validation Strategy

**Imputation task:** A portion of known feature values in the URIEL+ typological matrix are masked, then reconstructed using URIEL+'s built-in SoftImpute algorithm. Reconstruction quality is measured using accuracy, precision, recall, and F1 against the original values.

**LangRank downstream task:**
- **Model:** LightGBM `LGBMRanker` (lambdarank), identical configuration across all experiments. Only the distance columns change between runs.
- **Split:** Leave-one-language-out cross-validation grouped by task language.
- **Metric:** NDCG@3 per fold, averaged. Evaluated in two modes: all features and distance-only, to isolate the effect of our vectors.
- **Comparison:** Three configurations (URIEL+ baseline, URIEL+ + PCA, URIEL+ + phylo-PCA) compared with a Wilcoxon signed-rank test.
- **Leakage:** PCA and imputation are unsupervised and never see transfer performance scores, so no label leakage is introduced.

---

## Feasibility Risks

| Risk | Mitigation |
|---|---|
| **High sparsity:** most typological cells are filled in via imputation, so PCA may capture imputation patterns rather than real typology | Analyse how sparsity affects PCA results |
| **Few folds** for some tasks: low statistical power, significance may be undetectable | Report effect sizes; do not over-interpret small-task results |
| **Runtime:** approximately 9 hours total across three configurations | To be determined |
| **Reproducibility:** upstream scripts use Windows-style file paths, which break on macOS and Linux | Replace hardcoded paths using `pathlib.Path` or `os.path.join()`, which automatically use the correct file separator for the operating system the script is run on |

---

## References

- Hammarström, H., Forkel, R., Haspelmath, M., & Bank, S. (2026).
  *Glottolog 5.3* [Data set]. Max Planck Institute for Evolutionary
  Anthropology. https://glottolog.org
- Khan, A., Shipton, M., Anugraha, D., Duan, K., Hoang, P. H., Khiu, E.,
  Doğruöz, A. S., & Lee, E.-S. A. (2025). URIEL+: Enhancing linguistic
  inclusion and usability in a typological and multilingual knowledge
  base. *Proceedings of COLING 2025*, 6937–6952.
- Lin, Y.-H., Chen, C.-Y., et al. (2019). Choosing transfer languages for cross-lingual learning. *ACL 2019*, 3125-3135.
- Shipton, M., Ng, Y. H., Khan, A., Hoang, P. H., Lu, X., Doğruöz, A. S.,
  & Lee, E.-S. A. (2025). *Simple additions, substantial gains: Expanding
  scripts, languages, and lineage coverage in URIEL+*
  (arXiv:2510.27183). arXiv. https://arxiv.org/abs/2510.27183
