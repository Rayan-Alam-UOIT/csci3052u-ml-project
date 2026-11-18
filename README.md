# Phylogenetically-Informed Dimensionality Reduction of Linguistic Typological Data Using URIEL+

**Team Name:** URIEL Underdogs

**Team Number:** 25

## Team Members

| Name | Student ID | Position |
|---|---|---|
| Mason Shipton | 100910175 | Evaluation and Reproducibility Lead — Leads experimental design, evaluation, comparison, reproducibility, experiment tracking, and analysis of results. |
| Rayan Alam | 100908385 | Project Coordinator — Coordinates meetings, tracks overall progress, monitors milestone deadlines, and helps ensure that tasks are distributed and completed. |
| Adam Marcelo | 100591981 | Documentation and Presentation Lead — Coordinates documentation, report development, presentation preparation, and communication of the project's methods and findings. |
| Mohammed Abdulaziz | 100917802 | ML/Modeling Lead — Leads model selection, implementation, training, tuning, and explanation of the machine learning methods used. |
| Dillon Toshack | 100862159 | Data Manager — Leads dataset investigation, acquisition, cleaning, preprocessing, documentation, and data-related reproducibility. |

> These positions are temporary and are subject to change as time goes on.

## Team Resources

- **Meeting Room:** [Google Meet](https://meet.google.com/mjj-bbbh-pbo)
- **Project Board:** [Trello](https://trello.com/invite/b/6aac05071d61d26ec3de7cde/ATTI174667aaa6adcfb5b86a5f5f4ae97fd055D036C5/uriel-genetically-informed-pca)
- **Proposal:** [Proposal](https://docs.google.com/document/d/1CeyO7gn-_KbG_dhj3lbenXAqHYt0BHl8oGZj34D51XU/edit?tab=t.0)
- **Teamwork Contract:** [Teamwork Contract](https://docs.google.com/document/d/1SgG7BBqn87fXZtHvuYU7eVzqQYprCwfk/edit?tab=t.0)

---

## Overview

This project investigates whether phylogenetically informed dimensionality reduction can improve cross-lingual transfer language ranking. Using URIEL+ typological data, we compare the original feature representation, standard PCA, and phylogenetically weighted PCA across four LangRank tasks.

The experiments evaluate how feature count and language-family weighting influence ranking performance, with an emphasis on reproducibility, sensitivity analysis, and computational efficiency.


## Setup

Requirements: Python 3.11 (the pinned versions were verified on Python 3.11.3) and about 3 hours of uninterrupted compute for each of Steps 3 and 4. Other Python versions are untested and may not reproduce the expected results.

```
conda create -n urielplus python=3.11
conda activate urielplus
pip install -r requirements.txt
```

The pipeline was tested on Windows and macOS; however, it uses universal file paths and should work on Linux operating systems as well.

Note for macOS users: LightGBM requires the OpenMP runtime library (libomp). Ensure the latest compatible version is installed using your preferred package manager before running the project.

Open each notebook with the `urielplus` environment as its kernel. The LangRank notebooks check that LightGBM works (a small ranking fit) and install a compatible LightGBM and scikit-learn pair if it crashes.

The weighted PCA experiments additionally use the precomputed feature-selection archive at `jaccard_analysis/weighted_pca_sweep_output.zip`. Ensure this archive and the prepared `data.zip` are available before running the weighted PCA sweep.

## Running the Pipeline

The pipeline consists of five steps: preparing the URIEL+ dataset, applying standard PCA, evaluating the original and reduced representations, and exploring phylogenetically weighted PCA.

Run the steps in order. Run every cell in each notebook using **Run All**, including the setup, validation and packaging cells. Failed validation checks raise an error. ZIP extraction, data preparation and packaging are handled automatically by the scripts and notebooks.

| Step | Run | Produces | Time |
|---|---|---|---|
| 1 | `milestone_2_data.ipynb` | `data.zip` | minutes |
| 2 | `config_b_standard_pca.py` | `standard_pca.zip` | seconds |
| 3 | `URIELPlus-LangRank/langrank_uriel_plus_config_a.ipynb` | `base_urielplus_reproduction.zip` | about 2.5 to 3 h |
| 4 | `URIELPlus-LangRank/langrank_uriel_plus_config_b.ipynb` | `standard_pca_reproduction.zip` | about 2.5 to 3 h |
| 5 | `URIELPlus-LangRank/langrank_uriel_plus_config_c.py` | `weighted_pca_sweep_results.zip` | about 2.5 to 3 h |

Runtime estimates depend on hardware and system load.

### Step 1: Build the Imputed Matrix

Run `milestone_2_data.ipynb` with the repository root as the working directory. It builds the URIEL+ configuration below and automatically packages the resulting arrays into `data.zip` at the repository root.

- aggregation mode `U` (union)
- source databases UPDATED_SAPHON, BDPROTO, GRAMBANK and APiCS
- Glottolog integration skipped and eWAVE excluded
- BFS genetic imputation, then SoftImpute

The result is an 8,128-language by 593-feature binary typological matrix (`features.npz`), plus the family, script and geocoordinate arrays, all packaged inside `data.zip`. The pre-imputation version is stored under `non_imputed_data/` inside the ZIP.

### Step 2: Reduce the Features (Config B)

From the repository root, run:

```
python config_b_standard_pca.py
```

The script selects 300 features, as specified by `N = 300` inside the script. It automatically extracts the required data from `data.zip`, fits a centred, unstandardized PCA at 95% explained variance, scores each feature by its maximum absolute loading over the retained components, and keeps the top N features in their original column order. The Config B notebook's reference check corresponds to this 300-feature configuration.

It writes three files and packages them into `standard_pca.zip`:

- `features.npz`, the reduced matrix
- `standard_PCA_scores.csv`, the score for every one of the 593 features
- `standard_PCA_300_meta.json`, including the number of components kept

### Steps 3 and 4: Run LangRank (Configs A and B)

Run Config A to completion, including the final packaging cell, before starting Config B. Open each notebook inside the `URIELPlus-LangRank` folder (or set the `LANGRANK_DIR` environment variable) and use **Run All**. Each notebook:

1. Checks the environment,
2. Places the URIEL+ `.npz` files into `distances/urielplus/database/` (Config A takes all four from `data.zip`; Config B takes `features.npz` from `standard_pca.zip` and the other three from `data.zip`),
3. Checks the placed data by recomputing the Czech–Russian distance row,
4. Recomputes the distances for the four tasks (the slow part, over 5,754 language pairs), unless complete distance CSVs already exist,
5. Writes the distances into the experiment CSVs,
6. Runs the eight experiments (DEP, EL, MT, POS, each in `all` and `lang` mode),
7. Shows a table of average NDCG@3 and packages everything generated into its reproduction zip.

Keep the machine awake and the kernel running. If the distance step is interrupted, rerun the notebook and that Step restarts from scratch.


### Step 5: Evaluate Phylogenetically Weighted PCA (Config C)

Config C extends standard PCA by incorporating phylogenetic weights during feature selection. Languages belonging to densely represented families receive less influence, while languages from smaller groups receive relatively more influence.

The experiment evaluates seven feature counts (N = 100, 150, 200, 250, 300, 350, 400) across six weighting variants: `uniform`, `w_finest`, `w_finest_adminmedian`, `w_finest_adminzero`, `w_level3_a05`, and `w_level3_a1`.

This produces **42 configurations**, each evaluated on four LangRank tasks (DEP, EL, MT, POS) using both all-features and language-only modes.

The experiment runner is located at `URIELPlus-LangRank/langrank_uriel_plus_config_c.py`. It uses the precomputed feature-selection archive at `jaccard_analysis/weighted_pca_sweep_output.zip` and reuses fixed distance calculations while caching repeated language-pair computations.

From the repository root, run:

```bash
python URIELPlus-LangRank/langrank_uriel_plus_config_c.py
```

Results are packaged into `weighted_pca_sweep_results.zip`, with each configuration identified by its directory path, such as `N_300/uniform` or `N_150/w_finest_adminzero`.


## Outputs

The pipeline produces three primary experiment archives:

- `base_urielplus_reproduction.zip` — Config A, using the full URIEL+ feature set.
- `standard_pca_reproduction.zip` — Config B, using 300 features selected through standard PCA.
- `weighted_pca_sweep_results.zip` — Config C, containing all 42 standard and weighted PCA configurations.

Each reproduction archive contains the generated distance CSVs, updated experiment CSVs, timing logs, `scores.txt` (individual held-out-group and average NDCG@3 scores), and `results_table.csv` (average NDCG@3 by task and feature mode). The Config A and B archives also include the four URIEL+ database arrays.

The Config C archive organizes results by feature count and weighting variant using the format `N_<count>/<weight_variant>`. These paths serve as unique experiment identifiers and allow individual runs to be traced back to their corresponding results.

All NDCG@3 scores in the results tables are reported on a 0–100 scale. The saved fold-level results support evaluation across held-out language groups, although experiments were not repeated across independent random seeds.

## Expected Results

After completing Steps 3 and 4, the `results_table.csv` files in `base_urielplus_reproduction.zip` (Config A) and `standard_pca_reproduction.zip` (Config B) should report the following average NDCG@3 scores.

All scores are reported on a **0–100 scale**. Small differences may occur due to hardware, software versions, or execution environment.

### Config A: Base URIEL+

| Method | DEP | EL | MT | POS | Avg |
|---|---:|---:|---:|---:|---:|
| LangRank (all) | 76.0 | 65.5 | 53.6 | 23.6 | 54.7 |
| LangRank (lang feats) | 72.8 | 62.8 | 33.5 | 20.2 | 47.3 |

### Config B: Standard PCA (300 Features)

| Method | DEP | EL | MT | POS | Avg |
|---|---:|---:|---:|---:|---:|
| LangRank (all) | 76.8 | 75.1 | 54.0 | 28.9 | 58.7 |
| LangRank (lang feats) | 73.0 | 74.0 | 34.5 | 23.2 | 51.2 |


### Config C: Phylogenetically Weighted PCA

The 42-configuration sweep compares standard and phylogenetically weighted PCA across different feature counts and weighting strategies.

The strongest overall configuration is uniform PCA with 300 selected features. The best weighted configuration using all features is `N_150/w_finest_adminzero`, which achieves a similar but slightly lower average score.

| Configuration | N | All Features | Language Features |
|---|---:|---:|---:|
| Config A (full features) | Full | 54.7 | 47.3 |
| Uniform PCA | 300 | **58.7** | **51.2** |
| Weighted PCA (`w_finest_adminzero`) | 150 | 58.3 | — |

Although phylogenetic weighting changes the selected feature sets, no tested weighting strategy consistently outperforms uniform PCA across feature counts, tasks, and evaluation modes.


### Comparing Configurations

Standard PCA improves the four-task mean Average NDCG@3 over the full-feature baseline in both evaluation modes:

- **All features:** 58.7 vs. 54.7, an improvement of 4.0 points.
- **Language features only:** 51.2 vs. 47.3, an improvement of 3.9 points.

The largest improvement occurs in entity linking, while dependency parsing and machine translation show relatively small changes.

Phylogenetic weighting introduces alternative feature selections but does not produce consistent improvements over standard PCA. Performance varies across feature counts, weighting strategies, and tasks, suggesting that the benefits of phylogenetic balancing are sensitive to the experimental configuration.

Overall, standard PCA with 300 selected features provides the strongest results among the tested configurations.


## Data and Licenses

| Source | License |
|---|---|
| URIEL+ v1.3.2 (code and database) | CC BY-SA 4.0. Attribution required, and derived vectors must be shared alike. |
| Bundled Grambank and APiCS | CC BY 4.0 |
| Bundled eWAVE | CC BY 3.0 (bundled with URIEL+ but excluded here) |
| Glottolog 5.3 | CC BY 4.0 |
| LangRank code | BSD-3-Clause |
| URIELPlus-LangRank experiment CSVs | No license on file. Acceptable for coursework, not to be redistributed publicly. |

`data.zip` contains vectors derived from URIEL+, so it carries the CC BY-SA 4.0 share-alike condition. The experiment CSVs and the reproduction zips that contain them should not be published.

## Credits and References

Aditya Khan and York Hay Ng helped with the computation plan and with selecting N.

- Khan, A., et al. (2025). URIEL+: Enhancing linguistic inclusion and usability in a typological and multilingual knowledge base. COLING 2025.
- Shipton, M., et al. (2025). Simple additions, substantial gains: Expanding scripts, languages, and lineage coverage in URIEL+. arXiv:2510.27183.
- Lin, Y.-H., et al. (2019). Choosing transfer languages for cross-lingual learning. ACL 2019.
- Ng, Y. H., Hoang, P. H., & Lee, E.-S. A. (2025). Less is more: The effectiveness of compact typological language representations. EMNLP 2025.
- Hammarström, H., Forkel, R., Haspelmath, M., & Bank, S. (2026). Glottolog 5.3. Max Planck Institute for Evolutionary Anthropology. https://glottolog.org
