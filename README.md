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

_This README is a work in progress and will be updated as the project develops._

## Setup

Requirements: Python 3.10 or newer (the pinned versions were verified on Python 3.11.3) and about 3 hours of uninterrupted compute for each of Steps 3 and 4.

```
conda create -n urielplus python=3.11
conda activate urielplus
pip install -r requirements.txt
```

Open each notebook with the `urielplus` environment as its kernel. The LangRank notebooks check that LightGBM works (a small ranking fit) and install a compatible LightGBM and scikit-learn pair if it crashes.

## Running the Pipeline

Run the Steps in order. Every notebook is Run All, and each one stops with an error if a check fails. ZIP extraction, data preparation, and packaging are handled automatically by the scripts and notebooks.

| Step | Run | Produces | Time |
|---|---|---|---|
| 1 | `milestone_2_data.ipynb` | `data.zip` | minutes |
| 2 | `config_b_standard_pca.py` | `standard_pca.zip` | seconds |
| 3 | `URIELPlus-LangRank/langrank_uriel_plus_config_a.ipynb` | `base_urielplus_reproduction.zip` | about 2.5 to 3 h |
| 4 | `URIELPlus-LangRank/langrank_uriel_plus_config_b.ipynb` | `standard_pca_reproduction.zip` | about 2.5 |

### Step 1: Build the Imputed Matrix

Run `milestone_2_data.ipynb`. It builds the URIEL+ configuration below and automatically packages the resulting arrays into `data.zip` at the repository root.

- aggregation mode `U` (union)
- source databases UPDATED_SAPHON, BDPROTO, GRAMBANK and APiCS
- Glottolog integration skipped and eWAVE excluded
- BFS genetic imputation, then SoftImpute

The result is an 8,128-language by 593-feature binary typological matrix (`data/features.npz`) plus the family, script and geocoordinate arrays. The pre-imputation version is kept in `data/non_imputed_data/`.

### Step 2: Reduce the Features (Config B)

From the repository root, run:

```
python config_b_standard_pca.py --n 300
```

The script automatically extracts the required data from `data.zip`, fits a centred, unstandardized PCA at 95% explained variance, scores each feature by its maximum absolute loading over the retained components, and keeps the top N features in their original column order.

It writes three files and packages them into `standard_pca.zip`:

- `features.npz`, the reduced matrix
- `standard_PCA_scores.csv`, the score for every one of the 593 features
- `standard_PCA_300_meta.json`, including the number of components kept

### Steps 3 and 4: Run LangRank (Configs A and B)

Open the notebook inside the `URIELPlus-LangRank` folder (or set the `LANGRANK_DIR` environment variable) and Run All. Each notebook:

1. Checks the environment,
2. Places the URIEL+ `.npz` files into `distances/urielplus/database/` (Config A takes all four from `data.zip`; Config B takes `features.npz` from `standard_pca.zip` and the other three from `data.zip`),
3. Checks the placed data by recomputing the Czech–Russian distance row,
4. Recomputes the distances for the four tasks (the slow part, over 5,754 language pairs),
5. Writes the distances into the experiment CSVs,
6. Runs the eight experiments (DEP, EL, MT, POS, each in `all` and `lang` mode),
7. Shows a table of average NDCG@3 and packages everything generated into its reproduction zip.

Keep the machine awake and the kernel running. If the distance step is interrupted, rerun the notebook and that Step restarts from scratch.

## Outputs

Each reproduction zip contains the distance CSVs, the updated experiment CSVs, the timing logs, `scores.txt` (per-fold and average NDCG@3 for every task and mode) and `results_table.csv` (average NDCG@3 per task and mode, plus the mean over the four tasks). To compare configurations, read `results_table.csv` from `base_urielplus_reproduction.zip` and `standard_pca_reproduction.zip`.

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
