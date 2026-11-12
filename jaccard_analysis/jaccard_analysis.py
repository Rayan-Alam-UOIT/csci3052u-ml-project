import io
import json
import os
import shutil
import zipfile

import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

N_VALUES = [100, 150, 200, 250, 300, 350, 400] # Feature selection sizes to sweep.
VARIANCE = 0.95
INPUT_ZIP = os.path.join(PROJECT_ROOT, 'data.zip')
WEIGHTS_CSV = os.path.join(PROJECT_ROOT, 'phylogenetic_weights', 'language_weights.csv')
WEIGHT_COLUMNS = ['uniform', 'w_finest', 'w_level3_a1', 'w_level3_a05',
                  'w_finest_adminmedian', 'w_finest_adminzero']

OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, 'weighted_pca_sweep_output')
MASTER_SUMMARY_CSV = os.path.join(SCRIPT_DIR, 'master_sweep_summary.csv')

def read_npz_from_zip(path):
    with zipfile.ZipFile(path) as z:
        matches = [p for p in z.namelist() if os.path.basename(p) == 'features.npz' and
                   'non_imputed_data' not in p]
        if not matches:
            raise FileNotFoundError(f'No imputed features.npz in {path}')
        if len(matches) > 1:
            raise ValueError(f'Multiple candidate features.npz files in {path}: {matches}')
        with np.load(io.BytesIO(z.read(matches[0])), allow_pickle=True) as a:
            return {k: a[k].copy() for k in a.files}

def matrix(d):
    x = np.asarray(d['data'], dtype=np.float64)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
    if x.ndim != 2 or x.shape != (len(d['langs']), len(d['feats'])):
        raise ValueError(f'Unexpected matrix shape: {x.shape}')
    if not np.isfinite(x).all() or (x == -1).any():
        raise ValueError('Input contains missing/non-finite values')
    return x

def select(x, weights, n):
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (x.shape[0],) or not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
        raise ValueError('Weights must be nonnegative, finite, have positive sum, and match language rows')
    w = w / w.sum()
    mean = np.average(x, axis=0, weights=w)
    transformed = (x - mean) * np.sqrt(x.shape[0] * w)[:, None]
    _, singular_values, vt = svd(transformed, full_matrices=False, lapack_driver='gesdd')
    variance = singular_values ** 2
    ratios = variance / variance.sum()
    count = int(np.searchsorted(np.cumsum(ratios), VARIANCE) + 1)
    components = vt[:count]
    scores = np.max(np.abs(components), axis=0)
    class PCAResult:
        n_components_ = count
        explained_variance_ratio_ = ratios[:count]
    pca = PCAResult()
    order = np.argsort(-scores, kind='stable')
    keep = np.sort(order[:n])
    return keep, scores, pca

def standard_baseline(x, feats, n):
    pca = PCA(n_components=VARIANCE)
    pca.fit(x)
    scores = np.max(np.abs(pca.components_), axis=0)
    order = np.argsort(scores)[::-1]
    keep = np.sort(order[:n])
    return set(feats[keep])


def main():
    # Load dataset and exract arrays
    original = read_npz_from_zip(INPUT_ZIP)
    x = matrix(original)
    feats = np.asarray(original['feats']).astype(str)
    langs = np.asarray(original['langs']).astype(str)
    sources = original['sources']

    if len(feats) != len(set(feats)) or len(langs) != len(set(langs)):
        raise ValueError('Duplicate feature or language IDs in input')
    if x.shape[0] != 8128 or x.shape[1] != 593:
        raise ValueError(f'Expected original 8128 x 593 matrix, got {x.shape}')

    # Load and align phylogenetic weights with language rows
    weights_df = pd.read_csv(WEIGHTS_CSV)
    if weights_df['language_id'].duplicated().any():
        raise ValueError('Duplicate language IDs in weights CSV')
    aligned = weights_df.set_index('language_id').reindex(langs)
    if aligned['w_finest'].isna().any():
        raise ValueError('Some original language IDs are missing from weights CSV')

    if os.path.exists(OUTPUT_BASE_DIR):
        shutil.rmtree(OUTPUT_BASE_DIR)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    master_summary = []

    # N sweep loop
    for n in N_VALUES:
        print(f"\nN = {n} Results")
        
        n_dir = os.path.join(OUTPUT_BASE_DIR, f"N_{n}")
        os.makedirs(n_dir, exist_ok=True)

        baseline_feats = standard_baseline(x, feats, n)

        summary = []

        # Evaluate phylogenetic weight variants against the baseline
        for col in WEIGHT_COLUMNS:
            w = np.ones(len(langs)) if col == 'uniform' else aligned[col].to_numpy(dtype=float)
            keep, scores, pca = select(x, w, n)
            chosen = set(feats[keep])

            # Jaccard metrics
            overlap = len(chosen & baseline_feats)
            jaccard_sim = overlap / len(chosen | baseline_feats)
            jaccard_dist = 1.0 - jaccard_sim

            print(f"  [{col}] Overlap: {overlap}/{n} | Jaccard Dist: {jaccard_dist:.4f} | PCs: {pca.n_components_}")

            summary.append({
                'N': n,
                'weight_variant': col,
                'overlap_with_standard': overlap,
                'jaccard_similarity': round(jaccard_sim, 4),
                'jaccard_distance': round(jaccard_dist, 4),
                'added': n - overlap,
                'removed': n - overlap,
                'n_components': int(pca.n_components_),
                'explained_variance': float(pca.explained_variance_ratio_.sum())
            })

            master_summary.append(summary[-1])

            # Save variant outputs
            var_dir = os.path.join(n_dir, col)
            os.makedirs(var_dir, exist_ok=True)

            pd.DataFrame({'feature': feats, 'score': scores}).to_csv(os.path.join(var_dir, 'scores.csv'), index=False)
            pd.DataFrame({'feature': sorted(chosen - baseline_feats)}).to_csv(os.path.join(var_dir, 'added_features.csv'), index=False)
            pd.DataFrame({'feature': sorted(baseline_feats - chosen)}).to_csv(os.path.join(var_dir, 'removed_features.csv'), index=False)

            out_data = x[:, keep]

            if original['data'].ndim == 3:
                out_data = out_data[:, :, None]

            sub_sources = sources[keep] if sources.ndim >= 1 and sources.shape[0] == len(feats) and len(feats) != len(langs) else sources

            np.savez(os.path.join(var_dir, 'features.npz'), data=out_data, feats=feats[keep], langs=langs, sources=sub_sources)

        # Save summary tables and metadata for each N
        pd.DataFrame(summary).to_csv(os.path.join(n_dir, 'comparison_summary.csv'), index=False)
        with open(os.path.join(n_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({'N': n, 'variance_threshold': VARIANCE, 'summary': summary}, f, indent=2)

    # Aggregated master summary
    pd.DataFrame(master_summary).to_csv(MASTER_SUMMARY_CSV, index=False)

    # ZIP output folder and remove uncompressed directory
    zip_path = shutil.make_archive(OUTPUT_BASE_DIR, 'zip', root_dir=SCRIPT_DIR, base_dir='weighted_pca_sweep_output')
    shutil.rmtree(OUTPUT_BASE_DIR)

    print(f"\nSummary saved to: {MASTER_SUMMARY_CSV}")
    print(f"Full results saved to: {zip_path}")

if __name__ == '__main__':
    main()