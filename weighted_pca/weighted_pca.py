import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
from scipy.linalg import svd

N = 300
VARIANCE = 0.95
INPUT_ZIP = 'data.zip'
BASELINE_ZIP = 'standard_pca.zip'
WEIGHTS_CSV = "phylogenetic_weights/language_weights.csv"
OUTPUT_ZIP = 'weighted_pca.zip'
WEIGHT_COLUMNS = ['uniform', 'w_finest', 'w_level3_a1', 'w_level3_a05',
                  'w_finest_adminmedian', 'w_finest_adminzero']

def read_npz_from_zip(path, baseline=False):
    with zipfile.ZipFile(path) as z:
        matches = [p for p in z.namelist() if os.path.basename(p) == 'features.npz' and
                   (baseline or 'non_imputed_data' not in p)]
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

def main():
    original = read_npz_from_zip(INPUT_ZIP)
    baseline = read_npz_from_zip(BASELINE_ZIP, baseline=True)
    x = matrix(original)
    feats = np.asarray(original['feats']).astype(str)
    langs = np.asarray(original['langs']).astype(str)
    baseline_feats = set(np.asarray(baseline['feats']).astype(str))
    if len(feats) != len(set(feats)) or len(langs) != len(set(langs)):
        raise ValueError('Duplicate feature or language IDs in input')
    if x.shape[0] != 8128 or x.shape[1] != 593:
        raise ValueError(f'Expected original 8128 x 593 matrix, got {x.shape}')
    if len(baseline_feats) != N or not baseline_feats.issubset(set(feats)):
        raise ValueError('Baseline features do not match the original feature universe')

    weights_df = pd.read_csv(WEIGHTS_CSV)
    if weights_df['language_id'].duplicated().any():
        raise ValueError('Duplicate language IDs in weights CSV')
    aligned = weights_df.set_index('language_id').reindex(langs)
    if aligned['w_finest'].isna().any():
        raise ValueError('Some original language IDs are missing from weights CSV')

    summary = []
    outputs = {}
    selected = {}
    for col in WEIGHT_COLUMNS:
        w = np.ones(len(langs)) if col == 'uniform' else aligned[col].to_numpy(dtype=float)
        keep, scores, pca = select(x, w, N)
        chosen = set(feats[keep])
        selected[col] = chosen
        overlap = len(chosen & baseline_feats)
        print(f'{col}: {overlap}/{N} baseline features; {pca.n_components_} PCs; '
              f'explained variance={pca.explained_variance_ratio_.sum():.6f}')
        summary.append({'weight_variant': col, 'overlap_with_standard': overlap,
                        'added': N - overlap, 'removed': N - overlap,
                        'n_components': int(pca.n_components_),
                        'explained_variance': float(pca.explained_variance_ratio_.sum())})
        outputs[f'{col}/scores.csv'] = pd.DataFrame({'feature': feats, 'score': scores}).to_csv(index=False)
        outputs[f'{col}/added_features.csv'] = pd.DataFrame({'feature': sorted(chosen - baseline_feats)}).to_csv(index=False)
        outputs[f'{col}/removed_features.csv'] = pd.DataFrame({'feature': sorted(baseline_feats - chosen)}).to_csv(index=False)
        out_data = x[:, keep]
        if original['data'].ndim == 3:
            out_data = out_data[:, :, None]
        buff = io.BytesIO()
        sources = original['sources']
        if sources.ndim >= 1 and sources.shape[0] == len(feats) and len(feats) != len(langs):
            sources = sources[keep]
        np.savez(buff, data=out_data, feats=original['feats'][keep],
                 langs=original['langs'], sources=sources)
        outputs[f'{col}/features.npz'] = buff.getvalue()

    if selected['uniform'] != baseline_feats:
        missing = sorted(baseline_feats - selected['uniform'])
        extra = sorted(selected['uniform'] - baseline_feats)
        raise AssertionError(f'Uniform PCA failed to reproduce sklearn top 300. '
                             f'Missing: {missing}; extra: {extra}. '
                             f'Check input data, PCA version, and ties.')
    print('PASS: uniform weighted PCA reproduces all 300 standard features.')
    outputs['comparison_summary.csv'] = pd.DataFrame(summary).to_csv(index=False)
    outputs['metadata.json'] = json.dumps({'N': N, 'variance_threshold': VARIANCE,
                                           'input_shape': list(x.shape), 'summary': summary}, indent=2)
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for path, content in outputs.items():
            z.writestr(path, content)
    print(f'Saved {OUTPUT_ZIP}')

if __name__ == '__main__':
    main()