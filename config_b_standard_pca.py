import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

N = 300
IN_ZIP = "data.zip"
OUT_ZIP = "standard_pca.zip"

EXPLAINED_VARIANCE = 0.95


def prefix_of(name):
    name = str(name)
    return "INV_" if name.startswith("INV_") else name[0] + "_"


if not os.path.isfile(IN_ZIP):
    raise FileNotFoundError(f"Input ZIP file not found: {IN_ZIP}")

with tempfile.TemporaryDirectory() as temp_dir:
    extract_dir = os.path.join(temp_dir, "extracted")
    output_dir = os.path.join(temp_dir, "output")

    os.makedirs(extract_dir)
    os.makedirs(output_dir)

    with zipfile.ZipFile(IN_ZIP, "r") as archive:
        archive.extractall(extract_dir)

    input_path = None
    for root, dirs, files in os.walk(extract_dir):
        if "features.npz" in files:
            input_path = os.path.join(root, "features.npz")
            break

    if not input_path or not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Expected imputed features.npz not found anywhere in {IN_ZIP}"
        )

    print(f"Using input: {input_path}")

    with np.load(input_path, allow_pickle=True) as npz:
        data = npz["data"]
        feats = npz["feats"]
        langs = npz["langs"]
        sources = npz["sources"]

    print(f"Loaded data {data.shape}, feats {feats.shape}, langs {langs.shape}, sources {sources.shape}")

    had_trailing_axis = data.ndim == 3 and data.shape[-1] == 1

    if had_trailing_axis:
        data = data[:, :, 0]
        print(f"Squeezed trailing axis -> data {data.shape}")

    assert data.shape == (len(langs), len(feats)), "data must be (languages x features)"

    assert not np.any(data == -1) and not np.isnan(data.astype(float)).any(), "matrix still has missing values; use the IMPUTED matrix"

    assert 1 <= N <= len(feats), "N must be between 1 and {len(feats)}"

    X = data.astype(np.float64)

    pca = PCA(n_components=EXPLAINED_VARIANCE)
    pca.fit(X)

    print(f"Number of PCs retained at {EXPLAINED_VARIANCE:.0%} variance: {pca.n_components_}")
    print(f"Actual explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    scores = np.max(np.abs(pca.components_), axis=0)

    scores_path = os.path.join(output_dir, "standard_PCA_scores.csv")

    pd.DataFrame({"feature": feats, "score": scores}).to_csv(scores_path, index=False)

    print(f"Saved scores -> {scores_path}")

    order = np.argsort(scores)[::-1]
    keep = np.sort(order[:N])

    if (sources.ndim >= 1 and sources.shape[0] == len(feats) and sources.shape[0] != len(langs)):
        new_sources = sources[keep]
    else:
        new_sources = sources

    out_path = os.path.join(output_dir, "features.npz")

    out_data = data[:, keep]

    if had_trailing_axis:
        out_data = out_data[:, :, None]

    np.savez(out_path, data=out_data, feats=feats[keep], langs=langs, sources=new_sources)

    print(f"Saved reduced matrix -> {out_path}")

    meta = {
        "N": int(N),
        "n_components_95pct": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "n_languages": int(len(langs)),
        "n_features_total": int(len(feats)),
    }

    meta_path = os.path.join(output_dir, f"standard_PCA_{N}_meta.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with np.load(out_path, allow_pickle=True) as chk:
        expected_shape = (
            (len(langs), N, 1)
            if had_trailing_axis
            else (len(langs), N)
        )

        assert chk["data"].shape == expected_shape, f"got {chk['data'].shape}, expected {expected_shape}"

        assert len(chk["feats"]) == N
        assert np.array_equal(chk["langs"], langs)

        assert np.array_equal(chk["data"].reshape(len(langs), N), data[:, keep])

        assert (chk["sources"].shape == sources.shape or sources.shape[0] == len(feats))

        assert set(chk["feats"]) <= set(feats)

    print("Reload checks passed.")

    full = pd.Series(
        [prefix_of(f) for f in feats]
    ).value_counts()

    sel = pd.Series([prefix_of(f) for f in feats[keep]]).value_counts()

    comp = pd.DataFrame({"all_features": full, f"selected_{N}": sel}).fillna(0).astype(int)

    comp["pct_selected"] = (100 * comp[f"selected_{N}"] / N).round(1)

    print("\nFeature-type composition:")
    print(comp)

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in ("features.npz", "standard_PCA_scores.csv", f"standard_PCA_{N}_meta.json"):
            file_path = os.path.join(output_dir, filename)
            archive.write(file_path, arcname=filename)

    print(f"\nCreated output archive: {OUT_ZIP}")

    with zipfile.ZipFile(OUT_ZIP, "r") as archive:
        print("ZIP contents:")
        for name in archive.namelist():
            print(f"  {name}")

print("\nDone.")