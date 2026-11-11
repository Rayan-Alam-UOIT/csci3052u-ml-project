
import os
import re
import numpy as np
import pandas as pd

# Administrative Rules
# A language is an "administrative isolate" if its lineage ROOT is one of these (the root is a
# bookkeeping category, not a family), or its finest label is one of ADMIN_FINEST.
ADMIN_WHOLE_ROOT = {"Bookkeeping", "Unclassifiable", "Unattested", "Artificial Language", "Pidgin", "Mixed Language", "Speech Register"}
ADMIN_FINEST = ADMIN_WHOLE_ROOT | {"Sign Language", "L1 Sign Language", "Pending Report Release"}
# "Unclassified X" -> fall back to parent label
UNCL = re.compile(r"^Unclassified\b", re.I)


def parse(df):
    out = []
    for s in df["lineage"]:
        out.append([] if pd.isna(s) or not str(s).strip()
                   else [x.strip() for x in str(s).split(", ")])
    return out


def phylo_weights(df, level=None, alpha=1.0, admin_mode="one"):
    chains = parse(df)
    keys, kind, fallback, finest = [], [], [], []
    for lid, c in zip(df["language_id"], chains):
        fb = 0
        if not c:                                            # True isolate (own family)
            keys.append(("ISO", lid)); kind.append("isolate"); fallback.append(0); finest.append("")
            continue
        if c[0] in ADMIN_WHOLE_ROOT or c[-1] in ADMIN_FINEST:  # Administrative pseudo-group
            keys.append(("ADM", lid)); kind.append("admin"); fallback.append(0); finest.append(c[-1])
            continue
        while len(c) > 1 and UNCL.match(c[-1]):              # "Unclassified X" -> parent
            c = c[:-1]; fb = 1
        if level:
            c = c[:level]
        keys.append(tuple(c)); kind.append("normal"); fallback.append(fb); finest.append(c[-1])
    s = pd.Series(keys, dtype=object)
    n = s.map(s.value_counts()).astype(float).values        # Unique keys -> n = 1 for isolates
    w = n ** -alpha
    kind = np.array(kind)
    adm = kind == "admin"
    if adm.any() and admin_mode != "one":
        w[adm] = np.median(w[~adm]) if admin_mode == "median" else 0.0
    w = w / w.sum()
    return dict(w=w, kind=kind, n=n, fallback=np.array(fallback), finest=finest,
                depth=np.array([len(c) for c in parse(df)]))

# Configuration
csv_path = os.path.join(
    "URIELPlus",
    "urielplus",
    "database",
    "urielplus_csvs",
    "lang_fam_geo.csv"
)
nrows = 8128
out_dir = "."

df = pd.read_csv(csv_path, nrows=nrows)
lines = [f"rows used: {len(df)} (first {nrows} language rows of {csv_path})"]

P = phylo_weights(df)                                   # Primary: finest label, alpha=1
variants = {
    "w_finest": P["w"],                                  # PRIMARY (Configuration C)
    "w_level3_a1": phylo_weights(df, level=3, alpha=1.0)["w"],
    "w_level3_a05": phylo_weights(df, level=3, alpha=0.5)["w"],
    "w_finest_adminmedian": phylo_weights(df, admin_mode="median")["w"],
    "w_finest_adminzero": phylo_weights(df, admin_mode="zero")["w"],
}
uniform = np.full(len(df), 1.0 / len(df))

out = df[["language_id", "language_name", "lineage"]].copy()
out["kind"] = P["kind"]; out["depth"] = P["depth"]
out["finest_label_used"] = P["finest"]; out["unclassified_fallback"] = P["fallback"]
out["group_size"] = P["n"].astype(int)
for k, v in variants.items():
    assert abs(v.sum() - 1) < 1e-9, k
    out[k] = v
os.makedirs(out_dir, exist_ok=True)
out.to_csv(os.path.join(out_dir, "language_weights.csv"), index=False)

# Diagnostics for the Write-Up
w = P["w"]; N = len(w)
lines += ["", "Weight variants (normalized to sum to 1):"]
for k, v in variants.items():
    pos = v[v > 0]
    lines.append(f"  {k:22s} min={pos.min():.2e} median={np.median(v):.2e} max={v.max():.2e} "
                    f"max/min={v.max()/pos.min():.1f}  n_eff={1/np.sum(v**2):.0f} of {N} "
                    f"({100/np.sum(v**2)/N:.0f}%)")
lines.append(f"  (uniform weights: n_eff = {N})")
kc = pd.Series(P["kind"]).value_counts().to_dict()
lines += ["", f"language kinds: {kc}",
            f"'Unclassified' fallbacks to parent label: {int(P['fallback'].sum())}",
            f"corr(lineage depth, primary weight) = {np.corrcoef(P['depth'], w)[0,1]:.3f}"]

normal = out[out.kind == "normal"]
g = (normal.assign(path=normal.finest_label_used)
            .groupby(normal.lineage.fillna("")).size().sort_values(ascending=False))
lines += ["", "Most heavily down-weighted groups (genuine groups, by lineage; size = languages):"]
for path, cnt in g.head(10).items():
    lines.append(f"  {cnt:4d}  {path.split(', ')[-1]}   [{path[:70]}]")
roots = pd.Series([c[0] if c else "(isolate)" for c in parse(df)]).value_counts()
lines += ["", "Lineage roots (check for pseudo-families missing from the administrative list):"]
for r, cnt in roots.head(25).items():
    flag = "  <- administrative" if r in ADMIN_WHOLE_ROOT else ""
    lines.append(f"  {cnt:5d}  {r}{flag}")

txt = "\n".join(lines)
print(txt)
with open(os.path.join(out_dir, "weights_summary.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")