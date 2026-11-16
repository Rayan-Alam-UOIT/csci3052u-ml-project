import re, sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["DEP", "EL", "MT", "POS"]
ROWS = TASKS + ["Mean (4 tasks)"]
MODES = {"all": "LangRank (all)", "lang": "LangRank (lang feats)"}
MODE_TITLE = {"all": "All Features", "lang": "Language Features Only"}

# Config A: full 593-feature baseline
A = {"all":  {"DEP": 76.0, "EL": 65.5, "MT": 53.6, "POS": 23.6, "Mean": 54.7},
     "lang": {"DEP": 72.8, "EL": 62.8, "MT": 33.5, "POS": 20.2, "Mean": 47.3}}

ORDER = ["uniform", "w_finest", "w_finest_adminmedian", "w_finest_adminzero", "w_level3_a05", "w_level3_a1"]
COLORS = {"uniform": "#000000", "w_finest": "#0072B2", "w_finest_adminmedian": "#56B4E9",
          "w_finest_adminzero": "#009E73", "w_level3_a05": "#E69F00", "w_level3_a1": "#D55E00"}
FALLBACK = ["#CC79A7", "#999999", "#F0E442", "#8C564B"]
NOTE = "Single deterministic runs; no uncertainty shown, so small differences may not be meaningful."


def load(root):
    rx = re.compile(r"^N_(\d+)$")
    recs = []

    with ZipFile(root) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith("/results_table.csv"):
                continue

            p = PurePosixPath(name)
            m = rx.match(p.parent.parent.name)
            if not m:
                continue

            with zf.open(name) as f:
                df = pd.read_csv(f, index_col=0)

            df.index, df.columns = df.index.str.strip(), df.columns.str.strip()

            for mode, col in MODES.items():
                if col not in df.columns:
                    sys.exit(f"{name}: column '{col}' not found (columns: {list(df.columns)})")

                for r in ROWS:
                    if r not in df.index:
                        sys.exit(f"{name}: row '{r}' not found (rows: {list(df.index)})")

                    recs.append({
                        "N": int(m.group(1)),
                        "variant": p.parent.name,
                        "task": "Mean" if r.startswith("Mean") else r,
                        "mode": mode,
                        "ndcg": float(df.loc[r, col])
                    })

    if not recs:
        sys.exit(f"No N_xxx/<variant>/results_table.csv files found in {root}")

    return pd.DataFrame(recs)


def variant_list(master):
    vs = [v for v in ORDER if v in set(master.variant)]
    return vs + sorted(set(master.variant) - set(vs))


def color(v, extra=[0]):
    if v not in COLORS:
        COLORS[v] = FALLBACK[extra[0] % len(FALLBACK)]
        extra[0] += 1
    return COLORS[v]


def check(master):
    Ns, vs = sorted(set(master.N)), sorted(set(master.variant))
    have = set(zip(master.N, master.variant))
    missing = [(n, v) for n in Ns for v in vs if (n, v) not in have]
    if missing:
        print("WARNING: missing configurations (left blank in plots):", missing)
    if "uniform" not in vs:
        sys.exit("No 'uniform' variant found: cannot compute differences from standard PCA.")
    for n in Ns:
        if (n, "uniform") not in have:
            print(f"WARNING: no uniform result at N={n}: no differences for that N")


def add_delta(master):
    u = master[master.variant == "uniform"].rename(columns={"ndcg": "uniform_ndcg"})[["N", "task", "mode", "uniform_ndcg"]]
    d = master.merge(u, on=["N", "task", "mode"], how="left")
    d["delta_vs_uniform"] = d.ndcg - d.uniform_ndcg
    return d


def wide(master, key, mode, task="Mean"):
    d = master[(master["mode"] == mode) & (master.task == task)]
    return d.pivot(index="variant", columns="N", values=key).reindex(variant_list(master))


def finish(fig, handles, labels, path):
    """Legend + footnote below the axes, with room sized to the legend (no overlap with axis labels)."""
    h_in = fig.get_figheight()
    ncol = min(len(labels), 4)
    rows = -(-len(labels) // ncol)
    foot_in, leg_in = 0.3, rows * 0.24 + 0.15
    fig.tight_layout(rect=(0, (foot_in + leg_in) / h_in, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, foot_in / h_in), ncol=ncol,
               frameon=False, fontsize=9)
    fig.text(0.01, 0.1 / h_in, NOTE, fontsize=7.5, color="#555", ha="left", va="bottom")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def line_ax(ax, master, mode, task, variants, ylabel=True):
    for v in variants:
        d = master[(master.variant == v) & (master["mode"] == mode) & (master.task == task)].sort_values("N")
        ax.plot(d.N, d.ndcg, marker="o", ms=4, lw=3 if v == "uniform" else 1.8, color=color(v),
                zorder=3 if v == "uniform" else 2, label="uniform (Config B)" if v == "uniform" else v)
    ax.axhline(A[mode][task], ls="--", lw=1.3, color="#777", label="full 593 features (Config A)")
    ax.set_xticks(sorted(set(master.N)))
    ax.grid(alpha=0.25)
    ax.set_xlabel("N (selected features)")
    if ylabel:
        ax.set_ylabel("NDCG@3 (x100)")


def fig_lines(master, variants, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mode in zip(axes, MODES):
        line_ax(ax, master, mode, "Mean", variants, ylabel=(mode == "all"))
        ax.set_title(f"Mean NDCG@3 over 4 tasks - {MODE_TITLE[mode]}", fontsize=10.5)
    h, l = axes[0].get_legend_handles_labels()
    finish(fig, h, l, out / "fig1_mean_vs_N.png")


def fig_tasks(master, variants, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for i, mode in enumerate(MODES):
        for j, t in enumerate(TASKS):
            ax = axes[i, j]
            line_ax(ax, master, mode, t, variants, ylabel=(j == 0))
            ax.set_title(f"{t} - {MODE_TITLE[mode]}", fontsize=10)
            if i == 0:
                ax.set_xlabel("")
    h, l = axes[0, 0].get_legend_handles_labels()
    finish(fig, h, l, out / "fig3_per_task_vs_N.png")


def fig_heat(master, variants, out):
    dv = [v for v in variants if v != "uniform"]
    Ns = sorted(set(master.N))
    mats = {m: wide(master, "delta_vs_uniform", m).reindex(dv)[Ns] for m in MODES}
    vmax = max(np.nanmax(np.abs(mats[m].values)) for m in MODES)
    vmax = max(vmax, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6 + 0.3 * len(dv)), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        M = mats[mode]
        im = ax.imshow(M.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(Ns)), Ns); ax.set_yticks(range(len(dv)), dv)
        ax.set_xlabel("N (selected features)")
        ax.set_title(f"Change vs uniform PCA - {MODE_TITLE[mode]}", fontsize=10.5)
        for i in range(len(dv)):
            for j in range(len(Ns)):
                x = M.values[i, j]
                if not np.isnan(x):
                    ax.text(j, i, f"{x:+.1f}", ha="center", va="center", fontsize=8.5,
                            color="white" if abs(x) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=axes, location="right", shrink=0.9, pad=0.02)
    cb.set_label("mean NDCG@3 difference (points)")
    fig.supxlabel(NOTE, fontsize=7.5, color="#555", x=0.01, ha="left")
    fig.savefig(out / "fig2_delta_heatmap.png", dpi=200)
    plt.close(fig)
    print("wrote", out / "fig2_delta_heatmap.png")



def main():
    root = Path("weighted_pca_sweep_results.zip")
    out = Path("sweep_analysis")
    out.mkdir(parents=True, exist_ok=True)

    master = load(root)
    check(master)
    master = add_delta(master).sort_values(["mode", "task", "N", "variant"])
    variants = variant_list(master)
    print(f"{master.N.nunique()} N values x {master.variant.nunique()} variants: N={sorted(set(master.N))}, variants={variants}")

    fig_lines(master, variants, out)
    fig_heat(master, variants, out)
    fig_tasks(master, variants, out)

if __name__ == "__main__":
    main()