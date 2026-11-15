import os, re, sys, time, shutil, subprocess, zipfile
from pathlib import Path
import pandas as pd

N_VALUES = [100, 150, 200, 250, 300, 350, 400]
WEIGHT_VARIANTS = ["uniform", "w_finest", "w_finest_adminmedian", "w_finest_adminzero", "w_level3_a1", "w_level3_a05"]

# Setup
TASKS = ["dep", "el", "mt", "pos"]
MODES = ["all", "lang"]      # LangRank (all features) / LangRank (lang feats)
FORCE = False                # True -> recompute distances even if complete CSVs exist

def find_langrank_dir():
    """LANGRANK_DIR env var, else walk up from the notebook's folder."""
    env = os.environ.get("LANGRANK_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for base in [Path.cwd(), *Path.cwd().parents]:
        for cand in (base, base / "URIELPlus-LangRank"):
            if (cand / "distances" / "calculate_distances.py").exists():
                return cand
    raise FileNotFoundError("URIELPlus-LangRank repo not found. Open this notebook inside it or set LANGRANK_DIR.")

LANGRANK_DIR = find_langrank_dir()
DIST_DIR    = LANGRANK_DIR / "distances"
DB_DIR      = DIST_DIR / "urielplus" / "database"          # where URIEL+ reads its .npz files
INPUT_DIR   = LANGRANK_DIR / "experiment_csvs" / "URIEL"   # original experiment CSVs (language pairs)
CSV_DIR     = LANGRANK_DIR / "src" / "csv_datasets"        # updated experiment CSVs (output of Step 4)
RESULTS_DIR = LANGRANK_DIR / "results"
TIMING_DIR  = RESULTS_DIR / "timing_logs"
NPZ = ["family_features.npz", "script_features.npz", "geocoord_features.npz"]
DIST_COLS = ["GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC", "MORPHOLOGICAL", "SCRIPT"]
PYTHON = sys.executable

def run(cmd, hide=None, tick=None):
    """Run a command from the repo root, stream its output, return it. Raises on failure.
    hide: regex of lines not to display. tick: regex of lines to count, shown as a progress line every 250."""
    print("$", " ".join(map(str, cmd)))
    hide, tick = (re.compile(hide) if hide else None), (re.compile(tick) if tick else None)
    env = {**os.environ, "LOKY_MAX_CPU_COUNT": str(os.cpu_count() or 1)}   # silences a harmless joblib warning on Windows
    t0, n, lines = time.time(), 0, []
    proc = subprocess.Popen(cmd, cwd=LANGRANK_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        lines.append(line)
        if tick and tick.search(line):
            n += 1
            if n % 250 == 0:
                print(f"  ... {n} done ({(time.time() - t0) / 60:.1f} min)")
        if not (hide and hide.search(line)):
            print(line, end="")
    rc = proc.wait()
    print(f"-> exit {rc} in {time.time() - t0:.1f}s")
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {cmd}")
    return "".join(lines)

def collect_timing_logs():
    """The repo scripts write timing logs next to themselves; gather them in results/timing_logs/."""
    for p in [*DIST_DIR.glob("*_timing_log.txt"), *LANGRANK_DIR.glob("langrank_*_timing_log_*.txt")]:
        os.replace(p, TIMING_DIR / p.name)

SMOKE = """
import numpy as np, pandas as pd, sklearn, lightgbm as lgb
from lightgbm import LGBMRanker
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.random((60, 4)), columns=list("abcd")); y = pd.Series(rng.integers(0, 5, 60)).astype(float)
r = LGBMRanker(objective="lambdarank", n_estimators=5, num_leaves=4, min_data_in_leaf=2, verbose=-1)
r.fit(X, y, group=[20, 20, 20], eval_set=[(X[:20], y[:20])], eval_group=[[20]], eval_at=[3])
print("SMOKE_OK lightgbm", lgb.__version__, "| numpy", np.__version__, "| scikit-learn", sklearn.__version__, "| pandas", pd.__version__)
"""

def lightgbm_works():
    r = subprocess.run([PYTHON, "-c", SMOKE], capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[-300:])
    return next((l for l in r.stdout.splitlines() if l.startswith("SMOKE_OK")), None)

ENV_LINE = lightgbm_works()

def distances_complete(t):
    out = DIST_DIR / f"{t}_distances.csv"
    return out.exists() and len(pd.read_csv(out)) == len(pd.read_csv(INPUT_DIR / f"{t}.csv"))


# Load Genetic, Geographic, and Script Data
find = lambda n: next((d / n for d in (LANGRANK_DIR, LANGRANK_DIR.parent) if (d / n).exists()), None)
zpath = find("data.zip")
assert zpath, "data.zip not found next to or above the repo folder"

DB_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zpath) as z:
    for name in NPZ:
        zf, member = z, name
        with zf.open(member) as src, open(DB_DIR / name, "wb") as dst:
            shutil.copyfileobj(src, dst)
print("copied", NPZ, "from", zpath)

ZIP_PATH = LANGRANK_DIR.parent / "weighted_pca_sweep_results.zip"
ZIP_PATH.unlink(missing_ok=True)  # Start with a fresh ZIP

for N in N_VALUES:
    for W_V in WEIGHT_VARIANTS:
        # Load Phylogenetically-Weighted PCA Data
        ppath = find("jaccard_analysis/weighted_pca_sweep_output.zip")
        assert ppath, "jaccard_analysis/weighted_pca_sweep_output.zip not found next to or above the repo folder"
        with zipfile.ZipFile(zpath) as z, zipfile.ZipFile(ppath) as zp:
            zf, member = zp, f"weighted_pca_sweep_output/N_{N}/{W_V}/features.npz"
            with zf.open(member) as src, open(DB_DIR / "features.npz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        print("copied features.npz from", ppath)


        # Compute Distances
        status = {t: distances_complete(t) for t in TASKS}
        print("Distance CSVs complete:", status)
        if all(status.values()) and not FORCE:
            print("Skipping calculation (FORCE = True to recompute).")
        else:
            for t in TASKS:
                (DIST_DIR / f"{t}_distances.csv").unlink(missing_ok=True)
            run([PYTHON, "distances/calculate_distances.py"], hide=r" - root - INFO - |^\d\.\d{4},", tick=r"^\d\.\d{4},")
            collect_timing_logs()
            assert all(distances_complete(t) for t in TASKS), "Distance calculation did not produce complete CSVs"


        # Update Experiment CSVs
        run([PYTHON, "distances/replace_distances.py"])
        collect_timing_logs()
        for t in TASKS:
            df = pd.read_csv(CSV_DIR / f"{t}.csv")
            assert len(df) == len(pd.read_csv(INPUT_DIR / f"{t}.csv")), f"{t}: row count mismatch"
            assert not df[DIST_COLS].isna().any().any(), f"{t}: missing distance values"
            print(f"{t}: {len(df)} rows OK")

        # Run Experiments
        NAMES = {"dep": "Dependency Parsing (DEP)", "el": "Entity Linking (EL)",
                "mt": "Machine Translation (MT)", "pos": "Part of Speech (POS) Tagging"}
        LABEL = {"all": "LangRank (all)", "lang": "LangRank (lang feats)"}

        averages, blocks = {}, [f"# environment: {ENV_LINE}\n"]
        for t in TASKS:
            for m in MODES:
                out = run([PYTHON, f"src/langrank_models/{t}.py", "--mode", m], hide=r"^\[Leave-one-out fold")
                scores = re.findall(r"^\[[0-9eE.,\s\-]+\]\s*$", out, re.M)[-1].strip()
                avg = float(re.search(r"Average NDCG@3:\s*([0-9.]+)", out).group(1))
                averages[(t, m)] = avg
                blocks.append(f"{NAMES[t]} Experiments - {LABEL[m]}\n{scores}\nAverage NDCG@3: {avg}\n")
        (RESULTS_DIR / "scores.txt").write_text("\n".join(blocks), encoding="utf-8")
        collect_timing_logs()
        print("Wrote", RESULTS_DIR / "scores.txt")

        # Results Table
        NAMES = {"dep": "Dependency Parsing (DEP)", "el": "Entity Linking (EL)",
                "mt": "Machine Translation (MT)", "pos": "Part of Speech (POS) Tagging"}
        LABEL = {"all": "LangRank (all)", "lang": "LangRank (lang feats)"}

        text = (RESULTS_DIR / "scores.txt").read_text(encoding="utf-8")
        pattern = re.compile(r"^(.+?) Experiments - (.+)\n\[.*\]\nAverage NDCG@3:\s*([0-9.]+)", re.M)
        found = {(name, label): float(avg) for name, label, avg in pattern.findall(text)}

        rows = {}
        for t in TASKS:
            for m in MODES:
                assert (NAMES[t], LABEL[m]) in found, f"{t}/{m} not found in scores.txt"
            rows[t.upper()] = {LABEL[m]: found[(NAMES[t], LABEL[m])] for m in MODES}

        results_df = pd.DataFrame(rows).T
        results_df.loc["Mean (4 tasks)"] = results_df.loc[[t.upper() for t in TASKS]].mean()
        results_df.index.name = "Task"
        results_df = results_df.round(1)
        results_df.to_csv(RESULTS_DIR / "results_table.csv")

        # Package Results
        MOVE = True   # True -> delete the originals once the zip is checked; False -> leave them in place

        URIELPLUS_DB_DIR = DIST_DIR / "urielplus" / "database"
        URIELPLUS_DB_FILES = [
            "family_features.npz",
            "features.npz",
            "geocoord_features.npz",
            "script_features.npz",
        ]

        collect_timing_logs()

        files = {}   # File on disk -> Path inside the zip
        for t in TASKS:
            files[DIST_DIR / f"{t}_distances.csv"] = f"distances/{t}_distances.csv"
            files[CSV_DIR / f"{t}.csv"] = f"experiment_csvs/{t}.csv"
        files[RESULTS_DIR / "scores.txt"] = "scores.txt"
        files[RESULTS_DIR / "results_table.csv"] = "results_table.csv"
        for name in URIELPLUS_DB_FILES:
            files[URIELPLUS_DB_DIR / name] = f"urielplus_database/{name}"
        for p in sorted(TIMING_DIR.glob("*.txt")):
            files[p] = f"timing_logs/{p.name}"
            # Put this configuration's files inside N_xxx/weight_variant/
        prefix = f"N_{N}/{W_V}/"
        files = {src: prefix + arc for src, arc in files.items()}

        missing = [str(p) for p in files if not p.exists()]
        assert not missing, f"Cannot package, missing files (run sections 3-6 first): {missing}"
        assert any(a.startswith(prefix + "timing_logs/") for a in files.values()), "No timing logs found in results/timing_logs/"

        with zipfile.ZipFile(ZIP_PATH, "a", zipfile.ZIP_DEFLATED) as z:
            for src, arc in files.items():
                z.write(src, arc)

        with zipfile.ZipFile(ZIP_PATH) as z:
            assert z.testzip() is None, "Zip is corrupt"
            assert set(files.values()).issubset(z.namelist()), "Zip is missing files"
            for src, arc in files.items():
                assert z.getinfo(arc).file_size == src.stat().st_size, f"Size mismatch: {arc}"
            print(f"Wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB), {len(files)} files:")
            for info in z.infolist():
                print(f"  {info.filename}  ({info.file_size / 1e3:.1f} KB)")

        if MOVE:
            for src in files:
                if src.name not in NPZ:
                    src.unlink()
            print("Removed configuration-specific files (now only inside the zip).")
        else:
            print("MOVE = False: originals left in place.")

if MOVE:
    for name in NPZ:
        (DB_DIR / name).unlink(missing_ok=True)

    print("Removed shared database files after completing the sweep.")