"""
GSE17681 miRNA test: recall with 95% Wilson CI and exact McNemar tests (Holm-corrected) between the models.

Every GSE17681 sequence is a miRNA, so "right" means the model called it miRNA.
Run (any folder, conda env mini_project):   python gse17681_mcnemar.py
Reads the four models' prediction files made earlier (steps 10, 5 and mncr_step5), for both sets:
  mirna_precursors (674)  and  mirna_mature_probes (866).
Saves gse17681_recall.csv and gse17681_mcnemar.csv next to the script's working folder.
"""
import itertools, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
FILES = {   # model -> (folder, file pattern, column with the predicted class)
    "RNAErnie":    (BASE / "data",                      "{s}_predictions.csv",          "predicted_class"),
    "ncRDense":    (BASE / "mirna" / "ncrdense" / "data", "{s}_ncrdense_predictions.csv", "predicted_class"),
    "MncR SeqEnc": (BASE / "mirna" / "mncr" / "data",   "{s}_mncr_predictions.csv",     "seqenc_class"),
    "MncR full":   (BASE / "mirna" / "mncr" / "data",   "{s}_mncr_predictions.csv",     "mncr_class"),
}
SETS = {"mirna_precursors": "precursor hairpins", "mirna_mature_probes": "mature probes"}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj

rec_rows, mc_rows = [], []
for s, label in SETS.items():
    right = {}
    for m, (folder, pat, col) in FILES.items():
        f = folder / pat.format(s=s)
        if not f.exists():                       # not in the expected folder: search everywhere under BASE
            hits = [h for h in BASE.rglob(pat.format(s=s)) if "gse209924" not in str(h) and "gse310677" not in str(h)]
            if not hits: sys.exit(f"missing file: {pat.format(s=s)}  (looked in {folder} and everywhere under {BASE})")
            if len(hits) > 1: print(f"NOTE: several copies of {pat.format(s=s)} found, using {hits[0]}:", *map(str, hits), sep="\n   ")
            f = hits[0]
        print(f"{m:12s} {label:18s} <- {f}")
        d = pd.read_csv(f)
        right[m] = pd.Series((d[col].values == "miRNA"), index=d["id"].astype(str).values)
    ids = sorted(set.intersection(*[set(v.index) for v in right.values()]))
    n_all = {m: len(v) for m, v in right.items()}
    if len(set(n_all.values())) != 1 or len(ids) != n_all["RNAErnie"]:
        print(f"NOTE {label}: files differ in size/ids {n_all}; using the {len(ids)} ids found in all four")
    R = pd.DataFrame({m: v.loc[ids].values for m, v in right.items()}, index=ids)
    n = len(R)
    for m in R:
        k = int(R[m].sum()); lo, hi = wilson(k, n)
        rec_rows.append({"set": label, "model": m, "n": n, "called_miRNA": k, "recall": round(k / n, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    rows = []
    for a, b in itertools.combinations(R.columns, 2):
        a_only = int((R[a] & ~R[b]).sum()); b_only = int((~R[a] & R[b]).sum())
        p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
        rows.append({"set": label, "model_A": a, "model_B": b, "n": n, "A_right_B_wrong": a_only, "B_right_A_wrong": b_only, "p_value": p})
    ph = holm([r["p_value"] for r in rows])
    for r, x in zip(rows, ph): r["p_holm"] = x; r["real_gap"] = bool(x < 0.05)
    mc_rows += rows

rec = pd.DataFrame(rec_rows); mc = pd.DataFrame(mc_rows)
pd.set_option("display.width", 200)
print("\n=== Share called miRNA (recall), 95% Wilson CI ===")
print(rec.to_string(index=False))
print("\n=== McNemar exact, Holm-corrected within each set (6 comparisons) ===")
print(mc.assign(p_value=mc.p_value.map("{:.2g}".format), p_holm=mc.p_holm.map("{:.2g}".format)).to_string(index=False))
rec.to_csv("gse17681_recall.csv", index=False); mc.to_csv("gse17681_mcnemar.csv", index=False)
print("\nsaved gse17681_recall.csv and gse17681_mcnemar.csv in", Path.cwd())
