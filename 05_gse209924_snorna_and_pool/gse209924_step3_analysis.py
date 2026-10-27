"""
Single-experiment test (GSE209924, TGIRT-seq) - Step 3: analysis of the Step 2 predictions (no model is re-run; takes seconds).

  1. Share right overall and per true RNA type, with 95% (Wilson) confidence intervals. Models without a class for a
     type (RNAErnie/ncRDense: snRNA, lncRNA) score 0 on it by design - shown as "no class".
  2. McNemar exact test (Holm-corrected) on the sequences ALL four models can score: miRNA, snoRNA, tRNA, rRNA.
     And MncR full vs MncR SeqEnc on all sequences.
  3. Confusion tables: what each model called each true type.
  4. Exact overlap of the sequences with the nRC training / test sets.
2b. Precision, recall, F1, accuracy, balanced accuracy, MCC and Cohen kappa (with bootstrap CIs), for all types and for the types all models can name.
Run:  python gse209924_step3_analysis.py          Output: gse209924/results/analysis/
"""
import itertools
import os
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("GSE209924_DIR", BASE / "gse209924"))
DATA, RESULTS = DIR / "data", DIR / "results"
OUT = RESULTS / "analysis"
RNAERNIE = Path(os.environ.get("RNAERNIE_DATA", BASE / "mirna" / "RNAErnie" / "data"))
MAP = {"CD-box": "snoRNA", "HACA-box": "snoRNA", "scaRNA": "snoRNA", "5S_rRNA": "rRNA", "5_8S_rRNA": "rRNA"}
norm = lambda x: MAP.get(x, x)
SHORT = {"MncR SeqEnc (sequence only)": "MncR SeqEnc", "MncR full (sequence + graph)": "MncR full"}
ALL_TYPES = ["miRNA", "snoRNA", "tRNA", "rRNA", "snRNA", "lncRNA"]
NO_CLASS = {"RNAErnie": {"snRNA", "lncRNA"}, "ncRDense": {"snRNA", "lncRNA"}}


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holm(p):
    p = np.asarray(p, float); order = p.argsort(); out = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order): run = max(run, min(1.0, (len(p) - rank) * p[i])); out[i] = run
    return out


def fasta_set(path):
    seqs, cur = [], None
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith(">"): cur = []; seqs.append(cur)
        elif line and cur is not None: cur.append(line)
    return {"".join(s).upper().replace("T", "U") for s in seqs}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RESULTS / "pool_predictions.csv")
    cols = [m for m in ["RNAErnie", "ncRDense", "MncR SeqEnc (sequence only)", "MncR full (sequence + graph)"] if m in d]
    ok = pd.DataFrame({SHORT.get(m, m): d[m].map(norm).values == d["true"].values for m in cols})
    models = list(ok.columns); n = len(d)
    TYPES = [x for x in ALL_TYPES if (d["true"] == x).any()]      # only the types present in this dataset
    pd.set_option("display.width", 230)
    print("sequences by true type:", d["true"].value_counts().to_dict())

    rows = []
    for m in models:
        k = int(ok[m].sum()); lo, hi = wilson(k, n)
        rows.append({"model": m, "type": "ALL", "n": n, "right": k, "share": k / n, "ci_low": lo, "ci_high": hi, "note": ""})
        for t in TYPES:
            msk = (d["true"] == t).values
            if not msk.any(): continue
            k = int(ok[m][msk].sum()); lo, hi = wilson(k, int(msk.sum()))
            rows.append({"model": m, "type": t, "n": int(msk.sum()), "right": k, "share": k / msk.sum(), "ci_low": lo, "ci_high": hi,
                         "note": "no class" if t in NO_CLASS.get(m, set()) else ""})
    r1 = pd.DataFrame(rows); r1.round(4).to_csv(OUT / "1_share_right_ci.csv", index=False)
    print("\n=== 1. Share right, overall and per true type (95% CI) ===")
    print(r1.pivot(index="type", columns="model", values="share").reindex(["ALL"] + TYPES)[models].round(3).to_string())
    print(r1[r1.type != "ALL"].groupby("type")["n"].first().to_string())

    common = d["true"].isin(["miRNA", "snoRNA", "tRNA", "rRNA"]).values
    prow = []
    for a, b in itertools.combinations(models, 2):
        msk = common if not (a.startswith("MncR") and b.startswith("MncR")) else np.ones(n, bool)
        x, y = int((ok[a] & ~ok[b])[msk].sum()), int((~ok[a] & ok[b])[msk].sum())
        prow.append({"model_A": a, "model_B": b, "scope": "all" if msk.all() else "types all models can name", "n": int(msk.sum()),
                     "A_right_B_wrong": x, "B_right_A_wrong": y, "p_value": binomtest(x, x + y, 0.5).pvalue if x + y else 1.0})
    r2 = pd.DataFrame(prow); r2["p_holm"] = holm(r2["p_value"]); r2["real_gap"] = r2["p_holm"] < 0.05
    for c in ("p_value", "p_holm"): r2[c] = [float(f"{v:.3g}") for v in r2[c]]
    r2.to_csv(OUT / "2_mcnemar.csv", index=False)
    print(f"\n=== 2. McNemar exact test (Holm-corrected for {len(r2)} comparisons) ===\n" + r2.to_string(index=False))


    # ---- 2b. classification metrics: precision, recall, F1, accuracy, balanced accuracy, MCC, kappa (+ bootstrap CIs)
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef,
                                 precision_recall_fscore_support)
    def lab(series):                                         # any call that is not one of our types counts as "other"
        return series.map(norm).where(series.map(norm).isin(TYPES), "other").values
    truth = d["true"].values; rng = np.random.default_rng(0)
    def view(mask, types, tag):
        rows, per = [], []
        for m, mc in zip(models, cols):
            y, p = truth[mask], lab(d[mc])[mask]
            lbls = types + ["other"]
            pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=types, zero_division=0)
            for t, a, b, c, s_ in zip(types, pr, rc, f1, sup):
                per.append({"view": tag, "model": m, "type": t, "precision": a, "recall": b, "F1": c, "support": int(s_),
                            "note": "no class" if t in NO_CLASS.get(m, set()) else ""})
            def stats(idx):
                yy, pp = y[idx], p[idx]
                return [accuracy_score(yy, pp), balanced_accuracy_score(yy, pp), f1_score(yy, pp, labels=types, average="macro", zero_division=0),
                        f1_score(yy, pp, labels=types, average="weighted", zero_division=0), matthews_corrcoef(yy, pp), cohen_kappa_score(yy, pp)]
            base = stats(np.arange(len(y)))
            boot = np.array([stats(rng.integers(0, len(y), len(y))) for _ in range(500)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            row = {"view": tag, "model": m, "n": len(y)}
            for i, k in enumerate(["accuracy", "balanced_accuracy", "macro_F1", "weighted_F1", "MCC", "kappa"]):
                row[k] = base[i]; row[k + "_ci"] = f"{lo[i]:.3f}-{hi[i]:.3f}"
            rows.append(row)
        return pd.DataFrame(rows), pd.DataFrame(per)
    allv, perv = view(np.ones(n, bool), TYPES, f"all {len(TYPES)} types")
    common_types = [x for x in ["miRNA", "snoRNA", "tRNA", "rRNA"] if x in TYPES]
    cv, cper = view(np.isin(truth, common_types), common_types, "types all models can name")
    summ = pd.concat([allv, cv]); per = pd.concat([perv, cper])
    summ.round(4).to_csv(OUT / "2b_overall_metrics.csv", index=False); per.round(4).to_csv(OUT / "2b_per_type_precision_recall_f1.csv", index=False)
    print("\n=== 2b. Overall metrics (95% bootstrap CI in the *_ci columns; 'no class' types count as errors in the first view) ===")
    for v, g in summ.groupby("view", sort=False):
        print(f"\n--- view: {v} ---")
        print(g.drop(columns=["view"]).round(3)[["model", "n", "accuracy", "balanced_accuracy", "macro_F1", "weighted_F1", "MCC", "kappa"]].to_string(index=False))
        print(g[["model", "accuracy_ci", "macro_F1_ci", "MCC_ci"]].to_string(index=False))
    print("\n=== 2b. Per type: precision / recall / F1 (precision = of the sequences a model called X, how many really are X) ===")
    for v, g in per.groupby("view", sort=False):
        print(f"\n--- view: {v} ---")
        for met in ("precision", "recall", "F1"):
            print(f"[{met}]"); print(g.pivot(index="type", columns="model", values=met).reindex(common_types if v.startswith("types") else TYPES)[models].round(3).to_string())

    print("\n=== 3. What each model called each true type (raw class names, mapped as in Step 2) ===")
    for m, mc in zip(models, cols):
        ct = pd.crosstab(d["true"], d[mc].map(norm)); ct.to_csv(OUT / f"3_confusion_{m.replace(' ', '_')}.csv")
        print(f"\n--- {m} ---\n{ct.to_string()}")

    seqs = {}; cur = None
    for line in open(DATA / "pool.fasta"):
        line = line.strip()
        if line.startswith(">"): cur = line[1:].split()[0]; seqs[cur] = ""
        else: seqs[cur] += line.upper().replace("T", "U")
    tr_f, te_f = RNAERNIE / "nrc" / "dataset_Rfam_6320_13classes.fasta", RNAERNIE / "nrc" / "dataset_Rfam_validated_2600_13classes.fasta"
    if tr_f.exists() and te_f.exists():
        tr, te = fasta_set(tr_f), fasta_set(te_f)
        s = d["id"].map(seqs); tri, tei = s.isin(tr), s.isin(te)
        print(f"\n=== 4. Exact overlap with nRC: in training {int(tri.sum())}, in test {int(tei.sum())}, either {int((tri | tei).sum())} of {n} ===")
        print(pd.crosstab(d["true"], tri | tei).to_string()); print("(exact matches only; MncR's own training set was not checked)")
    else:
        print("\n4. skipped: nRC FASTA files not found in", RNAERNIE / "nrc")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv = r1[r1.type.isin(TYPES)].pivot(index="type", columns="model", values="share").reindex(TYPES)[models]
    ns = r1[r1.type != "ALL"].groupby("type")["n"].first().reindex(TYPES)
    fig, ax = plt.subplots(figsize=(8, 3.9)); w = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(np.arange(len(TYPES)) + i * w, pv[m].fillna(0), w, label=m)
    ax.set_xticks(np.arange(len(TYPES)) + 0.4 - w / 2); ax.set_xticklabels([f"{t}\n(n={int(ns[t])})" for t in TYPES])
    ax.set_ylabel("Share called the right type"); ax.set_ylim(0, 1.08); ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("GSE236538: expressed ncRNAs by true type (0 = model has no class for it)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT / "pool_figure.png", dpi=200)

    fig2, axs = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.6))
    axs = np.atleast_1d(axs); order = TYPES + ["other"]
    for a, m, mc in zip(axs, models, cols):
        ct = pd.crosstab(pd.Categorical(truth, TYPES), pd.Categorical(lab(d[mc]), order), dropna=False).values.astype(float)
        ct = ct / np.maximum(ct.sum(axis=1, keepdims=True), 1)
        a.imshow(ct, vmin=0, vmax=1, cmap="Blues")
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
        a.set_yticks(range(len(TYPES))); a.set_yticklabels(TYPES, fontsize=7); a.set_title(m, fontsize=8)
        for i in range(len(TYPES)):
            for j in range(len(order)):
                if ct[i, j] >= 0.005: a.text(j, i, f"{ct[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if ct[i, j] > 0.6 else "black")
    fig2.suptitle("Share of each true type (rows) given each call (columns); rows add up to 1", fontsize=8)
    fig2.tight_layout(); fig2.savefig(OUT / "confusion_heatmaps.png", dpi=200)
    print(f"\nsaved everything in {OUT}")


if __name__ == "__main__":
    main()
