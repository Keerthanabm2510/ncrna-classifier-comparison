"""
snoRNA - Step 3: deeper analysis of the Step 2 predictions (no models are re-run; takes seconds).

Answers, for RNAErnie, ncRDense, MncR SeqEnc and MncR full on the GSE209924 snoRNAs:
  1. How often is each model right (snoRNA recall) with a 95% confidence interval?
  2. Are the differences between models real? (McNemar exact test, every pair, Holm-corrected)
  3. Does it depend on the snoRNA subtype (CD-box / HACA-box / scaRNA / unnamed)?
  4. Does it depend on length?  5. Does it depend on how strongly the snoRNA is expressed (TPM)?
  6. Do the models fail on the same snoRNAs? (how many models got each one right)
  7. Could the sequences have been seen in training? (exact match with the nRC training / test sets)
  8. RNAErnie / ncRDense only: subtype confusion (CD-box vs HACA-box).

Run:  python snorna_step3_analysis.py        (from the snorna folder or anywhere)
Output (snorna/results/analysis/): CSV tables, snorna_analysis_figure.png, snorna_missed_by_all.csv
"""
import itertools
import os
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("SNORNA_DIR", BASE / "snorna"))
DATA = DIR / "data"             # inputs
RESULTS = DIR / "results"       # step 2 output; this step adds results/analysis/
OUT = RESULTS / "analysis"
RNAERNIE = Path(os.environ.get("RNAERNIE_DATA", BASE / "mirna" / "RNAErnie" / "data"))
NRC_SNO = {"CD-box", "HACA-box", "scaRNA"}
NATIVE = {"RNAErnie", "ncRDense"}


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def read_fasta_seqs(path):
    seqs, cur = [], None
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith(">"):
            cur = []; seqs.append(cur)
        elif line and cur is not None:
            cur.append(line)
    return {"".join(s).upper().replace("T", "U") for s in seqs}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RESULTS / "snorna_predictions.csv")
    tab = pd.read_csv(DATA / "snorna_expressed_table.csv")[["id", "max_TPM"]]
    d = pred.merge(tab, on="id", how="left")
    models = [m for m in ["RNAErnie", "ncRDense", "MncR SeqEnc (sequence only)", "MncR full (sequence + graph)"] if m in d]
    ok = pd.DataFrame({m: (d[m].isin(NRC_SNO) if m in NATIVE else d[m] == "snoRNA") for m in models})
    n = len(d)
    short = {"MncR SeqEnc (sequence only)": "MncR SeqEnc", "MncR full (sequence + graph)": "MncR full"}
    ok.columns = [short.get(m, m) for m in ok.columns]
    models = list(ok.columns)
    pd.set_option("display.width", 220)

    # 1. recall + CI
    r1 = pd.DataFrame([{"model": m, "n": n, "called_snoRNA": int(ok[m].sum()), "recall": ok[m].mean(),
                        "ci_low": wilson(ok[m].sum(), n)[0], "ci_high": wilson(ok[m].sum(), n)[1]} for m in models])
    r1.round(4).to_csv(OUT / "1_recall_ci.csv", index=False)
    print("=== 1. snoRNA recall with 95% confidence interval ==="); print(r1.round(4).to_string(index=False))

    # 2. McNemar
    rows = []
    for a, b in itertools.combinations(models, 2):
        x, y = int((ok[a] & ~ok[b]).sum()), int((~ok[a] & ok[b]).sum())
        p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
        rows.append({"model_A": a, "model_B": b, "A_right_B_wrong": x, "B_right_A_wrong": y, "p_value": p})
    r2 = pd.DataFrame(rows)
    order = r2["p_value"].argsort().values; run = 0.0; holm = np.empty(len(r2))
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(r2) - rank) * r2["p_value"].iloc[i])); holm[i] = run
    r2["p_holm"] = holm; r2["real_gap"] = r2["p_holm"] < 0.05
    for col in ("p_value", "p_holm"):
        r2[col] = [float(f"{v:.3g}") for v in r2[col]]
    r2.to_csv(OUT / "2_mcnemar.csv", index=False)
    print("\n=== 2. McNemar exact test, every pair (Holm-corrected for 6 comparisons) ==="); print(r2.to_string(index=False))

    # 3-5. recall by group
    def by(group, title, fname):
        g = pd.concat([d[group], ok], axis=1).groupby(group, observed=True)
        t = g.agg(n=(models[0], "size"), **{m: (m, "mean") for m in models}).round(3)
        t.to_csv(OUT / fname); print(f"\n=== {title} (share called snoRNA) ==="); print(t.to_string()); return t
    by("expected_subtype", "3. By subtype (from gene name)", "3_by_subtype.csv")
    d["length_bin"] = pd.cut(d["length"], [0, 80, 100, 150, 10000], labels=["<=80 nt", "81-100", "101-150", ">150"])
    t4 = by("length_bin", "4. By length", "4_by_length.csv")
    d["tpm_bin"] = pd.qcut(d["max_TPM"], 4, duplicates="drop")
    d["tpm_bin"] = d["tpm_bin"].astype(str)
    by("tpm_bin", "5. By expression (max TPM, quartiles)", "5_by_expression.csv")

    # 6. agreement
    d["n_models_right"] = ok.sum(axis=1)
    r6 = d["n_models_right"].value_counts().sort_index()
    r6.to_csv(OUT / "6_agreement.csv", header=["sequences"])
    print(f"\n=== 6. How many of the {len(models)} models got each snoRNA right ==="); print(r6.to_string())
    miss = d[d["n_models_right"] == 0][["id", "symbol", "length", "max_TPM"] + [c for c in pred.columns if c in
            ("RNAErnie", "ncRDense", "MncR SeqEnc (sequence only)", "MncR full (sequence + graph)")]]
    miss.to_csv(OUT / "snorna_missed_by_all.csv", index=False)
    print(f"missed by ALL models: {len(miss)} (saved snorna_missed_by_all.csv)")
    if len(miss): print(miss.to_string(index=False))

    # 7. training overlap with nRC
    seqs_fa = {}
    for line in open(DATA / "snorna_expressed.fasta"):
        line = line.strip()
        if line.startswith(">"): cur = line[1:].split()[0]; seqs_fa[cur] = ""
        else: seqs_fa[cur] += line.upper().replace("T", "U")
    d["seq"] = d["id"].map(seqs_fa)
    tr_f, te_f = RNAERNIE / "nrc" / "dataset_Rfam_6320_13classes.fasta", RNAERNIE / "nrc" / "dataset_Rfam_validated_2600_13classes.fasta"
    if tr_f.exists() and te_f.exists():
        tr, te = read_fasta_seqs(tr_f), read_fasta_seqs(te_f)
        d["in_nrc_train"] = d["seq"].isin(tr); d["in_nrc_test"] = d["seq"].isin(te)
        d["seen_in_nrc"] = d["in_nrc_train"] | d["in_nrc_test"]
        print(f"\n=== 7. Exact sequence overlap with nRC: in training {int(d.in_nrc_train.sum())}, in test {int(d.in_nrc_test.sum())}, "
              f"either {int(d.seen_in_nrc.sum())} of {n} ===")
        t7 = pd.concat([d["seen_in_nrc"], ok], axis=1).groupby("seen_in_nrc").agg(
            n=(models[0], "size"), **{m: (m, "mean") for m in models}).round(3)
        t7.to_csv(OUT / "7_overlap_with_nrc.csv"); print("share called snoRNA, by whether the exact sequence is in nRC:"); print(t7.to_string())
        print("(exact matches only; near-identical sequences are not detected. MncR's own training set was not checked.)")
    else:
        print("\n7. skipped: nRC FASTA files not found in", RNAERNIE / "nrc")

    # 8. subtype confusion
    for m in ["RNAErnie", "ncRDense"]:
        if m in pred:
            k = d[d["expected_subtype"] != "unknown"]
            ct = pd.crosstab(k["expected_subtype"], pred.loc[k.index, m].where(pred.loc[k.index, m].isin(
                ["CD-box", "HACA-box", "scaRNA"]), "other (not snoRNA)"))
            ct.to_csv(OUT / f"8_subtype_confusion_{m}.csv")
            print(f"\n=== 8. {m}: expected subtype (rows) vs predicted (columns) ==="); print(ct.to_string())

    # figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    err = [r1.recall - r1.ci_low, r1.ci_high - r1.recall]
    ax[0].bar(r1.model, r1.recall, yerr=err, capsize=4, color="#2f6f9f")
    for i, v in enumerate(r1.recall): ax[0].text(i, min(v + 0.03, 1.0), f"{v:.3f}", ha="center", fontsize=8)
    ax[0].set_ylim(0.8, 1.04); ax[0].set_ylabel("Share called snoRNA"); ax[0].set_title("All expressed snoRNAs (95% CI)", fontsize=10)
    ax[0].tick_params(axis="x", labelsize=8)
    for m in models: ax[1].plot(t4.index.astype(str), t4[m], marker="o", label=m)
    ax[1].set_ylim(0.5, 1.02); ax[1].set_title("By snoRNA length", fontsize=10); ax[1].legend(fontsize=7, frameon=False)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "snorna_analysis_figure.png", dpi=200)
    print(f"\nsaved everything in {OUT}")


if __name__ == "__main__":
    main()
