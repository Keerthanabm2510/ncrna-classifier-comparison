"""
Four-model summary - Step A: significance tests + one figure.

What it does (all on the SAME 1,125 Rfam benchmark sequences, 4 classes):
  1. Gets each model's prediction for every sequence (re-uses compare_rfam1125.py).
  2. Accuracy with a 95% confidence interval (bootstrap, 5,000 resamples).
  3. McNemar exact test for every pair of models: "do they make different mistakes
     more often than chance would explain?"  (p < 0.05 -> the accuracy gap is real)
  4. Saves a bar chart (accuracy with error bars + macro-F1) and two CSV tables.

Setup (once):
    cd ~/Documents/mini_project/2_project/mirna
    mkdir -p summary
    cp mncr/compare_rfam1125.py summary/        # this script imports it
    # put this file in mirna/summary/ as well

Run:
    conda activate mini_project
    cd ~/Documents/mini_project/2_project/mirna/summary
    python summary_stats_and_figure.py

Needs: numpy, pandas, scipy, matplotlib, scikit-learn (all normally present in mini_project;
if matplotlib is missing:  pip install matplotlib)
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_rfam1125 as cmp                      # noqa: E402  (same folder)

OUT = HERE
FOUR = cmp.FOUR
N_BOOT = 5000
RNG = np.random.default_rng(0)


def get_predictions():
    """Returns truth (Series by id) and {model name: Series of 4-class predictions by id}."""
    bench = cmp.read_benchmark()
    truth = bench.set_index("id")["true"]
    preds = {}

    p13, _, _ = cmp.rnaernie_predictions(bench)
    preds["RNAErnie"] = p13.map(lambda x: cmp.to4(x) if isinstance(x, str) else None)
    p13, _, _ = cmp.ncrdense_predictions(bench)
    preds["ncRDense (light)"] = p13.map(lambda x: cmp.to4(x) if isinstance(x, str) else None)

    s = cmp.read_mncr(cmp.MNCR / "rfam_benchmark_seqenc_predictions.csv", "predicted_class")
    f = cmp.read_mncr(cmp.MNCR / "rfam_benchmark_mncr_predictions.csv", "mncr_class")
    if s is None or f is None:
        sys.exit("MncR prediction files not found in " + str(cmp.MNCR))
    preds["MncR SeqEnc"] = s
    preds["MncR full"] = f

    common = sorted(set.intersection(*[set(p.dropna().index) for p in preds.values()]) & set(truth.index))
    truth = truth[common]
    preds = {k: v[common] for k, v in preds.items()}
    return truth, preds


def bootstrap_ci(correct):
    n = len(correct)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    accs = correct[idx].mean(axis=1)
    return np.percentile(accs, [2.5, 97.5])


def main():
    truth, preds = get_predictions()
    n = len(truth)
    print(f"{n} sequences scored by all four models")

    correct = {k: (v.values == truth.values) for k, v in preds.items()}

    # ---- accuracy + CI + macro F1
    rows = []
    for k in preds:
        lo, hi = bootstrap_ci(correct[k])
        rows.append({"model": k, "n": n, "correct": int(correct[k].sum()),
                     "accuracy": correct[k].mean(), "ci_low": lo, "ci_high": hi,
                     "macro_F1": f1_score(truth.values, preds[k].values, labels=FOUR,
                                          average="macro", zero_division=0)})
    acc = pd.DataFrame(rows)
    acc.round(4).to_csv(OUT / "summary_accuracy_ci.csv", index=False)
    print("\n=== Accuracy with 95% confidence interval ===")
    print(acc.round(4).to_string(index=False))

    # ---- McNemar exact, all pairs
    prow = []
    for a, b in itertools.combinations(preds, 2):
        only_a = int((correct[a] & ~correct[b]).sum())     # a right, b wrong
        only_b = int((~correct[a] & correct[b]).sum())     # b right, a wrong
        disc = only_a + only_b
        p = binomtest(only_a, disc, 0.5).pvalue if disc > 0 else 1.0
        prow.append({"model_A": a, "model_B": b, "A_right_B_wrong": only_a,
                     "B_right_A_wrong": only_b, "p_value": p,
                     "significant_0.05": p < 0.05})
    mc = pd.DataFrame(prow)
    # Holm correction for the 6 comparisons
    order = mc["p_value"].argsort().values
    holm = np.empty(len(mc))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(mc) - rank) * mc["p_value"].iloc[i]))
        holm[i] = running
    mc["p_holm"] = holm
    mc["significant_after_holm"] = mc["p_holm"] < 0.05
    mc["p_value"] = mc["p_value"].map(lambda x: float(f"{x:.3g}"))
    mc["p_holm"] = mc["p_holm"].map(lambda x: float(f"{x:.3g}"))
    mc.to_csv(OUT / "summary_mcnemar.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== McNemar exact test, every pair (Holm-corrected for 6 comparisons) ===")
    print(mc.to_string(index=False))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(acc)); w = 0.38
    err = [acc["accuracy"] - acc["ci_low"], acc["ci_high"] - acc["accuracy"]]
    b1 = ax.bar(x - w / 2, acc["accuracy"], w, yerr=err, capsize=4, label="Accuracy (95% CI)", color="#2f6f9f")
    b2 = ax.bar(x + w / 2, acc["macro_F1"], w, label="Macro F1", color="#e08a2c")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, f"{r.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(acc["model"], fontsize=9)
    ax.set_ylim(0.8, 1.02); ax.set_ylabel("Score")
    ax.set_title(f"Four classes (miRNA, rRNA, snoRNA, tRNA), same {n:,} Rfam sequences", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "summary_accuracy_figure.png", dpi=200)
    print(f"\nsaved in {OUT}:\n  summary_accuracy_ci.csv\n  summary_mcnemar.csv\n  summary_accuracy_figure.png")


if __name__ == "__main__":
    main()
