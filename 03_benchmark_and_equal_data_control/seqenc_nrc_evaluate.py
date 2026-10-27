"""
SeqEnc-on-nRC - Step 2: evaluate the model trained in seqenc_nrc_train.py.

  A. 13-class result on the full nRC test set (same test set RNAErnie and ncRDense used).
  B. 4-class result on the SAME 1,125 Rfam benchmark sequences used for all other models
     (13 nRC classes mapped to miRNA / rRNA / snoRNA / tRNA; any other prediction counts as wrong).
  C. If the summary folder is set up, a McNemar test of this model against the other four.

Saves into mirna/seqenc_nrc/:  nrc_test_predictions.csv, rfam1125_predictions.csv,
                              seqenc_nrc_metrics.csv, seqenc_nrc_mcnemar.csv
Run:  python seqenc_nrc_evaluate.py
"""
import itertools
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support
from scipy.stats import binomtest

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
NCRDENSE = Path(os.environ.get("NCRDENSE_DATA", BASE / "mirna" / "ncrdense" / "data"))
MNCR = Path(os.environ.get("MNCR_DATA", BASE / "mirna" / "mncr" / "data"))
OUT = Path(os.environ.get("SEQENC_NRC_DIR", BASE / "mirna" / "seqenc_nrc"))
SUMMARY = Path(os.environ.get("SUMMARY_DIR", BASE / "mirna" / "summary"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seqenc_nrc_train import encode, MAXLEN            # same encoding as in training

FOUR = ["miRNA", "rRNA", "snoRNA", "tRNA"]
MAP13 = {"miRNA": "miRNA", "5S_rRNA": "rRNA", "5_8S_rRNA": "rRNA",
         "CD-box": "snoRNA", "HACA-box": "snoRNA", "scaRNA": "snoRNA", "tRNA": "tRNA"}
to4 = lambda lab: MAP13.get(lab, "other")
NAME = "SeqEnc on nRC (equal data)"


def main():
    classes = pd.read_csv(OUT / "classes.csv")["label"].tolist()
    model = tf.keras.models.load_model(OUT / "seqenc_nrc_best.keras")

    # ---------- A. 13-class nRC test set
    test = pd.read_csv(NCRDENSE / "nrc_test_structure.csv")
    proba = model.predict(encode(test["seq"].tolist()), batch_size=64, verbose=0)
    pred13 = pd.Series([classes[i] for i in proba.argmax(1)], index=test["id"].values)
    truth13 = pd.Series(test["label"].values, index=test["id"].values)
    p, r, f1, _ = precision_recall_fscore_support(truth13, pred13, average="macro", zero_division=0)
    print(f"=== A. nRC test set, 13 classes, n = {len(test)} ===")
    print(f"accuracy {accuracy_score(truth13, pred13):.4f} | macro F1 {f1:.4f} | MCC {matthews_corrcoef(truth13, pred13):.4f}")
    print("  (earlier: RNAErnie 0.9041 / 0.8992 / 0.8960;  ncRDense 0.8664 / 0.8596 / 0.8554)")
    pd.DataFrame({"id": pred13.index, "true": truth13.values, "pred": pred13.values}).to_csv(
        OUT / "nrc_test_predictions.csv", index=False)

    # ---------- B. 1,125 Rfam benchmark, 4 classes
    ids, labs = [], []
    for line in open(MNCR / "rfam_testset_used_for_ncrdense.fasta"):
        if line.startswith(">"):
            parts = line[1:].split(); ids.append(parts[0]); labs.append(parts[-1])
    bench = pd.Series(labs, index=ids)
    first = pred13[~pred13.index.duplicated()]
    ok = bench.index.isin(first.index)
    print(f"\n=== B. Rfam benchmark: matched {ok.sum()} / {len(bench)} sequences in the nRC test set ===")
    bench = bench[ok]
    pred4 = first[bench.index].map(to4)
    pd.DataFrame({"id": bench.index, "true": bench.values, "pred4": pred4.values}).to_csv(
        OUT / "rfam1125_predictions.csv", index=False)
    pr, rc, f1m, _ = precision_recall_fscore_support(bench, pred4, labels=FOUR, average="macro", zero_division=0)
    per = precision_recall_fscore_support(bench, pred4, labels=FOUR, zero_division=0)[2]
    row = {"n": len(bench), "accuracy": accuracy_score(bench, pred4), "macro_F1": f1m,
           "MCC": matthews_corrcoef(bench, pred4), **{f"F1_{c}": v for c, v in zip(FOUR, per)}}
    pd.DataFrame([row], index=[NAME]).round(4).to_csv(OUT / "seqenc_nrc_metrics.csv")
    pd.set_option("display.width", 200)
    print(pd.DataFrame([row], index=[NAME]).round(4).to_string())
    print("  (others: MncR full 0.9716 | MncR SeqEnc [authors, 60k seqs] 0.9476 | RNAErnie 0.9369 | ncRDense 0.9093)")

    # ---------- C. McNemar against the other four models
    try:
        sys.path.insert(0, str(SUMMARY))
        import summary_stats_and_figure as s
        truth, preds = s.get_predictions()
    except Exception as e:                                     # summary folder not set up
        print(f"\n(C skipped: could not load the other models' predictions: {e})")
        return
    mine = pd.Series(pred4.values, index=bench.index)
    common = [i for i in truth.index if i in mine.index]
    t = truth[common].values
    c_me = mine[common].values == t
    rows = []
    for k, v in preds.items():
        c_o = v[common].values == t
        a, b = int((c_me & ~c_o).sum()), int((~c_me & c_o).sum())
        pv = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
        rows.append({"other_model": k, "other_accuracy": c_o.mean(), "SeqEnc_nRC_right_other_wrong": a,
                     "other_right_SeqEnc_nRC_wrong": b, "p_value": float(f"{pv:.3g}"), "significant_0.05": pv < 0.05})
    m = pd.DataFrame(rows)
    m["p_holm"] = np.minimum(1, m["p_value"].rank(method="first").rsub(len(m) + 1) * m["p_value"]).round(5)
    m.to_csv(OUT / "seqenc_nrc_mcnemar.csv", index=False)
    print(f"\n=== C. McNemar: {NAME} (accuracy {c_me.mean():.4f}) vs each model, same {len(common)} sequences ===")
    print(m.to_string(index=False))
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
