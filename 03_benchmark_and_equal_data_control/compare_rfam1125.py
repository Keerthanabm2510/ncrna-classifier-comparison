"""
Same-subset comparison: score RNAErnie, ncRDense and MncR on the SAME 1,125 Rfam sequences
that the MncR paper used (4 classes: miRNA, rRNA, snoRNA, tRNA).

Why: the three models report on different test sets / class sets. This puts them on one footing.
How:
  * The 1,125 sequences come from the nRC test set, so RNAErnie and ncRDense (trained on nRC) can score them.
  * Their 13 classes are mapped to the 4 shared ones:
        miRNA -> miRNA | 5S_rRNA, 5_8S_rRNA -> rRNA | CD-box, HACA-box, scaRNA -> snoRNA | tRNA -> tRNA
    Any other prediction (IRES, Intron_gpI/II, leader, riboswitch, ribozyme) counts as wrong, as in the MncR paper.
  * MncR uses the 6-class pretrained models (sequence-only SeqEnc; full MncR too, once Step 4/5 have produced
    its predictions for these sequences).

Needs: nothing new (tensorflow, scikit-learn, joblib, pandas are already in the mini_project environment).
Run (from any folder):  python compare_rfam1125.py
"""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, matthews_corrcoef, precision_recall_fscore_support)

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
MNCR = Path(os.environ.get("MNCR_DATA", BASE / "mirna" / "mncr" / "data"))
NCRDENSE = Path(os.environ.get("NCRDENSE_DATA", BASE / "mirna" / "ncrdense" / "data"))
_r = BASE / "mirna" / "RNAErnie" / "data"
RNAERNIE = Path(os.environ.get("RNAERNIE_DATA", _r if _r.exists() else BASE / "data"))
OUT = Path(os.environ.get("COMPARE_OUT", BASE / "mirna" / "comparison"))

FOUR = ["miRNA", "rRNA", "snoRNA", "tRNA"]
MAP13 = {"miRNA": "miRNA", "5S_rRNA": "rRNA", "5_8S_rRNA": "rRNA",
         "CD-box": "snoRNA", "HACA-box": "snoRNA", "scaRNA": "snoRNA", "tRNA": "tRNA"}
to4 = lambda lab: MAP13.get(lab, "other")

SEQ_ONEHOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype="float32")
STRUCT_ONEHOT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype="float32")


def read_benchmark():
    ids, labels = [], []
    for line in open(MNCR / "rfam_testset_used_for_ncrdense.fasta"):
        if line.startswith(">"):
            parts = line[1:].split()
            ids.append(parts[0]); labels.append(parts[-1])
    return pd.DataFrame({"id": ids, "true": labels})


def rnaernie_predictions(bench):
    meta = pd.read_csv(RNAERNIE / "nrc" / "nrc_test_meta.csv")
    X = np.load(RNAERNIE / "embeddings" / "nrc_test_rnaernie.npy")
    assert len(meta) == len(X), "RNAErnie embeddings and metadata differ in length"
    pos = {i: k for k, i in reversed(list(enumerate(meta["id"])))}       # first occurrence of each id
    rows = [pos.get(i) for i in bench["id"]]
    ok = np.array([r is not None for r in rows])
    clf = joblib.load(RNAERNIE / "nrc" / "rnaernie_nrc_classifier.joblib")
    pred13 = pd.Series(index=bench["id"], dtype=object)
    sel = [r for r in rows if r is not None]
    pred13[bench["id"][ok].values] = clf.predict(X[sel])
    truth13 = pd.Series(index=bench["id"], dtype=object)
    truth13[bench["id"][ok].values] = meta["label"].values[sel]
    return pred13, truth13, ok


def ncrdense_predictions(bench):
    import tensorflow as tf
    meta = pd.read_csv(NCRDENSE / "nrc_test_encoded_meta.csv")
    S = np.load(NCRDENSE / "nrc_test_seq_codes.npy"); T = np.load(NCRDENSE / "nrc_test_struct_codes.npy")
    classes = pd.read_csv(NCRDENSE / "nrc_classes.csv")["label"].tolist()
    pos = {i: k for k, i in reversed(list(enumerate(meta["id"])))}
    rows = [pos.get(i) for i in bench["id"]]
    ok = np.array([r is not None for r in rows])
    sel = np.array([r for r in rows if r is not None])
    model = tf.keras.models.load_model(NCRDENSE / "ncrdense_best.keras")
    proba = model.predict([SEQ_ONEHOT[S[sel]], STRUCT_ONEHOT[T[sel]]], batch_size=128, verbose=0)
    pred13 = pd.Series(index=bench["id"], dtype=object)
    pred13[bench["id"][ok].values] = [classes[i] for i in proba.argmax(1)]
    truth13 = pd.Series(index=bench["id"], dtype=object)
    truth13[bench["id"][ok].values] = meta["label"].values[sel]
    return pred13, truth13, ok


def read_mncr(path, col):
    if not path.exists():
        return None
    d = pd.read_csv(path)
    return pd.Series(d[col].values, index=d["id"].values)


def metrics(truth, pred):
    p, r, f1, _ = precision_recall_fscore_support(truth, pred, labels=FOUR, average="macro", zero_division=0)
    per = precision_recall_fscore_support(truth, pred, labels=FOUR, zero_division=0)[2]
    return {"n": len(truth), "accuracy": accuracy_score(truth, pred), "macro_precision": p, "macro_recall": r,
            "macro_F1": f1, "MCC": matthews_corrcoef(truth, pred), **{f"F1_{c}": v for c, v in zip(FOUR, per)}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bench = read_benchmark()
    print(f"benchmark: {len(bench)} sequences -> {bench.true.value_counts().to_dict()}")
    preds = {}

    p13, t13, ok = rnaernie_predictions(bench)
    print(f"RNAErnie: matched {ok.sum()}/{len(bench)} benchmark ids in its nRC test set")
    agree = (t13.dropna().map(to4) == bench.set_index("id").true[t13.dropna().index]).mean()
    print(f"  label check (nRC label mapped to 4 classes == benchmark label): {agree:.1%}")
    preds["RNAErnie (frozen + LR)"] = p13.map(lambda x: to4(x) if isinstance(x, str) else None)

    p13, t13, ok = ncrdense_predictions(bench)
    print(f"ncRDense: matched {ok.sum()}/{len(bench)} benchmark ids in its nRC test set")
    agree = (t13.dropna().map(to4) == bench.set_index("id").true[t13.dropna().index]).mean()
    print(f"  label check: {agree:.1%}")
    preds["ncRDense (light, re-built)"] = p13.map(lambda x: to4(x) if isinstance(x, str) else None)

    s = read_mncr(MNCR / "rfam_benchmark_seqenc_predictions.csv", "predicted_class")
    if s is not None:
        preds["MncR SeqEnc (sequence only)"] = s
    for tag, fn in [("rfam_benchmark_mncr_predictions.csv", "full"), ("rfam_benchmark_subset_mncr_predictions.csv", "subset")]:
        f = read_mncr(MNCR / tag, "mncr_class")
        if f is not None:
            preds["MncR full (sequence + graph)" + (" [subset]" if fn == "subset" else "")] = f

    truth = bench.set_index("id")["true"]
    rows, ids_by_model = {}, {}
    for name, p in preds.items():
        p = p.dropna()
        p = p[p.index.isin(truth.index)]
        ids_by_model[name] = set(p.index)
        rows[name] = metrics(truth[p.index].values, p.values)
    table = pd.DataFrame(rows).T
    cols = ["n", "accuracy", "macro_precision", "macro_recall", "macro_F1", "MCC"] + [f"F1_{c}" for c in FOUR]
    pd.set_option("display.width", 200)
    print("\n=== Each model on every benchmark sequence it could score ===")
    print(table[cols].round(4).to_string())
    table[cols].round(4).to_csv(OUT / "comparison_rfam1125.csv")

    common = set.intersection(*ids_by_model.values())
    if len(common) < len(bench):
        rows2 = {n: metrics(truth[sorted(common)].values, preds[n].dropna()[sorted(common)].values) for n in preds}
        t2 = pd.DataFrame(rows2).T[cols]
        print(f"\n=== All models on the same {len(common)} sequences (the subset every model covers) ===")
        print(t2.round(4).to_string())
        t2.round(4).to_csv(OUT / "comparison_rfam_common_subset.csv")
    print(f"\nsaved to {OUT}")
    print("Note: 4-class scores are easier than the 13-class scores reported earlier, so do not compare "
          "them with 0.904 / 0.866.")


if __name__ == "__main__":
    main()
