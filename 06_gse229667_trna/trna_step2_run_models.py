"""
tRNA - Step 2: run RNAErnie, ncRDense and MncR on the tRNAs from Step 1.

Every sequence here is a tRNA, so "% called tRNA" is the RECALL for the tRNA class only (see the summary report).
RNAErnie / ncRDense: the nRC class "tRNA". MncR: its own "tRNA" class. Any other class counts as "not tRNA".

Run:   python trna_step2_run_models.py              (all models)
       MODELS=seqenc,ncrdense python trna_step2_run_models.py    (choose some: rnaernie, ncrdense, seqenc, mncr)
"mncr" (sequence + graph) needs the GraphProt conda environment from the MncR steps; if it is missing, that
model is skipped with a message and the others still run.

Output (trna/results/):
  trna_summary.csv            one row per model (the table printed at the end)
  trna_predictions.csv        every sequence, with the call of every model side by side
  predictions_<model>.csv     one file per model (id, amino acid, anticodon, length, its call, tRNA yes/no)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("TRNA_DIR", BASE / "trna"))
DATA = DIR / "data"            # inputs (made by step 1)
RESULTS = DIR / "results"      # everything this step produces
RNAERNIE = Path(os.environ.get("RNAERNIE_DATA", BASE / "mirna" / "RNAErnie" / "data"))
NCRDENSE = Path(os.environ.get("NCRDENSE_DATA", BASE / "mirna" / "ncrdense" / "data"))
MNCR_DIR = BASE / "mirna" / "mncr"
MODELS = [m.strip() for m in os.environ.get("MODELS", "rnaernie,ncrdense,seqenc,mncr").split(",")]

TARGET = "tRNA"
MAXLEN = 750


def read_fasta(path):
    ids, seqs = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            ids.append(line[1:].split()[0]); seqs.append("")
        elif line:
            seqs[-1] += line.upper()
    return ids, seqs


# ---------------------------------------------------------------- RNAErnie
def run_rnaernie(ids, seqs):
    import joblib, torch
    for cand in [RNAERNIE.parent / "cache", BASE / "cache"]:
        if (cand / "hf").exists():
            os.environ.setdefault("HF_HOME", str(cand / "hf")); break
    from multimolecule import RnaTokenizer, RnaErnieModel
    tok = RnaTokenizer.from_pretrained("multimolecule/rnaernie")
    model = RnaErnieModel.from_pretrained("multimolecule/rnaernie").eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            enc = tok(seqs[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=512)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            chunks.append(((h * m).sum(1) / m.sum(1)).numpy())
    X = np.vstack(chunks).astype("float32")
    clf = joblib.load(RNAERNIE / "nrc" / "rnaernie_nrc_classifier.joblib")
    return list(clf.predict(X))


# ---------------------------------------------------------------- ncRDense
def run_ncrdense(ids, seqs):
    import RNA
    import tensorflow as tf
    seq_lut = np.full(256, 4, dtype=np.int8)
    for i, ch in enumerate("AUGC"): seq_lut[ord(ch)] = i
    st_lut = np.full(256, 3, dtype=np.int8)
    for i, ch in enumerate(".()"): st_lut[ord(ch)] = i
    S = np.full((len(seqs), MAXLEN), 4, dtype=np.int8); T = np.full((len(seqs), MAXLEN), 3, dtype=np.int8)
    for k, s in enumerate(seqs):
        r = s.upper().replace("T", "U")[:MAXLEN]
        struct, _ = RNA.fold(r)
        S[k, :len(r)] = seq_lut[np.frombuffer(r.encode(), dtype=np.uint8)]
        T[k, :len(r)] = st_lut[np.frombuffer(struct.encode(), dtype=np.uint8)]
    seq_oh = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype="float32")
    st_oh = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype="float32")
    classes = pd.read_csv(NCRDENSE / "nrc_classes.csv")["label"].tolist()
    model = tf.keras.models.load_model(NCRDENSE / "ncrdense_best.keras")
    proba = model.predict([seq_oh[S], st_oh[T]], batch_size=64, verbose=0)
    return [classes[i] for i in proba.argmax(1)]


# ---------------------------------------------------------------- MncR
def run_mncr(ids, seqs, graph):
    sys.path.insert(0, str(MNCR_DIR))
    import mncr_common as c
    if not graph:
        p = c.predict(c.load_model("seqenc_fold8"), seqs)
        return [c.CLASSES[i] for i in p.argmax(1)]
    import mncr_step4_graphprot_features as g4
    feat = g4.make_features(ids, seqs, c.DATA / "graphprot_tmp", "trna")
    lines = c.read_feature_lines(feat)
    p = c.predict(c.load_model("mncr_fold7"), seqs, lines)
    return [c.CLASSES[i] for i in p.argmax(1)]


def main():
    ids, seqs = read_fasta(DATA / "trna.fasta")
    tab = pd.read_csv(DATA / "trna_table.csv").set_index("id").reindex(ids)
    print(f"{len(ids)} tRNAs, length {min(map(len, seqs))}-{max(map(len, seqs))} nt")
    out = pd.DataFrame({"id": ids, "amino_acid": tab["amino_acid"].values, "anticodon": tab["anticodon"].values,
                        "length": [len(s) for s in seqs]})
    runners = {"rnaernie": ("RNAErnie", lambda: run_rnaernie(ids, seqs)),
               "ncrdense": ("ncRDense", lambda: run_ncrdense(ids, seqs)),
               "seqenc": ("MncR SeqEnc (sequence only)", lambda: run_mncr(ids, seqs, False)),
               "mncr": ("MncR full (sequence + graph)", lambda: run_mncr(ids, seqs, True))}
    rows, done = [], []
    for key in MODELS:
        name, fn = runners[key]
        print(f"\n--- {name} ---", flush=True)
        try:
            pred = fn()
        except (Exception, SystemExit) as e:                     # one model failing must not stop the others
            print(f"SKIPPED {name}: {e}"); continue
        out[name] = pred; done.append(name)
        s = pd.Series(pred)
        called = (s == TARGET).mean()
        rows.append({"model": name, "n": len(pred), "called_tRNA": round(called, 4),
                     "most_common_other_calls": "; ".join(f"{k} {v}" for k, v in s[s != TARGET].value_counts().head(4).items())})
        print(f"called tRNA: {called:.1%}   ({int(round(called * len(pred)))} of {len(pred)})")
        print(s.value_counts().to_string())
    summ = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "trna_predictions.csv", index=False); summ.to_csv(RESULTS / "trna_summary.csv", index=False)
    for name in done:                                             # one simple file per model
        one = out[["id", "amino_acid", "anticodon", "length"]].copy()
        one["model_call"] = out[name]; one["called_tRNA"] = (out[name] == TARGET).map({True: "yes", False: "no"})
        one.to_csv(RESULTS / f"predictions_{name.split(' (')[0].replace(' ', '_')}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print("\n=== Summary: share of human tRNAs called tRNA (this is recall, not accuracy) ===")
    print(summ.to_string(index=False))
    print(f"\nAll results are in {RESULTS}")


if __name__ == "__main__":
    main()
