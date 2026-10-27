"""
snoRNA - Step 2: run RNAErnie, ncRDense and MncR on the snoRNAs from Step 1.

Every sequence here is a snoRNA, so "% called snoRNA" is the RECALL for the snoRNA class only (see the summary
report). Mapping used (same as for the Rfam benchmark): RNAErnie / ncRDense CD-box, HACA-box and scaRNA -> snoRNA;
MncR's own snoRNA class. Any other class counts as "not snoRNA".
Extra for RNAErnie / ncRDense: the gene name tells the subtype (SNORD = CD-box, SNORA = HACA-box, SCARNA = scaRNA),
so we also report how often they give exactly the right subtype.

Run:   python snorna_step2_run_models.py              (all models)
       MODELS=seqenc,ncrdense python snorna_step2_run_models.py    (choose some: rnaernie, ncrdense, seqenc, mncr)
"mncr" (sequence + graph) needs the GraphProt conda environment from the MncR steps; if it is missing, that
model is skipped with a message and the others still run.

Output (snorna/results/):
  snorna_summary.csv            one row per model (the table printed at the end)
  snorna_predictions.csv        every sequence, with the call of every model side by side
  predictions_<model>.csv       one file per model (id, symbol, length, expected subtype, its call, snoRNA yes/no)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("SNORNA_DIR", BASE / "snorna"))
DATA = DIR / "data"            # inputs (made by step 1)
RESULTS = DIR / "results"      # everything this step produces
RNAERNIE = Path(os.environ.get("RNAERNIE_DATA", BASE / "mirna" / "RNAErnie" / "data"))
NCRDENSE = Path(os.environ.get("NCRDENSE_DATA", BASE / "mirna" / "ncrdense" / "data"))
MNCR_DIR = BASE / "mirna" / "mncr"
MODELS = [m.strip() for m in os.environ.get("MODELS", "rnaernie,ncrdense,seqenc,mncr").split(",")]

NRC_TO_SNO = {"CD-box": "snoRNA", "HACA-box": "snoRNA", "scaRNA": "snoRNA"}
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
    feat = g4.make_features(ids, seqs, c.DATA / "graphprot_tmp", "snorna_expressed")
    lines = c.read_feature_lines(feat)
    p = c.predict(c.load_model("mncr_fold7"), seqs, lines)
    return [c.CLASSES[i] for i in p.argmax(1)]


def main():
    ids, seqs = read_fasta(DATA / "snorna_expressed.fasta")
    tab = pd.read_csv(DATA / "snorna_expressed_table.csv").set_index("id").reindex(ids)
    print(f"{len(ids)} snoRNAs, length {min(map(len, seqs))}-{max(map(len, seqs))} nt")
    out = pd.DataFrame({"id": ids, "symbol": tab["symbol"].values, "expected_subtype": tab["expected_subtype"].values,
                        "length": [len(s) for s in seqs]})
    runners = {"rnaernie": ("RNAErnie", lambda: run_rnaernie(ids, seqs)),
               "ncrdense": ("ncRDense", lambda: run_ncrdense(ids, seqs)),
               "seqenc": ("MncR SeqEnc (sequence only)", lambda: run_mncr(ids, seqs, False)),
               "mncr": ("MncR full (sequence + graph)", lambda: run_mncr(ids, seqs, True))}
    rows, native = [], {}
    for key in MODELS:
        name, fn = runners[key]
        print(f"\n--- {name} ---", flush=True)
        try:
            pred = fn()
        except (Exception, SystemExit) as e:                     # one model failing must not stop the others
            print(f"SKIPPED {name}: {e}"); continue
        out[name] = pred
        native[name] = key in ("rnaernie", "ncrdense")
        mapped = pd.Series([NRC_TO_SNO.get(p, p) for p in pred]) if native[name] else pd.Series(pred)
        called = (mapped == "snoRNA").mean()
        row = {"model": name, "n": len(pred), "called_snoRNA": round(called, 4),
               "most_common_other_calls": "; ".join(f"{k} {v}" for k, v in mapped[mapped != "snoRNA"].value_counts().head(4).items())}
        if native[name]:
            known = out["expected_subtype"] != "unknown"
            row["exact_subtype_correct"] = round(float((out.loc[known, name] == out.loc[known, "expected_subtype"]).mean()), 4)
        rows.append(row)
        print(f"called snoRNA: {called:.1%}   ({int(round(called * len(pred)))} of {len(pred)})")
        print(mapped.value_counts().to_string())
    summ = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "snorna_predictions.csv", index=False); summ.to_csv(RESULTS / "snorna_summary.csv", index=False)
    for name in native:                                           # one simple file per model
        mapped = out[name].map(lambda p: NRC_TO_SNO.get(p, p)) if native[name] else out[name]
        one = out[["id", "symbol", "expected_subtype", "length"]].copy()
        one["model_call"] = out[name]; one["called_snoRNA"] = (mapped == "snoRNA").map({True: "yes", False: "no"})
        safe = name.split(" (")[0].replace(" ", "_")
        one.to_csv(RESULTS / f"predictions_{safe}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print("\n=== Summary: share of GSE209924 snoRNAs called snoRNA (this is recall, not accuracy) ===")
    print(summ.to_string(index=False))
    print(f"\nAll results are in {RESULTS}")


if __name__ == "__main__":
    main()
