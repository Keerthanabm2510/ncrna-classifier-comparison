"""
Single-experiment test (GSE209924, TGIRT-seq) - Step 2: run RNAErnie, ncRDense and MncR on the sequences from Step 1.

Each sequence has a known type (miRNA, snoRNA, tRNA, rRNA, snRNA or lncRNA, from Ensembl / GtRNAdb). Calls are brought to common names:
  nRC models (RNAErnie, ncRDense): miRNA -> miRNA; CD-box, HACA-box, scaRNA -> snoRNA; 5S/5.8S rRNA -> rRNA;
     tRNA -> tRNA; any other class stays as it is. They have NO class for snRNA or lncRNA, so those are always wrong for them.
  MncR (6 classes: lncRNA, miRNA, rRNA, snRNA, snoRNA, tRNA): its own call.
A call is right if it equals the true type.

Run:   python gse209924_step2_run_models.py              (all models)
       MODELS=seqenc,ncrdense python gse209924_step2_run_models.py    (choose some: rnaernie, ncrdense, seqenc, mncr)
Output (gse209924/results/): pool_predictions.csv (raw calls), pool_summary.csv, predictions_<model>.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("GSE209924_DIR", BASE / "gse209924"))
DATA = DIR / "data"            # inputs (made by step 1)
RESULTS = DIR / "results"      # everything this step produces
RNAERNIE = Path(os.environ.get("RNAERNIE_DATA", BASE / "mirna" / "RNAErnie" / "data"))
NCRDENSE = Path(os.environ.get("NCRDENSE_DATA", BASE / "mirna" / "ncrdense" / "data"))
MNCR_DIR = BASE / "mirna" / "mncr"
MODELS = [m.strip() for m in os.environ.get("MODELS", "rnaernie,ncrdense,seqenc,mncr").split(",")]

MAP = {"CD-box": "snoRNA", "HACA-box": "snoRNA", "scaRNA": "snoRNA", "5S_rRNA": "rRNA", "5_8S_rRNA": "rRNA"}
norm = lambda x: MAP.get(x, x)
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
    import hashlib
    import json
    import mncr_step4_graphprot_features as g4
    # The graph-feature program (EDeN) was killed for lack of memory when given all sequences at once, so the
    # sequences are processed in chunks of at most MAX_NT nucleotides (env MAX_NT, default 30000; lower it if still killed).
    max_nt = int(os.environ.get("MAX_NT", 30000))
    chunks, cur, nt = [], [], 0
    for i, s in enumerate(seqs):
        if cur and nt + len(s) > max_nt: chunks.append(cur); cur, nt = [], 0
        cur.append(i); nt += len(s)
    if cur: chunks.append(cur)
    cache = RESULTS / "mncr_chunks"; cache.mkdir(parents=True, exist_ok=True)
    print(f"graph features in {len(chunks)} chunks of at most {max_nt} nt (finished chunks are saved: if you stop, run again and it continues)", flush=True)
    lines = []
    for k, idx in enumerate(chunks):
        tag = hashlib.md5(" ".join(ids[i] for i in idx).encode()).hexdigest()[:8]
        cf = cache / f"chunk_{k:03d}_{len(idx)}_{tag}.json"
        part = None
        if cf.exists():
            try: part = json.load(open(cf))
            except Exception: part = None
        if part is None:
            print(f"  chunk {k + 1}/{len(chunks)}: {len(idx)} sequences, {sum(len(seqs[i]) for i in idx)} nt", flush=True)
            feat = g4.make_features([ids[i] for i in idx], [seqs[i] for i in idx], c.DATA / "graphprot_tmp", f"pool_c{k:03d}")
            part = list(c.read_feature_lines(feat))
            if len(part) != len(idx):
                raise RuntimeError(f"chunk {k + 1} gave {len(part)} feature lines for {len(idx)} sequences (EDeN probably ran out of memory); "
                                   f"run again with a smaller chunk, e.g.  MAX_NT=10000 MODELS=mncr python {Path(__file__).name}")
            try: json.dump(part, open(cf, "w"))
            except Exception: pass
        lines.extend(part)
    p = c.predict(c.load_model("mncr_fold7"), seqs, lines)
    return [c.CLASSES[i] for i in p.argmax(1)]


def main():
    ids, seqs = read_fasta(DATA / "pool.fasta")
    tab = pd.read_csv(DATA / "pool_table.csv").set_index("id").reindex(ids)
    print(f"{len(ids)} sequences, length {min(map(len, seqs))}-{max(map(len, seqs))} nt; {tab['true'].value_counts().to_dict()}")
    out = pd.DataFrame({"id": ids, "true": tab["true"].values, "gene": tab["gene"].values, "length": [len(s) for s in seqs]})
    runners = {"rnaernie": ("RNAErnie", lambda: run_rnaernie(ids, seqs)),
               "ncrdense": ("ncRDense", lambda: run_ncrdense(ids, seqs)),
               "seqenc": ("MncR SeqEnc (sequence only)", lambda: run_mncr(ids, seqs, False)),
               "mncr": ("MncR full (sequence + graph)", lambda: run_mncr(ids, seqs, True))}
    done = []
    for key in MODELS:
        name, fn = runners[key]
        print(f"\n--- {name} ---", flush=True)
        try:
            pred = fn()
        except (Exception, SystemExit) as e:                     # one model failing must not stop the others
            print(f"SKIPPED {name}: {e}"); continue
        out[name] = pred; done.append(name)
    RESULTS.mkdir(parents=True, exist_ok=True)
    old_path = RESULTS / "pool_predictions.csv"
    if old_path.exists():                                         # keep the calls of models that were not re-run this time
        old = pd.read_csv(old_path)
        if list(old["id"]) == list(out["id"]):
            for c in old.columns:
                if c not in out.columns:
                    out[c] = old[c].values; print(f"kept the earlier results of {c}")
        else:
            print("NOTE: the sequence list changed since the earlier run, so earlier model results were NOT kept.")
    model_cols = [m for m in ["RNAErnie", "ncRDense", "MncR SeqEnc (sequence only)", "MncR full (sequence + graph)"] if m in out.columns]
    rows = []
    for name in model_cols:
        ok = out[name].map(norm).values == out["true"].values
        row = {"model": name, "n": len(out), "accuracy": round(ok.mean(), 4)}
        for t in ["miRNA", "snoRNA", "tRNA", "rRNA", "snRNA", "lncRNA"]:
            m = out["true"].values == t
            row[f"recall_{t}"] = round(ok[m].mean(), 4) if m.any() else float("nan")
        rows.append(row)
        if name in done: print(f"{name}: accuracy {ok.mean():.1%}   " + "  ".join(f"{k[7:]} {v:.1%}" for k, v in row.items() if k.startswith("recall_")))
    summ = pd.DataFrame(rows)
    out.to_csv(RESULTS / "pool_predictions.csv", index=False); summ.to_csv(RESULTS / "pool_summary.csv", index=False)
    for name in model_cols:
        one = out[["id", "true", "gene", "length"]].copy(); one["model_call"] = out[name]
        one["right"] = (out[name].map(norm) == out["true"]).map({True: "yes", False: "no"})
        one.to_csv(RESULTS / f"predictions_{name.split(' (')[0].replace(' ', '_')}.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Summary: share right overall and per true RNA type (models with no class for a type score 0 on it) ===")
    print(summ.to_string(index=False))
    print(f"\nAll results are in {RESULTS}")


if __name__ == "__main__":
    main()
