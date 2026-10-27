"""
Step 8: embed the nRC train/test sequences with RNAErnie.

- reads the two nRC FASTA files (class label = last word of each header)
- converts T->U, replaces any non-A/C/G/U letter with N
- removes test sequences that also appear in the training set (avoids inflated scores)
- saves 768-number embeddings + a small metadata table (data/embeddings/, data/nrc/)

Resumable: if an embedding file already exists it is skipped, so you can stop and rerun.
Run: python step8_embed_nrc.py      (about 8,500 sequences; expect roughly 15-40 min on CPU)
"""
import os
import re
from pathlib import Path

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DATA, CACHE = BASE / "data", BASE / "cache"
NRC, EMB = DATA / "nrc", DATA / "embeddings"
os.environ.setdefault("HF_HOME", str(CACHE / "hf"))    # reuse the weights from step 6

import numpy as np
import pandas as pd

TRAIN_FA = NRC / "dataset_Rfam_6320_13classes.fasta"
TEST_FA = NRC / "dataset_Rfam_validated_2600_13classes.fasta"
MODEL_NAME = "multimolecule/rnaernie"
BATCH, MAXLEN = 16, 512


def read_labeled_fasta(path):
    ids, labels, seqs = [], [], []
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            parts = line[1:].split()
            ids.append(parts[0])
            labels.append(parts[-1] if len(parts) > 1 else "unknown")
            seqs.append("")
        else:
            seqs[-1] += line          # handles wrapped sequences
    return ids, labels, seqs


def clean(seq):
    return re.sub(r"[^ACGU]", "N", seq.upper().replace("T", "U"))


def embed(seqs, tok, model, device):
    """Mean-pooled embedding per sequence. Sorted by length so batches waste little padding."""
    import torch
    order = np.argsort([len(s) for s in seqs])
    out = None
    with torch.no_grad():
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = tok([seqs[j] for j in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXLEN).to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            vec = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
            if out is None:
                out = np.zeros((len(seqs), vec.shape[1]), dtype="float32")
            out[idx] = vec
            if (i // BATCH) % 25 == 0:
                print(f"    {min(i + BATCH, len(seqs))}/{len(seqs)}", flush=True)
    return out


def load_and_prepare():
    tr_ids, tr_lab, tr_seq = read_labeled_fasta(TRAIN_FA)
    te_ids, te_lab, te_seq = read_labeled_fasta(TEST_FA)
    tr = pd.DataFrame({"id": tr_ids, "label": tr_lab, "seq": [clean(s) for s in tr_seq]})
    te = pd.DataFrame({"id": te_ids, "label": te_lab, "seq": [clean(s) for s in te_seq]})
    print(f"train: {len(tr)} seqs, {tr.label.nunique()} classes | test: {len(te)} seqs, {te.label.nunique()} classes")
    print("train class counts:\n" + tr.label.value_counts().to_string())
    dup = te.seq.isin(set(tr.seq))
    print(f"\ntest sequences identical to a training sequence: {dup.sum()} -> removed from test")
    te = te[~dup].reset_index(drop=True)
    for d in (tr, te):
        d["length"] = d.seq.str.len()
    print(f"length (nt) train {tr.length.min()}-{tr.length.max()}, test {te.length.min()}-{te.length.max()}; "
          f">512 nt (truncated): train {(tr.length > 512).sum()}, test {(te.length > 512).sum()}")
    print(f"sequences containing N: train {tr.seq.str.contains('N').sum()}, test {te.seq.str.contains('N').sum()}")
    return tr, te


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    tr, te = load_and_prepare()
    tr.drop(columns="seq").to_csv(NRC / "nrc_train_meta.csv", index=False)
    te.drop(columns="seq").to_csv(NRC / "nrc_test_meta.csv", index=False)

    import torch
    from multimolecule import RnaTokenizer, RnaErnieModel
    tok = RnaTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RnaErnieModel.from_pretrained(MODEL_NAME).eval().to(device)
    print(f"\nModel loaded on {device}")

    for name, df in [("nrc_test", te), ("nrc_train", tr)]:
        path = EMB / f"{name}_rnaernie.npy"
        if path.exists():
            print(f"{name}: already done, skipping"); continue
        print(f"{name}: embedding {len(df)} sequences ...")
        np.save(path, embed(df.seq.tolist(), tok, model, device))
        print(f"  saved {path}")
    print("\nStep 8 done.")


if __name__ == "__main__":
    main()
