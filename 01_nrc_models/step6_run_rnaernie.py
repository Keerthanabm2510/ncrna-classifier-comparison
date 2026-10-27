"""
Step 6: run pretrained RNAErnie on your FASTA files and save sequence embeddings.

Uses the PyTorch port of RNAErnie (package `multimolecule`), so PaddlePaddle is NOT needed.
Weights (~420 MB) download once from Hugging Face; we point the cache at your
project's cache/ folder so it is easy to delete later.

Run:  python3 step6_run_rnaernie.py
"""
import os
from pathlib import Path

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DATA, CACHE = BASE / "data", BASE / "cache"
os.environ.setdefault("HF_HOME", str(CACHE / "hf"))   # keep model weights inside cache/

import numpy as np
import pandas as pd
import torch
from multimolecule import RnaTokenizer, RnaErnieModel

MODEL_NAME = "multimolecule/rnaernie"
BATCH = 32
OUT = DATA / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)


def read_fasta(path):
    ids, seqs = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            ids.append(line[1:]); seqs.append("")
        elif line:
            seqs[-1] += line.upper()
    return ids, seqs


tok = RnaTokenizer.from_pretrained(MODEL_NAME)
model = RnaErnieModel.from_pretrained(MODEL_NAME).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Loaded {MODEL_NAME} on {device}")


@torch.no_grad()
def embed(seqs):
    """Mean-pooled 768-d embedding per sequence (padding ignored)."""
    chunks = []
    for i in range(0, len(seqs), BATCH):
        enc = tok(seqs[i:i + BATCH], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        h = model(**enc).last_hidden_state                  # (batch, tokens, 768)
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        chunks.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.vstack(chunks).astype("float32")


for name in ["mirna_mature_probes", "mirna_precursors"]:
    ids, seqs = read_fasta(DATA / f"{name}.fasta")
    print(f"{name}: embedding {len(seqs)} sequences ...")
    emb = embed(seqs)
    np.save(OUT / f"{name}_rnaernie.npy", emb)
    pd.Series(ids, name="id").to_csv(OUT / f"{name}_ids.csv", index=False)
    print(f"  saved {emb.shape} -> {OUT / (name + '_rnaernie.npy')}")

print("\nDone. Embeddings are in data/embeddings/ (a few MB total).")
