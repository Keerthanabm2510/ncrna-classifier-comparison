"""
tRNA - Step 1: build the list of human tRNA sequences to give the models.

Source: GtRNAdb (tRNAscan-SE predictions on the human genome hg38), 432 tRNA genes, downloaded once (~50 KB).
This is a set of annotated genes, not an expression dataset, so there is no expression filter.
Each header carries the amino acid, anticodon and the tRNAscan-SE score (Sc, higher = more tRNA-like).

Run (from any folder):   python trna_step1_prepare.py
Optional:  MIN_SCORE=50 python trna_step1_prepare.py     keep only genes with tRNAscan-SE score >= 50 (default 0 = keep all)

Folders (inside trna/):
  raw/      the downloaded file
  data/     inputs for the models (made by this step)
  results/  predictions and analysis (made by steps 2 and 3)

Output of this step (trna/data/):
  trna.fasta       the sequences the models are run on
  trna_table.csv   id, amino acid, anticodon, score, length, chromosome
"""
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("TRNA_DIR", BASE / "trna"))
RAW, DATA = DIR / "raw", DIR / "data"
MIN_SCORE = float(os.environ.get("MIN_SCORE", 0))
URL = "https://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/hg38-tRNAs.fa"
SRC = Path(os.environ.get("TRNA_FASTA", RAW / "hg38-tRNAs.fa"))


def main():
    RAW.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    if not SRC.exists() or SRC.stat().st_size == 0:
        print("downloading hg38-tRNAs.fa ..."); urllib.request.urlretrieve(URL, SRC)
    recs, cur = [], None
    for line in open(SRC, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            h = line[1:]
            name = h.split()[0].replace("Homo_sapiens_", "")
            aa = re.search(r"\)\s+(\S+)\s+\((\S+)\)", h)
            sc = re.search(r"Sc:\s*([\d.]+)", h)
            ch = re.search(r"\s(chr\S+):", h)
            cur = {"id": name, "amino_acid": aa.group(1) if aa else "?", "anticodon": aa.group(2) if aa else "?",
                   "score": float(sc.group(1)) if sc else float("nan"), "chrom": ch.group(1) if ch else "?", "seq": ""}
            recs.append(cur)
        elif cur is not None:
            cur["seq"] += line.strip().upper()
    print(f"read {len(recs)} tRNA genes from {SRC.name}")
    keep = [r for r in recs if not (r["score"] == r["score"] and r["score"] < MIN_SCORE)]
    if len(keep) != len(recs):
        print(f"kept {len(keep)} with score >= {MIN_SCORE}")
    seen, out = set(), []
    for r in keep:                                    # make ids unique (they already should be)
        i, n = r["id"], 1
        while r["id"] in seen: n += 1; r["id"] = f"{i}_{n}"
        seen.add(r["id"]); out.append(r)
    with open(DATA / "trna.fasta", "w") as f:
        for r in out: f.write(f">{r['id']} tRNA\n{r['seq']}\n")
    tab = pd.DataFrame([{k: r[k] for k in ("id", "amino_acid", "anticodon", "score", "chrom")} | {"length": len(r["seq"])} for r in out])
    tab.to_csv(DATA / "trna_table.csv", index=False)
    print(f"{len(tab)} sequences; length min {tab.length.min()}, median {int(tab.length.median())}, max {tab.length.max()} nt")
    print(f"amino acids: {tab.amino_acid.nunique()} kinds; score min {tab.score.min()}, median {tab.score.median()}")
    print(f"\nsaved in {DATA}\nNext: python trna_step2_run_models.py")


if __name__ == "__main__":
    main()
