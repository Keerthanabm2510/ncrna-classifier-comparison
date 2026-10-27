"""
tRNA from GEO GSE229667 (MapID-tRNA-seq, human cells) - Step 1: choose the EXPRESSED tRNAs and get their sequences.

Input 1: the 16 Salmon files (GSM*_quant.sf.gz) in  trna/geo229667/raw/   (already downloaded and extracted by you)
Input 2: the GtRNAdb sequences  trna/raw/hg38-tRNAs.fa  (the file you already have) - the Salmon files have no sequences,
         so each expressed tRNA is matched BY NAME to its GtRNAdb gene and that sequence is used.
Mitochondrial tRNAs (names starting MT_) are matched only if they exist in the GtRNAdb file; otherwise they are listed as unmatched.

A tRNA counts as EXPRESSED if TPM >= MIN_TPM (default 1) in at least MIN_SAMPLES of the 16 samples (default 4).

Run:   python trna_geo_step1_prepare.py          (optional: MIN_TPM=5 MIN_SAMPLES=8 python ...)
Output (trna/geo229667/data/):  trna.fasta, trna_table.csv (id, amino_acid, anticodon, score, chrom, length)
   Here "score" = mean TPM over the 16 samples (expression level), not the tRNAscan-SE score.
   unmatched_names.csv lists expressed tRNAs for which no sequence was found.
Then:  TRNA_DIR=<this folder> python trna_step2_run_models.py   and   TRNA_DIR=<this folder> python trna_step3_analysis.py
"""
import glob
import gzip
import os
import re
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("TRNA_DIR", BASE / "trna" / "geo229667"))
RAW, DATA = DIR / "raw", DIR / "data"
GTRNADB = Path(os.environ.get("TRNA_FASTA", BASE / "trna" / "raw" / "hg38-tRNAs.fa"))
MIN_TPM = float(os.environ.get("MIN_TPM", 1)); MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", 4))


def read_gtrnadb():
    recs, cur = {}, None
    for line in open(GTRNADB, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            h = line[1:]; name = h.split()[0].replace("Homo_sapiens_", "")
            ch = re.search(r"\s(chr\S+):", h)
            cur = {"name": name, "chrom": ch.group(1) if ch else "?", "seq": ""}; recs[name.upper()] = cur
        elif cur is not None:
            cur["seq"] += line.strip().upper().replace("U", "T")
    return recs


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(RAW / "*quant.sf*")))
    if not files: raise SystemExit(f"no quant.sf files in {RAW}")
    tpm = {}
    for f in files:
        d = pd.read_csv(f, sep="\t", compression="gzip" if f.endswith(".gz") else None)
        tpm[Path(f).name.split("_")[0]] = d.set_index("Name")["TPM"]
    T = pd.DataFrame(tpm).fillna(0)
    print(f"{len(files)} samples, {len(T)} tRNA entries (MapIDs)")
    expr = (T >= MIN_TPM).sum(axis=1) >= MIN_SAMPLES
    print(f"expressed (TPM >= {MIN_TPM} in >= {MIN_SAMPLES} samples): {int(expr.sum())} of {len(T)}")
    print(f"  of which mitochondrial (MT_): {int(expr[expr].index.str.startswith('MT_').sum())}")
    gt = read_gtrnadb(); by_prefix = {}
    for k in gt: by_prefix.setdefault(k.rsplit("-", 1)[0], k)
    print(f"GtRNAdb file: {len(gt)} genes")
    rows, miss, seen = [], [], set()
    for name in expr[expr].index:
        key = name.upper(); hit = gt.get(key) or gt.get(by_prefix.get(key.rsplit("-", 1)[0], ""))
        if hit is None: miss.append({"mapid": name, "mean_TPM": T.loc[name].mean()}); continue
        if hit["name"] in seen: continue                       # several MapIDs may point to one gene: keep it once
        seen.add(hit["name"])
        p = hit["name"].split("-")
        rows.append({"id": hit["name"], "amino_acid": p[1] if len(p) > 1 else "?", "anticodon": p[2] if len(p) > 2 else "?",
                     "score": round(float(T.loc[name].mean()), 2), "chrom": hit["chrom"], "seq": hit["seq"], "mapid": name})
    pd.DataFrame(miss).to_csv(DATA / "unmatched_names.csv", index=False)
    with open(DATA / "trna.fasta", "w") as f:
        for r in rows: f.write(f">{r['id']} tRNA\n{r['seq']}\n")
    tab = pd.DataFrame(rows); tab["length"] = tab["seq"].str.len()
    tab.drop(columns="seq").to_csv(DATA / "trna_table.csv", index=False)
    print(f"matched to a GtRNAdb sequence: {len(tab)}   not matched: {len(miss)} (see unmatched_names.csv)")
    if miss: print("  examples:", ", ".join(m["mapid"] for m in miss[:8]))
    print(f"length min {tab.length.min()}, median {int(tab.length.median())}, max {tab.length.max()} nt")
    print(f"\nsaved in {DATA}\nNext: TRNA_DIR={DIR} python trna_step2_run_models.py")


if __name__ == "__main__":
    main()
