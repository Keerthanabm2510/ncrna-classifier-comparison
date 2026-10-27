"""
snoRNA - Step 1: build the list of snoRNA sequences to give the models.

Dataset GSE209924 (human, TGIRT-seq of 4 cell lines x 2 replicates) gives how ABUNDANT each snoRNA is
(TPM table), not the sequences. So we:
  1. download the TPM table from GEO and the human ncRNA sequences from Ensembl (once, ~20 MB),
  2. keep the Ensembl transcripts whose biotype is snoRNA (one per gene: the longest),
  3. keep those that are expressed in this dataset (TPM >= MIN_TPM in at least one sample).

Run (from any folder):   python snorna_step1_prepare.py
Optional:  MIN_TPM=0.1 python snorna_step1_prepare.py     (default 1.0)

Folders (inside snorna/):
  raw/      files downloaded from GEO and Ensembl
  data/     inputs for the models (made by this step)
  results/  predictions and analysis (made by steps 2 and 3)

Output of this step (in snorna/data/):
  snorna_expressed.fasta     snoRNAs expressed in GSE209924  (this is what the models are run on)
  snorna_all_ensembl.fasta   every human Ensembl snoRNA (backup / for reference)
  snorna_expressed_table.csv id, symbol, expected subtype (from the name), length, max TPM
"""
import gzip
import os
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("SNORNA_DIR", BASE / "snorna"))
DATA = DIR / "data"                 # inputs made from the downloads (FASTA, table)
RAW = DIR / "raw"                   # files downloaded from GEO and Ensembl
MIN_TPM = float(os.environ.get("MIN_TPM", 1.0))

TPM_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209924/suppl/GSE209924_tpm_matrix.tsv.gz"
ENS_URL = "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz"
TPM_FILE = Path(os.environ.get("TPM_FILE", RAW / "GSE209924_tpm_matrix.tsv.gz"))
ENS_FILE = Path(os.environ.get("ENSEMBL_FILE", RAW / "Homo_sapiens.GRCh38.ncrna.fa.gz"))


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def subtype_from_symbol(sym):
    s = sym.upper()
    if s.startswith("SNORD"): return "CD-box"
    if s.startswith("SNORA"): return "HACA-box"
    if s.startswith("SCARNA"): return "scaRNA"
    return "unknown"


def read_ensembl_snornas():
    """{gene_id_without_version: dict(symbol, seq, transcript)} - longest snoRNA transcript per gene."""
    genes, cur = {}, None
    with gzip.open(ENS_FILE, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur: _store(genes, cur)
                cur = None
                m_bt = re.search(r"transcript_biotype:(\S+)", line)
                if not m_bt or m_bt.group(1) != "snoRNA":
                    continue
                gid = re.search(r"gene:(\S+)", line)
                sym = re.search(r"gene_symbol:(\S+)", line)
                gid = gid.group(1).split(".")[0] if gid else line[1:].split()[0]
                cur = {"gene": gid, "symbol": sym.group(1) if sym else gid,
                       "transcript": line[1:].split()[0], "seq": ""}
            elif cur is not None:
                cur["seq"] += line.strip().upper()
    if cur: _store(genes, cur)
    return genes


def _store(genes, rec):
    old = genes.get(rec["gene"])
    if old is None or len(rec["seq"]) > len(old["seq"]):
        genes[rec["gene"]] = rec


def write_fasta(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(f">{r['gene']}|{r['symbol']} snoRNA\n{r['seq']}\n")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    fetch(TPM_URL, TPM_FILE); fetch(ENS_URL, ENS_FILE)

    genes = read_ensembl_snornas()
    print(f"Ensembl: {len(genes)} snoRNA genes (one longest transcript each)")
    write_fasta(DATA / "snorna_all_ensembl.fasta", genes.values())
    by_symbol = {}
    for g in genes.values():
        by_symbol.setdefault(g["symbol"].upper(), g)

    df = pd.read_csv(TPM_FILE, sep="\t")
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    num = df.select_dtypes("number")
    txt = df.drop(columns=num.columns)
    print(f"TPM table: {len(df)} rows; {num.shape[1]} numeric (sample) columns; text columns: {list(txt.columns)}")
    if txt.shape[1] == 0 or num.shape[1] == 0:
        print(df.head()); sys.exit("Could not tell the ID column from the sample columns - send me the lines above.")

    matched = {}                       # gene id -> max TPM
    used_col = None
    for col in txt.columns:
        keys = txt[col].astype(str).str.strip()
        hits = 0
        for k, row_max in zip(keys, num.max(axis=1)):
            k0 = k.split(".")[0]
            rec = genes.get(k0) or by_symbol.get(k.upper()) or by_symbol.get(k.split("|")[-1].upper())
            if rec is not None:
                matched[rec["gene"]] = max(matched.get(rec["gene"], 0), float(row_max)); hits += 1
        print(f"  column '{col}': matched {hits} rows to Ensembl snoRNAs")
        if hits:
            used_col = col; break
    if not matched:
        print(txt.head(10)); sys.exit("No TPM row matched an Ensembl snoRNA - send me the lines above.")

    expr = [genes[g] for g, mx in matched.items() if mx >= MIN_TPM]
    expr.sort(key=lambda r: r["gene"])
    print(f"snoRNAs found in the TPM table: {len(matched)}; expressed (TPM >= {MIN_TPM} in at least one sample): {len(expr)}")
    write_fasta(DATA / "snorna_expressed.fasta", expr)
    tab = pd.DataFrame({"id": [f"{r['gene']}|{r['symbol']}" for r in expr], "symbol": [r["symbol"] for r in expr],
                        "expected_subtype": [subtype_from_symbol(r["symbol"]) for r in expr],
                        "length": [len(r["seq"]) for r in expr], "max_TPM": [round(matched[r["gene"]], 2) for r in expr]})
    tab.to_csv(DATA / "snorna_expressed_table.csv", index=False)
    print(f"length: min {tab.length.min()}, median {int(tab.length.median())}, max {tab.length.max()} nt")
    print(tab.expected_subtype.value_counts().to_string())
    print(f"\nsaved in {DATA}\nNext: python snorna_step2_run_models.py")


if __name__ == "__main__":
    main()
