"""
Mixed RNA-type pool from GEO GSE310677 (human ESCs and iPSCs, 6 libraries) - Step 1.

The GEO file (GSE310677_smallRNA.CPM.csv.gz) lists every Ensembl gene with its RNA type (gene_biotype) and CPM in 6 libraries.
It has no sequences, so:
  1. EXPRESSED genes: CPM >= MIN_CPM (default 1) in >= MIN_SAMPLES of the 6 libraries (default 2);
  2. the RNA type comes from the file's own gene_biotype column;
  3. the sequence is taken from the Ensembl GRCh38 ncRNA FASTA (same file as in the snoRNA study), looked up by gene ID
     (the longest transcript of the gene).
Types kept: miRNA (precursor hairpin), snoRNA (scaRNA counted as snoRNA), snRNA, lncRNA (200-MAX_LNC_LEN nt; random sample of
N_LNC if more), rRNA (<= 200 nt, i.e. 5S / 5.8S-sized), tRNA (the mitochondrial Mt_tRNA genes - the only tRNAs Ensembl labels).
Not used: protein-coding genes (no model has an mRNA class), misc_RNA, pseudogenes.

Run (from any folder):   python gse310677_step1_prepare.py
Folders (inside 2_project/gse310677/):  raw/ (the downloaded CSV)   data/ (made here)   results/ (steps 2 and 3)
The CSV is downloaded if missing (or copied from smallrna/geo310677/ if you already have it there).
Output (data/): pool.fasta (">id TYPE"), pool_table.csv (id, true, gene, symbol, length, mean_count = mean CPM)
Then:  python gse310677_step2_run_models.py   and   python gse310677_step3_analysis.py
"""
import gzip
import os
import re
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("GSE310677_DIR", BASE / "gse310677"))
DATA = DIR / "data"
NAME = "GSE310677_smallRNA.CPM.csv.gz"
RAW = DIR / "raw"
URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE310nnn/GSE310677/suppl/" + NAME
OLD = BASE / "smallrna" / "geo310677" / NAME           # where an earlier download may be
CSV = RAW / NAME
ENS = Path(os.environ.get("ENSEMBL_FILE", BASE / "snorna" / "raw" / "Homo_sapiens.GRCh38.ncrna.fa.gz"))
MIN_CPM = float(os.environ.get("MIN_CPM", 1)); MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", 2))
N_LNC = int(os.environ.get("N_LNC", 300)); MAX_LNC_LEN = int(os.environ.get("MAX_LNC_LEN", 1000))
TYPES = {"miRNA": "miRNA", "snoRNA": "snoRNA", "scaRNA": "snoRNA", "snRNA": "snRNA", "lncRNA": "lncRNA",
         "rRNA": "rRNA", "Mt_tRNA": "tRNA"}


def main():
    RAW.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        if OLD.exists(): print("re-using", OLD); shutil.copy(OLD, CSV)
        else: print("downloading", URL); urllib.request.urlretrieve(URL, CSV)
    if not ENS.exists():
        raise SystemExit(f"missing file: {ENS} (the Ensembl ncRNA FASTA from the snoRNA study; set ENSEMBL_FILE=... if it is elsewhere)")
    d = pd.read_csv(CSV, index_col=0, low_memory=False)
    s = [c for c in d.columns if c.endswith(".bam")]
    d["Geneid"] = d["Geneid"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    cpm = d[s].apply(pd.to_numeric, errors="coerce").fillna(0)
    expr = d[(cpm >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES].copy(); expr["mean_cpm"] = cpm.loc[expr.index].mean(axis=1)
    print(f"{len(d)} genes, {len(s)} libraries; expressed (CPM >= {MIN_CPM} in >= {MIN_SAMPLES}): {len(expr)}")
    keep = expr[expr["gene_biotype"].isin(TYPES)]
    print("expressed genes of the types used:", keep["gene_biotype"].value_counts().to_dict())
    info = {r.Geneid: (TYPES[r.gene_biotype], str(r.gene_name) if pd.notna(r.gene_name) else "", r.mean_cpm) for r in keep.itertuples()}
    best, cur = {}, None
    for line in gzip.open(ENS, "rt", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            g = re.search(r"gene:(\S+?)(\.\d+)?(\s|$)", line); cur = None
            if g and g.group(1) in info:
                t, sym, mc = info[g.group(1)]
                cur = {"id": line[1:].split()[0], "gene": g.group(1), "true": t, "symbol": sym, "mean_count": mc, "seq": ""}
                best.setdefault(g.group(1), []).append(cur)
        elif cur is not None: cur["seq"] += line.strip().upper().replace("U", "T")
    rows = []
    for g, lst in best.items():
        t = lst[0]["true"]
        if t == "lncRNA": lst = [r for r in lst if 200 <= len(r["seq"]) <= MAX_LNC_LEN]
        if t == "rRNA": lst = [r for r in lst if 0 < len(r["seq"]) <= 200]
        if lst: rows.append(max(lst, key=lambda r: len(r["seq"])))
    df = pd.DataFrame(rows)
    print(f"genes found in the Ensembl ncRNA FASTA (after length filters): {df['true'].value_counts().to_dict()}")
    print(f"genes of these types without a sequence there: {len(info) - len(best)}")
    lnc = df[df["true"] == "lncRNA"]
    if len(lnc) > N_LNC:
        df = pd.concat([df[df["true"] != "lncRNA"], lnc.sample(N_LNC, random_state=0)]); print(f"lncRNA: random sample of {N_LNC} from {len(lnc)}")
    df = df.drop_duplicates("seq").sort_values(["true", "id"]).reset_index(drop=True)
    df["length"] = df["seq"].str.len(); df["mean_count"] = df["mean_count"].astype(float).round(2)
    with open(DATA / "pool.fasta", "w") as f:
        for r in df.itertuples(): f.write(f">{r.id} {r.true}\n{r.seq}\n")
    df[["id", "true", "gene", "symbol", "length", "mean_count"]].to_csv(DATA / "pool_table.csv", index=False)
    print(f"\n{len(df)} distinct sequences in total:")
    print(df.groupby("true")["length"].agg(["count", "min", "median", "max"]).to_string())
    print(f"\nsaved in {DATA}\nNext: python gse310677_step2_run_models.py")


if __name__ == "__main__":
    main()
