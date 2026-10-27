"""
Single-experiment mixed ncRNA test on GEO GSE209924 (TGIRT-seq of 4 human cell lines, 8 samples) - Step 1.

One experiment, one protocol: the TPM matrix has Ensembl genes (lncRNA, miRNA, snoRNA, snRNA, rRNA ...) AND tRNA rows.
  * Ensembl rows (ENSG...): expressed genes are looked up in the Ensembl GRCh38 ncRNA FASTA -> sequence + RNA type (gene_biotype).
  * tRNA rows ("tRNA-Asn-GTT-..."): matched BY NAME to the GtRNAdb hg38 tRNA sequences.
  * snoDB_, NR_ and cluster_ rows have no Ensembl/GtRNAdb link and are not used (counted and printed).
A gene is EXPRESSED if TPM >= MIN_TPM (default 1) in >= MIN_SAMPLES of the 8 samples (default 2).
Types kept: miRNA (precursor hairpin), snoRNA (scaRNA counted as snoRNA), tRNA (+ mitochondrial tRNA), rRNA (5S/5.8S, <= 200 nt),
snRNA, lncRNA. mRNA is not in the ncRNA file and is not used (no model has an mRNA class).
lncRNA is large, so a random sample of N_LNC (default 300) lncRNAs of 200-MAX_LNC_LEN nt (default 1000) is taken.

Files are re-used from your earlier steps: snorna/raw/ (TPM matrix, Ensembl FASTA) and trna/raw/hg38-tRNAs.fa.
Run (from any folder):   python gse209924_step1_prepare.py
Folders (inside 2_project/gse209924/): data/ and results/ are made here. The inputs are re-used from snorna/raw/ and trna/raw/ (nothing is downloaded).
Output (gse209924/data/): pool.fasta (">id TYPE"), pool_table.csv (id, true, gene, symbol, length, mean_count = mean TPM)
   (the data file names match the GSE310677 test, so steps 2 and 3 work the same way)
Then:  python gse209924_step2_run_models.py   and   python gse209924_step3_analysis.py
"""
import gzip
import os
import re
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("GSE209924_DIR", BASE / "gse209924"))
DATA = DIR / "data"
TPM = Path(os.environ.get("TPM_FILE", BASE / "snorna" / "raw" / "GSE209924_tpm_matrix.tsv.gz"))
ENS = Path(os.environ.get("ENSEMBL_FILE", BASE / "snorna" / "raw" / "Homo_sapiens.GRCh38.ncrna.fa.gz"))
GTRNADB = Path(os.environ.get("TRNA_FASTA", BASE / "trna" / "raw" / "hg38-tRNAs.fa"))
MIN_TPM = float(os.environ.get("MIN_TPM", 1)); MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", 2))
N_LNC = int(os.environ.get("N_LNC", 300)); MAX_LNC_LEN = int(os.environ.get("MAX_LNC_LEN", 1000))
TYPES = {"miRNA": "miRNA", "snoRNA": "snoRNA", "scaRNA": "snoRNA", "snRNA": "snRNA", "lncRNA": "lncRNA",
         "rRNA": "rRNA", "Mt_tRNA": "tRNA"}


def read_gtrnadb():
    recs, cur = {}, None
    for line in open(GTRNADB, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            name = line[1:].split()[0].replace("Homo_sapiens_", ""); cur = {"name": name, "seq": ""}; recs[name.upper()] = cur
        elif cur is not None: cur["seq"] += line.strip().upper().replace("U", "T")
    return recs


def main():
    for p in (TPM, ENS, GTRNADB):
        if not p.exists(): raise SystemExit(f"missing file: {p}")
    DATA.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(TPM, sep="\t"); ids = m["gene_id"].astype(str); sym = m["gene_name"].astype(str)
    num = m.drop(columns=["gene_id", "gene_name"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    expr = (num >= MIN_TPM).sum(axis=1) >= MIN_SAMPLES; mean = num.mean(axis=1)
    kind = pd.Series("other", index=m.index)
    kind[ids.str.startswith("ENSG")] = "ensembl"; kind[ids.str.lower().str.startswith("trna")] = "trna"
    print(f"{len(m)} rows, {num.shape[1]} samples; expressed (TPM >= {MIN_TPM} in >= {MIN_SAMPLES} samples) by row kind:")
    print(pd.crosstab(kind, expr).rename(columns={False: "not expressed", True: "expressed"}).to_string())
    print("unused row kinds (snoDB_, NR_, cluster_ ...): expressed", int(((kind == "other") & expr).sum()))

    ens_rows = {g.split(".")[0]: (s, mean[i]) for i, (g, s) in zip(m.index[(kind == "ensembl") & expr], zip(ids[(kind == "ensembl") & expr], sym[(kind == "ensembl") & expr]))}
    best, cur = {}, None
    for line in gzip.open(ENS, "rt", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            g = re.search(r"gene:(\S+?)(\.\d+)?(\s|$)", line); b = re.search(r"gene_biotype:(\S+)", line); cur = None
            if g and b and g.group(1) in ens_rows and b.group(1) in TYPES:
                cur = {"id": line[1:].split()[0], "gene": g.group(1), "true": TYPES[b.group(1)], "symbol": ens_rows[g.group(1)][0],
                       "mean_count": ens_rows[g.group(1)][1], "seq": ""}
                best.setdefault(g.group(1), []).append(cur)
        elif cur is not None: cur["seq"] += line.strip().upper().replace("U", "T")
    rows = []
    for g, lst in best.items():
        t = lst[0]["true"]
        if t == "lncRNA": lst = [r for r in lst if 200 <= len(r["seq"]) <= MAX_LNC_LEN]
        if t == "rRNA": lst = [r for r in lst if len(r["seq"]) <= 200]
        if lst: rows.append(max(lst, key=lambda r: len(r["seq"])))
    df = pd.DataFrame(rows)
    print("\nEnsembl genes with a sequence, by type:", df["true"].value_counts().to_dict())

    gt = read_gtrnadb(); by_prefix = {}
    for k in gt: by_prefix.setdefault(k.rsplit("-", 1)[0], k)
    tr, miss = [], []
    for i in m.index[(kind == "trna") & expr]:
        key = ids[i].upper(); hit = gt.get(key) or gt.get(by_prefix.get(key.rsplit("-", 1)[0], ""))
        if hit is None: miss.append(ids[i]); continue
        tr.append({"id": hit["name"], "gene": ids[i], "true": "tRNA", "symbol": ids[i], "mean_count": mean[i], "seq": hit["seq"]})
    pd.Series(miss, dtype=str).to_csv(DATA / "unmatched_trna_names.csv", index=False, header=["name"])
    print(f"tRNA rows expressed: {int(((kind == 'trna') & expr).sum())}; matched to a GtRNAdb sequence: {len(tr)}; not matched: {len(miss)}"
          + (f" (e.g. {', '.join(miss[:4])})" if miss else ""))
    df = pd.concat([df, pd.DataFrame(tr)], ignore_index=True)
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
    print(f"\nsaved in {DATA}\nNext: python gse209924_step2_run_models.py")


if __name__ == "__main__":
    main()
