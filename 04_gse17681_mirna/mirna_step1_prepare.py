"""
miRNA - Step 1: build the list of human miRNA precursor (hairpin) sequences to give the models.

Source: miRBase hairpin.fa (all species, ~1 MB). Only human entries (hsa-) are kept.
Why hairpins and not the 22-nt mature miRNAs: RNAErnie, ncRDense and MncR were trained on precursor-length
sequences (about 60-120 nt), so the hairpin is the fair input.
This is a set of annotated genes, not an expression dataset, so there is no expression filter.

Run (from any folder):   python mirna_step1_prepare.py
If the download is refused, get hairpin.fa yourself from https://mirbase.org/download/ and put it in
mirna_mirbase/raw/ (hairpin.fa or hairpin.fa.gz), then run this again.

Folders (inside mirna_mirbase/):  raw/ (download)   data/ (made here)   results/ (steps 2 and 3)
Output (data/):  mirna.fasta  (">id miRNA")   mirna_table.csv  (id, accession, family, score, length)
  "family" = name without the trailing -number (hsa-let-7a-1 -> hsa-let-7, hsa-mir-30a -> hsa-mir-30);  "score" = MI accession number.
"""
import gzip
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DIR = Path(os.environ.get("MIRNA_DIR", BASE / "mirna_mirbase"))
RAW, DATA = DIR / "raw", DIR / "data"
URLS = ["https://mirbase.org/download/hairpin.fa", "https://mirbase.org/ftp/CURRENT/hairpin.fa.gz",
        "https://www.mirbase.org/download/hairpin.fa"]


def find_src():
    for n in ("hairpin.fa", "hairpin.fa.gz"):
        if (RAW / n).exists() and (RAW / n).stat().st_size > 0:
            return RAW / n
    for u in URLS:
        dest = RAW / u.rsplit("/", 1)[1]
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            if dest.stat().st_size > 1000:
                print("downloaded", u); return dest
        except Exception as e:
            print("failed:", u, e)
    raise SystemExit("Could not download hairpin.fa. Download it in a browser from https://mirbase.org/download/ "
                     f"and put it in {RAW}")


def main():
    RAW.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    src = find_src()
    op = gzip.open if str(src).endswith(".gz") else open
    recs, cur = [], None
    for line in op(src, "rt", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            parts = line[1:].split()
            cur = {"id": parts[0], "accession": parts[1] if len(parts) > 1 else "?", "seq": ""}
            recs.append(cur)
        elif cur is not None:
            cur["seq"] += line.strip().upper().replace("U", "T")
    print(f"read {len(recs)} entries; keeping human (hsa-)")
    hs = [r for r in recs if r["id"].startswith("hsa-")]
    with open(DATA / "mirna.fasta", "w") as f:
        for r in hs: f.write(f">{r['id']} miRNA\n{r['seq']}\n")
    tab = pd.DataFrame([{"id": r["id"], "accession": r["accession"],
                         "family": re.sub(r"(?<=\d)[a-z]$", "", re.sub(r"(?<=[a-z])-\d+$", "", r["id"])),
                         "score": int(re.sub(r"\D", "", r["accession"]) or 0), "length": len(r["seq"])} for r in hs])
    tab.to_csv(DATA / "mirna_table.csv", index=False)
    print(f"{len(tab)} human hairpins; length min {tab.length.min()}, median {int(tab.length.median())}, max {tab.length.max()} nt")
    print(f"{tab.family.nunique()} families; largest: " + ", ".join(f"{k} {v}" for k, v in tab.family.value_counts().head(5).items()))
    print(f"\nsaved in {DATA}\nNext: python mirna_step2_run_models.py")


if __name__ == "__main__":
    main()
