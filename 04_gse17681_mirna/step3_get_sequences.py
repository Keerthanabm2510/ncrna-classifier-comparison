"""
Step 3 (FIXED): build the mature-miRNA FASTA directly from the platform's own
SEQUENCE column, instead of matching names against miRBase.

Why: GPL9040 uses old miRBase 13.0 names (e.g. hsa-miR-126, hsa-miR-18a*),
which no longer exist in current miRBase (now hsa-miR-126-3p, hsa-miR-18a-3p),
so name matching failed for ~700 probes. The array already ships the probe
sequence, so no matching is needed. miRBase is used only as a cross-check.

Needs: pandas  (cache/mature.fa comes from your earlier step 3 run)
"""
import os
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DATA, CACHE = BASE / "data", BASE / "cache"

# --- 1. Load platform table (ID, SEQUENCE) ---
p = pd.read_csv(DATA / "GPL9040_platform_table.csv")[["ID", "SEQUENCE"]].dropna()
p["seq"] = p["SEQUENCE"].str.upper().str.replace("T", "U")   # RNAErnie alphabet: A/U/C/G
assert p["seq"].str.fullmatch(r"[ACGU]+").all(), "unexpected characters in SEQUENCE"

# --- 2. Cross-check against current human mature miRBase (exact sequence) ---
seq2names = {}
name = None
for line in open(CACHE / "mature.fa"):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:].split()[0]
    elif name and name.startswith("hsa-"):
        seq2names.setdefault(line.upper(), []).append(name)

# --- 3. QC table: length, shared sequences, current miRBase name ---
shared = p.groupby("seq")["ID"].apply(lambda s: ";".join(s))
p["length"] = p["seq"].str.len()
p["shares_seq_with"] = p["seq"].map(lambda s: shared[s] if ";" in shared[s] else "")
p["current_mirbase_name"] = p["seq"].map(lambda s: ";".join(seq2names.get(s, [])))
p.rename(columns={"ID": "probe_id"})[
    ["probe_id", "seq", "length", "current_mirbase_name", "shares_seq_with"]
].to_csv(DATA / "probe_qc.csv", index=False)

# --- 4. Mature FASTA: one entry per probe (866), ID = probe name from the array ---
with open(DATA / "mirna_mature_probes.fasta", "w") as f:
    for pid, s in zip(p["ID"], p["seq"]):
        f.write(f">{pid}\n{s}\n")

print(f"Probes written           : {len(p)}")
print(f"Length range (nt)        : {p['length'].min()}-{p['length'].max()}")
print(f"Match current miRBase    : {(p['current_mirbase_name'] != '').sum()}")
print(f"Not exact in miRBase now : {(p['current_mirbase_name'] == '').sum()} (older sequence versions; still valid input)")
print(f"Probes sharing a sequence: {(p['shares_seq_with'] != '').sum()} (array design quirk, see probe_qc.csv)")
print(f"Saved: {DATA/'mirna_mature_probes.fasta'} and probe_qc.csv")
