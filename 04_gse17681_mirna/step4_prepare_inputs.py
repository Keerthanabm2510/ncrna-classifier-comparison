"""
Step 4: prepare the final RNAErnie inputs.

Mature miRNAs are only ~22 nt, but the classifiers in your papers were trained on
precursor (hairpin) miRNAs (Rfam 'miRNA' class ~60-100 nt). So we also build a
precursor FASTA by locating each mature sequence inside human hairpins from miRBase.
That gives you a fairer comparison set. Both files are validated for RNAErnie
(A/U/C/G only, <=512 nt).

Needs: pandas (cache/hairpin.fa from earlier)
"""
import os
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DATA, CACHE = BASE / "data", BASE / "cache"

qc = pd.read_csv(DATA / "probe_qc.csv")

# --- human hairpins: {name: seq} ---
# NOTE: hairpin.fa wraps sequences at 70 characters, so join all lines per record.
hairpins, name = {}, None
for line in open(CACHE / "hairpin.fa"):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:].split()[0]
    elif name and name.startswith("hsa-"):
        hairpins[name] = hairpins.get(name, "") + line.upper()

# --- probe -> hairpin(s) containing the mature sequence ---
rows, used = [], {}
for pid, seq in zip(qc["probe_id"], qc["seq"]):
    hits = [h for h, hs in hairpins.items() if seq in hs]
    rows.append({"probe_id": pid, "n_hairpins": len(hits), "hairpins": ";".join(hits)})
    for h in hits:
        used[h] = hairpins[h]
pd.DataFrame(rows).to_csv(DATA / "probe_to_precursor.csv", index=False)

with open(DATA / "mirna_precursors.fasta", "w") as f:
    for h, s in used.items():
        f.write(f">{h}\n{s}\n")

# --- validation for RNAErnie ---
def check(path):
    n = bad = long = 0
    name = None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"): continue
        n += 1
        bad += (set(line) - set("ACGU") != set())
        long += (len(line) > 512)
    return n, bad, long

for fa in ["mirna_mature_probes.fasta", "mirna_precursors.fasta"]:
    n, bad, long = check(DATA / fa)
    print(f"{fa}: {n} seqs | non-ACGU: {bad} | >512 nt: {long}")

found = sum(r["n_hairpins"] > 0 for r in rows)
print(f"\nProbes with a precursor found: {found}/{len(rows)}")
print(f"Unique precursors written    : {len(used)}")
print("Probes with no precursor listed in data/probe_to_precursor.csv (n_hairpins = 0).")
