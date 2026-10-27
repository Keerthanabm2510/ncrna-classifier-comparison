"""
Step 7: download the nRC benchmark (train + test FASTA + readme) and peek at the format.
Small files only (a few MB in total). Saved to data/nrc/.

Run: python step7_download_nrc.py
"""
import os
from pathlib import Path
import requests

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
OUT = BASE / "data" / "nrc"
OUT.mkdir(parents=True, exist_ok=True)

URL = "http://tblab.pa.icar.cnr.it/public/nRC/paper_dataset/"
FILES = [
    "readme.txt",
    "dataset_Rfam_6320_13classes.fasta",            # training set
    "dataset_Rfam_validated_2600_13classes.fasta",  # test set
]

for name in FILES:
    dest = OUT / name
    if not dest.exists():
        r = requests.get(URL + name, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    print(f"{name}: {dest.stat().st_size / 1024:.0f} KB")

print("\n===== readme.txt =====")
print((OUT / "readme.txt").read_text(errors="replace")[:2000])

for name in FILES[1:]:
    headers = [l.strip() for l in open(OUT / name, errors="replace") if l.startswith(">")]
    print(f"\n===== {name}: {len(headers)} records; first 5 headers =====")
    print("\n".join(headers[:5]))
