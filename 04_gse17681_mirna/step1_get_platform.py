"""
Step 1: Download the GPL9040 platform annotation table.
Run from anywhere; paths are absolute under ~/Documents/mini_project.

pip install GEOparse pandas --break-system-packages
"""

from pathlib import Path
import GEOparse
import pandas as pd

BASE = Path("/home/keerthana/Documents/mini_project/2_project")
CACHE = BASE / "cache"     # raw GEO files — safe to delete after all steps done
DATA = BASE / "data"       # small, final outputs you keep

CACHE.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# --- Download platform (GPL9040), cached in CACHE ---
gpl = GEOparse.get_GEO(geo="GPL9040", destdir=str(CACHE))

print("Columns available in the platform table:")
print(gpl.table.columns.tolist())
print(gpl.table.head(10))

# Save full table to DATA so you can inspect and pick the miRNA-name column.
gpl.table.to_csv(DATA / "GPL9040_platform_table.csv", index=False)

# --- Fill this in after checking the printed columns above ---
MIRNA_ID_COLUMN = "REPLACE_WITH_ACTUAL_COLUMN_NAME"

if MIRNA_ID_COLUMN in gpl.table.columns:
    mapping = gpl.table[["ID", MIRNA_ID_COLUMN]].dropna()
    mapping.columns = ["probe_id", "mirna_id"]
    mapping.to_csv(DATA / "probe_to_mirna_mapping.csv", index=False)
    print(f"\nSaved {len(mapping)} probe->miRNA mappings to data/probe_to_mirna_mapping.csv")
else:
    print(f"\nColumn '{MIRNA_ID_COLUMN}' not found — inspect data/GPL9040_platform_table.csv, "
          f"set MIRNA_ID_COLUMN correctly, and re-run.")
