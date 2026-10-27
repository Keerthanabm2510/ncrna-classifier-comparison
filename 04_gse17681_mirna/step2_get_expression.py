"""
Step 2: Download the GSE17681 series (expression matrix + sample metadata).

pip install GEOparse pandas --break-system-packages
"""

from pathlib import Path
import GEOparse
import pandas as pd

BASE = Path("/home/keerthana/Documents/mini_project/2_project")
CACHE = BASE / "cache"
DATA = BASE / "data"

CACHE.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# --- Download the full series, cached in CACHE ---
gse = GEOparse.get_GEO(geo="GSE17681", destdir=str(CACHE))

print(f"Series title: {gse.metadata.get('title')}")
print(f"Number of samples: {len(gse.gsms)}")

# --- Sample metadata (normal vs lung cancer) ---
sample_meta = []
for gsm_name, gsm in gse.gsms.items():
    sample_meta.append({
        "sample_id": gsm_name,
        "title": gsm.metadata.get("title", [""])[0],
        "characteristics": "; ".join(gsm.metadata.get("characteristics_ch1", [])),
        "source": gsm.metadata.get("source_name_ch1", [""])[0],
    })

sample_meta_df = pd.DataFrame(sample_meta)
sample_meta_df.to_csv(DATA / "GSE17681_sample_metadata.csv", index=False)
print("\nSample metadata preview:")
print(sample_meta_df.head())
print("\nCheck the 'characteristics' column to confirm normal vs lung-cancer group labels.")

# --- Expression matrix (probes x samples) ---
expression_matrix = gse.pivot_samples("VALUE")
expression_matrix.to_csv(DATA / "GSE17681_expression_matrix.csv")
print(f"\nExpression matrix shape: {expression_matrix.shape}")

# --- Merge with the probe->miRNA mapping from Step 1, if available ---
mapping_path = DATA / "probe_to_mirna_mapping.csv"
if mapping_path.exists():
    mapping = pd.read_csv(mapping_path)
    merged = expression_matrix.merge(
        mapping, left_index=True, right_on="probe_id", how="left"
    )
    merged.to_csv(DATA / "GSE17681_expression_with_mirna_ids.csv", index=False)
    print("Merged expression matrix with miRNA IDs saved.")
else:
    print("(Run step1 first and set MIRNA_ID_COLUMN to enable this merge.)")
