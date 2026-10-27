"""
MncR - Step 1: download only what is needed from the authors' GitHub repository
(github.com/Stegobully/merged_ncRNAclassifier). The full repo is ~290 MB (training data);
we take ~15 MB: two pretrained models, the Rfam benchmark subset, a 180-sequence self-test set.
Also copies your two GSE17681 FASTA files from the RNAErnie folder (nothing is re-built).

Needs: pip install requests
Run:   python mncr_step1_download.py
"""
import os
import shutil
from pathlib import Path
import requests
from mncr_common import DATA, MODELS, BASE

RAW = "https://raw.githubusercontent.com/Stegobully/merged_ncRNAclassifier/main/"
RNAERNIE_DATA = Path(os.environ.get("RNAERNIE_DATA", BASE / "mirna" / "RNAErnie" / "data"))

# (path in repo, destination folder, expected size in bytes)
FILES = [
    ("model_files/mncr_fold7.hdf5", MODELS, 9091952),      # MncR = sequence CNN + graph ANN (best model)
    ("model_files/seqenc_fold8.hdf5", MODELS, 5142472),    # sequence-only CNN from the same paper
    ("testing_datasets/rfam_testset_used_for_ncrdense.fasta", DATA, 175748),   # 1,125-seq Rfam benchmark
    ("testing_datasets/small_testset_30.fasta", DATA, 102840),
    ("testing_datasets/small_testset_30_labels.txt", DATA, 1080),
    ("testing_datasets/small_testset_30_graphprot.feature", DATA, 52683762),   # 50 MB, see note below
]
# The 50 MB .feature file is only used for the self-test (Step 2) and to validate your GraphProt
# install (Step 4). Delete it afterwards if you want the space back.


def main():
    MODELS.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    for rel, folder, size in FILES:
        dest = folder / Path(rel).name
        if dest.exists() and dest.stat().st_size == size:
            print(f"have  {dest.name}"); continue
        r = requests.get(RAW + rel, timeout=300); r.raise_for_status()
        dest.write_bytes(r.content)
        ok = "OK" if dest.stat().st_size == size else f"SIZE MISMATCH (expected {size})"
        print(f"saved {dest.name}: {dest.stat().st_size/1e6:.1f} MB  {ok}")

    for name in ["mirna_precursors.fasta", "mirna_mature_probes.fasta"]:
        src = RNAERNIE_DATA / name
        if src.exists():
            shutil.copy(src, DATA / name); print(f"copied {name} from the RNAErnie folder")
        else:
            print(f"MISSING {src} - set RNAERNIE_DATA to the folder that has it")
    print(f"\nEverything is under: {BASE / 'mirna' / 'mncr'}")


if __name__ == "__main__":
    main()
