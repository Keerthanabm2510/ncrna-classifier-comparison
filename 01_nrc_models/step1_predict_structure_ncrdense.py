"""
ncRDense - Step 1: predict secondary structure (RNAfold/ViennaRNA) for every
sequence used so far. Reuses the FASTA files already built for RNAErnie -
nothing is re-downloaded.

Needs: pip install ViennaRNA pandas
Run:   python step1_predict_structure.py
"""
import os
from pathlib import Path
import RNA
import pandas as pd

# Points at the sibling RNAErnie folder so files are reused, not duplicated.
RNAERNIE_DATA = Path(os.environ.get(
    "RNAERNIE_DATA", "/home/keerthana/Documents/mini_project/2_project/mirna/RNAErnie/data"))
OUT = Path(os.environ.get(
    "NCRDENSE_DATA", "/home/keerthana/Documents/mini_project/2_project/mirna/ncrdense/data"))
OUT.mkdir(parents=True, exist_ok=True)

# (source fasta, output name) - all four files reused from the RNAErnie run
SOURCES = [
    (RNAERNIE_DATA / "nrc" / "dataset_Rfam_6320_13classes.fasta", "nrc_train"),
    (RNAERNIE_DATA / "nrc" / "dataset_Rfam_validated_2600_13classes.fasta", "nrc_test"),
    (RNAERNIE_DATA / "mirna_mature_probes.fasta", "mirna_mature_probes"),
    (RNAERNIE_DATA / "mirna_precursors.fasta", "mirna_precursors"),
]


def read_fasta(path):
    """Handles wrapped (multi-line) sequences and 'id label' style headers."""
    ids, labels, seqs = [], [], []
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            parts = line[1:].split()
            ids.append(parts[0])
            labels.append(parts[-1] if len(parts) > 1 else "")
            seqs.append("")
        else:
            seqs[-1] += line
    return ids, labels, seqs


def main():
    for src, name in SOURCES:
        if not src.exists():
            print(f"SKIP  {name}: not found at {src}")
            continue
        ids, labels, seqs = read_fasta(src)
        structures, mfes = [], []
        for s in seqs:
            s_rna = s.upper().replace("T", "U")
            struct, mfe = RNA.fold(s_rna)
            structures.append(struct)
            mfes.append(round(mfe, 2))

        df = pd.DataFrame({"id": ids, "label": labels, "seq": seqs,
                            "structure": structures, "mfe": mfes})
        df.to_csv(OUT / f"{name}_structure.csv", index=False)
        print(f"{name}: {len(df)} sequences folded -> {name}_structure.csv "
              f"(length {df.seq.str.len().min()}-{df.seq.str.len().max()} nt)")

    print(f"\nAll structure files saved under: {OUT}")


if __name__ == "__main__":
    main()
