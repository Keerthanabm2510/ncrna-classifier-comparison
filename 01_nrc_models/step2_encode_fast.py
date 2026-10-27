"""
ncRDense - Step 2: encode sequence + secondary structure into the fixed-length
representations the paper uses (max length 750 nt), following their exact
mapping (Methods, section 3.2.5).

FAST VERSION: uses NumPy lookup tables instead of per-character loops.

Storage note: rather than saving full one-hot arrays (which would be ~50x
larger), this saves compact integer codes (1 byte/position). Step 3 turns
these into one-hot vectors on the fly per training batch, which keeps disk
usage small without changing what the model sees.

Needs: pip install numpy pandas
Run:   python step2_encode.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(os.environ.get(
    "NCRDENSE_DATA", "/home/keerthana/Documents/mini_project/2_project/mirna/ncrdense/data"))

MAXLEN = 750                      # fixed length used in the ncRDense paper
SEQ_CODE = {"A": 0, "U": 1, "G": 2, "C": 3}   # anything else (incl. padding) -> 4 ("N"/pad, all-zero one-hot)
STRUCT_CODE = {".": 0, "(": 1, ")": 2}         # padding -> 3 (all-zero one-hot)

SOURCES = ["nrc_train", "nrc_test", "mirna_mature_probes", "mirna_precursors"]


# 256-entry lookup tables: byte value -> code. Everything not listed gets the pad/N code.
_SEQ_LUT = np.full(256, 4, dtype="int8")
for _ch, _c in SEQ_CODE.items():
    _SEQ_LUT[ord(_ch)] = _c
_STRUCT_LUT = np.full(256, 3, dtype="int8")
for _ch, _c in STRUCT_CODE.items():
    _STRUCT_LUT[ord(_ch)] = _c


def _encode(text, lut, pad):
    codes = np.full(MAXLEN, pad, dtype="int8")
    raw = np.frombuffer(text[:MAXLEN].encode("ascii", "replace"), dtype=np.uint8)
    codes[:len(raw)] = lut[raw]          # one vectorised lookup instead of a Python loop
    return codes


def encode_seq(seq: str) -> np.ndarray:
    return _encode(seq.upper().replace("T", "U"), _SEQ_LUT, 4)


def encode_struct(struct: str) -> np.ndarray:
    return _encode(struct, _STRUCT_LUT, 3)


def main():
    for name in SOURCES:
        path = DATA / f"{name}_structure.csv"
        if not path.exists():
            print(f"SKIP {name}: {path} not found (run step1 first)")
            continue
        df = pd.read_csv(path)

        seq_codes = np.stack([encode_seq(s) for s in df["seq"]])
        struct_codes = np.stack([encode_struct(s) for s in df["structure"]])

        np.save(DATA / f"{name}_seq_codes.npy", seq_codes)
        np.save(DATA / f"{name}_struct_codes.npy", struct_codes)

        meta = df[["id", "label"]].copy()
        meta["length"] = df["seq"].str.len()
        meta["truncated"] = meta["length"] > MAXLEN
        meta.to_csv(DATA / f"{name}_encoded_meta.csv", index=False)

        n_trunc = meta["truncated"].sum()
        size_mb = (seq_codes.nbytes + struct_codes.nbytes) / 1e6
        print(f"{name}: {len(df)} sequences | truncated (>{MAXLEN}nt): {n_trunc} "
              f"| saved arrays: {size_mb:.1f} MB")

    print(f"\nAll encoded arrays saved under: {DATA}")


if __name__ == "__main__":
    main()
