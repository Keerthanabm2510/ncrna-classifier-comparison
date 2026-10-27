"""
Shared helpers for the MncR pipeline (standalone: does not import the authors' scripts,
so it works with any recent TensorFlow/Keras and pandas).

The pretrained models and their preprocessing come from the MncR authors
(Dunkel et al. 2023, github.com/Stegobully/merged_ncRNAclassifier). Preprocessing here
reproduces theirs exactly:
  * sequences use DNA letters (U -> T!), padded on the right with "_" to 12,000 nt
  * letters are mapped to integers in alphabetical order (A B C D G H K M N R S T V W Y _) ; unknown -> 16
  * MncR also needs a 32,768-feature GraphProt vector per sequence (.feature file)
"""
import os
from pathlib import Path
import numpy as np

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
MNCR = BASE / "mirna" / "mncr"
DATA = MNCR / "data"
MODELS = MNCR / "model_files"

CLASSES = ["lncRNA", "miRNA", "rRNA", "snRNA", "snoRNA", "tRNA"]   # output order of the models
MAXLEN = 12000
N_GRAPH = 32768

# NOTE: the authors fit scikit-learn's OrdinalEncoder on these letters, which SORTS them, so the
# integer codes follow alphabetical order (A=0, B=1, C=2, D=3, G=4, ... T=11 ... "_"=15), not list order.
_CATS = sorted(["A", "C", "G", "T", "N", "R", "K", "S", "Y", "M", "W", "B", "H", "D", "V", "_"])
_LUT = np.full(256, 16, dtype=np.int16)
for _i, _c in enumerate(_CATS):
    _LUT[ord(_c)] = _i


def read_fasta(path):
    """Returns ids, labels, seqs. Label = last word of the header if there is more than one word."""
    ids, labels, seqs = [], [], []
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            parts = line[1:].split()
            ids.append(parts[0]); labels.append(parts[-1] if len(parts) > 1 else ""); seqs.append("")
        else:
            seqs[-1] += line
    return ids, labels, seqs


def encode_batch(seqs):
    """Sequences -> int array (n, 12000). U is converted to T first (the models were trained on DNA letters)."""
    out = np.full((len(seqs), MAXLEN), _CATS.index("_"), dtype=np.int16)
    for i, s in enumerate(seqs):
        s = s.upper().replace("U", "T")[:MAXLEN]
        raw = np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)
        out[i, :len(raw)] = _LUT[raw]
    return out


def read_feature_lines(path):
    """GraphProt .feature file: one line per sequence, 'index:value index:value ...'."""
    with open(path) as f:
        return [line for line in f]


def feature_batch(lines):
    X = np.zeros((len(lines), N_GRAPH), dtype=np.float32)
    for r, line in enumerate(lines):
        for tok in line.split():
            k, v = tok.split(":")
            X[r, int(k)] = float(v)
    return X


def load_model(name):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras
    return keras.models.load_model(MODELS / f"{name}.hdf5", compile=False)


def predict(model, seqs, feature_lines=None, batch=50):
    """Returns (probabilities (n,6)). feature_lines=None -> sequence-only model."""
    probs = []
    for s in range(0, len(seqs), batch):
        xs = encode_batch(seqs[s:s + batch])
        if feature_lines is None:
            p = model.predict(xs, verbose=0)
        else:
            p = model.predict([xs, feature_batch(feature_lines[s:s + batch])], verbose=0)
        probs.append(p)
    return np.vstack(probs)
