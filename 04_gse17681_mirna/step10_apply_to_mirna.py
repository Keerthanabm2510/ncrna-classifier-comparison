"""
Step 10: apply the nRC-trained classifier to your GSE17681 miRNAs
(866 mature probes and 674 precursors) and see what class RNAErnie calls them.

Note: nRC's miRNA class is made of precursor hairpins, so the precursor result is the
fair test; 22-nt mature sequences are outside what the classifier was trained on.
Run: python step10_apply_to_mirna.py
"""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DATA = BASE / "data"
NRC, EMB = DATA / "nrc", DATA / "embeddings"

clf = joblib.load(NRC / "rnaernie_nrc_classifier.joblib")
classes = list(clf.classes_)
mirna_label = next((c for c in classes if c.lower() == "mirna"), None) \
    or next((c for c in classes if "mirna" in c.lower()), None)
print("classes:", classes, "| miRNA label:", mirna_label)

for name in ["mirna_precursors", "mirna_mature_probes"]:
    X = np.load(EMB / f"{name}_rnaernie.npy")
    ids = pd.read_csv(EMB / f"{name}_ids.csv")["id"]
    proba = clf.predict_proba(X)
    out = pd.DataFrame({"id": ids, "predicted_class": np.array(classes)[proba.argmax(1)],
                        "confidence": proba.max(1).round(3)})
    if mirna_label:
        out["p_miRNA"] = proba[:, classes.index(mirna_label)].round(3)
    out.to_csv(DATA / f"{name}_predictions.csv", index=False)
    print(f"\n=== {name} ({len(out)} sequences) ===")
    print(out.predicted_class.value_counts().to_string())
    if mirna_label:
        print(f"called {mirna_label}: {(out.predicted_class == mirna_label).mean():.1%}")
    print(f"saved data/{name}_predictions.csv")
