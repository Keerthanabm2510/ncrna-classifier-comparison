"""
Step 9: train a small classifier on the RNAErnie embeddings of nRC train,
then score it on the nRC test set (same metrics as the papers' tables).

RNAErnie stays frozen (like the paper's FBTH setting); only a logistic-regression
head is trained. Needs: pip install scikit-learn joblib
Run: python step9_train_eval_nrc.py
"""
import os
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, classification_report)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)   # harmless scikit-learn version notices

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
DATA = BASE / "data"
NRC, EMB = DATA / "nrc", DATA / "embeddings"

Xtr = np.load(EMB / "nrc_train_rnaernie.npy")
Xte = np.load(EMB / "nrc_test_rnaernie.npy")
ytr = pd.read_csv(NRC / "nrc_train_meta.csv")["label"].values
yte = pd.read_csv(NRC / "nrc_test_meta.csv")["label"].values
assert len(Xtr) == len(ytr) and len(Xte) == len(yte), "embedding/metadata size mismatch - rerun step 8"
print(f"train {Xtr.shape}, test {Xte.shape}")

clf = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(Cs=[0.01, 0.1, 1.0], cv=3, max_iter=2000, n_jobs=-1, random_state=0),
)
print("training (a few minutes on CPU) ...")
clf.fit(Xtr, ytr)
print("chosen C:", clf[-1].C_[0])

pred = clf.predict(Xte)
p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
print("\n=== nRC test set, RNAErnie embeddings (frozen) + logistic regression ===")
print(f"Accuracy {accuracy_score(yte, pred):.4f} | Recall {r:.4f} | Precision {p:.4f} | "
      f"F1 {f1:.4f} | MCC {matthews_corrcoef(yte, pred):.4f}")
print("(paper Table 1 for reference: RNAErnie fine-tuned 0.9638 acc; ncRDense 0.9510; RNA-FM 0.9656)")
print("\n" + classification_report(yte, pred, digits=3, zero_division=0))

labels = sorted(set(ytr))
pd.DataFrame(confusion_matrix(yte, pred, labels=labels), index=labels, columns=labels).to_csv(
    NRC / "nrc_confusion_matrix.csv")
joblib.dump(clf, NRC / "rnaernie_nrc_classifier.joblib")
print(f"saved {NRC/'rnaernie_nrc_classifier.joblib'} and nrc_confusion_matrix.csv")
