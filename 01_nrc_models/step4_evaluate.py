"""
ncRDense - Step 4: evaluate the trained model on the nRC test set.

Removes test sequences that are identical to a training sequence first
(same reasoning as the RNAErnie pipeline - avoids inflated scores from
sequences the model has effectively already seen).

Needs: pip install tensorflow pandas numpy scikit-learn
Run:   python step4_evaluate.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils
from sklearn.metrics import (accuracy_score, matthews_corrcoef,
                              precision_recall_fscore_support, confusion_matrix,
                              classification_report)

DATA = Path(os.environ.get(
    "NCRDENSE_DATA", "/home/keerthana/Documents/mini_project/2_project/mirna/ncrdense/data"))
N_SEQ_VOCAB, N_STRUCT_VOCAB = 5, 4


def to_onehot(codes, depth, keep):
    return tf.one_hot(codes, depth=depth).numpy()[:, :, :keep]


def main():
    train_struct = pd.read_csv(DATA / "nrc_train_structure.csv")
    test_struct = pd.read_csv(DATA / "nrc_test_structure.csv")
    dup_mask = test_struct["seq"].isin(set(train_struct["seq"])).values
    print(f"test sequences identical to a training sequence: {dup_mask.sum()} -> removed")

    seq_codes = np.load(DATA / "nrc_test_seq_codes.npy")[~dup_mask]
    struct_codes = np.load(DATA / "nrc_test_struct_codes.npy")[~dup_mask]
    meta = pd.read_csv(DATA / "nrc_test_encoded_meta.csv")[~dup_mask].reset_index(drop=True)

    classes = pd.read_csv(DATA / "nrc_classes.csv")["label"].tolist()
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = meta["label"].map(label_to_idx).values

    model = tf.keras.models.load_model(DATA / "ncrdense_best.keras")
    xs = to_onehot(seq_codes, N_SEQ_VOCAB, 4)
    xt = to_onehot(struct_codes, N_STRUCT_VOCAB, 3)
    proba = model.predict([xs, xt], batch_size=128, verbose=1)
    pred_idx = proba.argmax(1)

    y_true = [classes[i] for i in y_true_idx]
    y_pred = [classes[i] for i in pred_idx]

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== nRC test set ({len(meta)} seqs after de-dup), ncRDense (sequence + structure) ===")
    print(f"Accuracy {accuracy_score(y_true, y_pred):.4f} | Recall {r:.4f} | "
          f"Precision {p:.4f} | F1 {f1:.4f} | MCC {matthews_corrcoef(y_true, y_pred):.4f}")
    print("(paper Table 2 for reference: ncRDense 0.9510 acc on the original, non-deduped test set)")
    print("\n" + classification_report(y_true, y_pred, digits=3, zero_division=0))

    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes
                 ).to_csv(DATA / "ncrdense_nrc_confusion_matrix.csv")
    print(f"saved {DATA/'ncrdense_nrc_confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
