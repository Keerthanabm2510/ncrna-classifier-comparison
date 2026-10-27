"""
ncRDense - Step 5: apply the trained model to the GSE17681 miRNAs
(precursors and mature probes) and record predicted class + confidence.

Needs: pip install tensorflow pandas numpy
Run:   python step5_apply_to_mirna.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

DATA = Path(os.environ.get(
    "NCRDENSE_DATA", "/home/keerthana/Documents/mini_project/2_project/mirna/ncrdense/data"))
N_SEQ_VOCAB, N_STRUCT_VOCAB = 5, 4


def to_onehot(codes, depth, keep):
    return tf.one_hot(codes, depth=depth).numpy()[:, :, :keep]


def main():
    classes = pd.read_csv(DATA / "nrc_classes.csv")["label"].tolist()
    model = tf.keras.models.load_model(DATA / "ncrdense_best.keras")

    for name in ["mirna_precursors", "mirna_mature_probes"]:
        seq_codes = np.load(DATA / f"{name}_seq_codes.npy")
        struct_codes = np.load(DATA / f"{name}_struct_codes.npy")
        meta = pd.read_csv(DATA / f"{name}_encoded_meta.csv")

        xs = to_onehot(seq_codes, N_SEQ_VOCAB, 4)
        xt = to_onehot(struct_codes, N_STRUCT_VOCAB, 3)
        proba = model.predict([xs, xt], batch_size=128, verbose=0)
        pred_idx = proba.argmax(1)

        out = pd.DataFrame({
            "id": meta["id"],
            "predicted_class": [classes[i] for i in pred_idx],
            "confidence": proba.max(1).round(3),
        })
        if "miRNA" in classes:
            out["p_miRNA"] = proba[:, classes.index("miRNA")].round(3)

        out.to_csv(DATA / f"{name}_ncrdense_predictions.csv", index=False)
        print(f"\n=== {name} ({len(out)} sequences) ===")
        print(out.predicted_class.value_counts().to_string())
        if "miRNA" in classes:
            print(f"called miRNA: {(out.predicted_class == 'miRNA').mean():.1%}")
        print(f"saved data/{name}_ncrdense_predictions.csv")


if __name__ == "__main__":
    main()
