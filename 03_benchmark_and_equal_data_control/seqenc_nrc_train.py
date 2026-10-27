"""
SeqEnc-on-nRC - Step 1: train MncR's sequence-only network (SeqEnc) from scratch on the nRC
training set, so it can be compared with RNAErnie and ncRDense on EQUAL training data.

What is kept from MncR's SeqEnc (read from the authors' seqenc_fold8.hdf5):
  embedding(17 letters -> 4) -> [Conv64 k3 x2, pool4] -> [Conv64 k7 x2, pool4]
  -> [Conv64 k11 x2, pool2] -> [Conv64 k17 x2, pool2] -> dropout 0.5 -> flatten -> Dense10 -> softmax
  and the same letter encoding (DNA letters, alphabetical integer codes, "_" padding).
What is different (and must be said in the report):
  * trained on nRC (6,320 sequences, 13 classes), not the authors' ~60,000 RNAcentral sequences, so the
    output layer has 13 units, not 6;
  * sequences are padded/truncated to MAXLEN (default 1000) instead of 12,000, to be feasible on a CPU
    (nRC sequences are short; the few longer ones are cut, as ncRDense does at 750);
  * optimiser settings are ours (Adam 1e-3, batch 64, early stopping): the authors' are not in the model file.

Same 90/10 train/validation split as ncRDense (seed 0). Resumable: if you stop it (Ctrl+C or shutdown),
run the same command again and it continues from the last finished epoch.

Needs: tensorflow, pandas, numpy (already in mini_project).
Run:   python seqenc_nrc_train.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils

BASE = Path(os.environ.get("MINI_PROJECT", "/home/keerthana/Documents/mini_project/2_project"))
NCRDENSE = Path(os.environ.get("NCRDENSE_DATA", BASE / "mirna" / "ncrdense" / "data"))
OUT = Path(os.environ.get("SEQENC_NRC_DIR", BASE / "mirna" / "seqenc_nrc"))
MAXLEN = int(os.environ.get("MAXLEN", 1000))
EPOCHS = int(os.environ.get("EPOCHS", 60))
PATIENCE = 8
BATCH = 64
SEED = int(os.environ.get("SEED", 1))
tf.keras.utils.set_random_seed(SEED)

# letter coding identical to MncR (alphabetical ordinal; U is written as T; unknown -> 16)
_CATS = sorted(["A", "C", "G", "T", "N", "R", "K", "S", "Y", "M", "W", "B", "H", "D", "V", "_"])
_LUT = np.full(256, 16, dtype=np.int16)
for _i, _c in enumerate(_CATS):
    _LUT[ord(_c)] = _i
PAD = _CATS.index("_")


def encode(seqs, maxlen=MAXLEN):
    out = np.full((len(seqs), maxlen), PAD, dtype=np.int16)
    for i, s in enumerate(seqs):
        raw = np.frombuffer(s.upper().replace("U", "T")[:maxlen].encode("ascii", "replace"), dtype=np.uint8)
        out[i, :len(raw)] = _LUT[raw]
    return out


def build_model(n_classes, maxlen=MAXLEN):
    # He-normal weights and a wider embedding start (-1..1) keep the signal alive through the 8 conv layers;
    # the first run (default start values) got stuck predicting one class. Layers/sizes are unchanged.
    he = dict(kernel_initializer="he_normal")
    m = models.Sequential([layers.Input(shape=(maxlen,), dtype="int32"),
                           layers.Embedding(17, 4, embeddings_initializer=tf.keras.initializers.RandomUniform(-1, 1))])
    for k, pool in [(3, 4), (7, 4), (11, 2), (17, 2)]:
        m.add(layers.Conv1D(64, k, padding="same", activation="relu", **he))
        m.add(layers.Conv1D(64, k, padding="same", activation="relu", **he))
        m.add(layers.MaxPooling1D(pool))
    m.add(layers.Dropout(0.5))
    m.add(layers.Flatten())
    m.add(layers.Dense(10, activation="relu", **he))
    m.add(layers.Dense(n_classes, activation="softmax"))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(NCRDENSE / "nrc_train_structure.csv")
    classes = sorted(df["label"].unique())
    pd.Series(classes).to_csv(OUT / "classes.csv", index=False, header=["label"])
    X = encode(df["seq"].tolist())
    y = utils.to_categorical(df["label"].map({c: i for i, c in enumerate(classes)}), len(classes))
    print(f"train file: {len(df)} sequences, {len(classes)} classes, max length kept {MAXLEN} nt "
          f"({(df['seq'].str.len() > MAXLEN).sum()} sequences cut)")

    n_val = max(1, int(0.1 * len(df)))
    rng = np.random.default_rng(0)
    val_idx = rng.choice(len(df), n_val, replace=False)
    tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)

    model = build_model(len(classes))
    model.summary()
    class StuckGuard(callbacks.Callback):
        """Stops early (after 5 epochs) if the model is still at chance, so no time is wasted."""
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 4 and logs.get("val_accuracy", 1) < 0.15:
                print("\n*** Still at chance after 5 epochs - stopping. Run again with another seed: "
                      "SEED=2 python seqenc_nrc_train.py  (delete the old output first, see the instructions) ***")
                self.model.stop_training = True

    cbs = [callbacks.BackupAndRestore(str(OUT / "train_backup")), StuckGuard(),
           callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
                                   start_from_epoch=5),
           callbacks.ModelCheckpoint(str(OUT / "seqenc_nrc_best.keras"), monitor="val_loss", save_best_only=True)]
    model.fit(X[tr_idx], y[tr_idx], validation_data=(X[val_idx], y[val_idx]),
              epochs=EPOCHS, batch_size=BATCH, callbacks=cbs, verbose=2)
    model.save(OUT / "seqenc_nrc_final.keras")
    import shutil
    shutil.rmtree(OUT / "train_backup", ignore_errors=True)
    print(f"\nSaved to {OUT}: seqenc_nrc_best.keras, seqenc_nrc_final.keras, classes.csv")
    print("Next: python seqenc_nrc_evaluate.py")


if __name__ == "__main__":
    main()
