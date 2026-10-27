"""
ncRDense - Step 3: build and train the two-branch DenseNet-style model
(sequence branch + secondary-structure branch), following the paper's
architecture (Methods, Fig. 5 / Table 4):
  - each branch: Conv1D(64, k=5) -> BN -> ReLU -> 2 dense blocks (4 conv
    layers of 128 filters, k=5 each; first layer's output is skip-connected
    into each later layer in the block) -> Conv1D(64, k=5)
  - branches concatenated -> Conv1D(64, k=3) -> Flatten
  - FC(256) -> FC(64) -> FC(13, softmax)
  - MaxPool size 4 inside dense blocks, size 2 elsewhere
  - Adam(lr=5e-4), categorical cross-entropy, batch 128, up to 200 epochs,
    early stopping (patience 25)

One-hot encoding of the int8 codes from Step 2 happens on the fly per batch
(keras Sequence), so nothing large is written to disk beyond the small
integer arrays Step 2 already saved.

Needs: pip install tensorflow pandas numpy
Run:   python step3_train.py
NOTE: this can take a while per epoch on CPU (no GPU). It saves the best
checkpoint as it goes, so an interrupted run is not wasted.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils

DATA = Path(os.environ.get(
    "NCRDENSE_DATA", "/home/keerthana/Documents/mini_project/2_project/mirna/ncrdense/data"))
MAXLEN = 750
N_SEQ_VOCAB, N_STRUCT_VOCAB = 5, 4     # A,U,G,C,N/pad  |  .,(,),pad  (see step2)
# Lookup tables: row = code, last row (pad/N) = all zeros. Plain NumPy indexing is much
# faster than building a tf.one_hot tensor and converting it back each batch.
SEQ_ONEHOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]], dtype="float32")
STRUCT_ONEHOT = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]], dtype="float32")
BATCH = 128
# LIGHT SETTINGS for a CPU-only laptop (paper uses FILTERS=128, EPOCHS=200, PATIENCE=25).
# Override without editing:  FILTERS=128 EPOCHS=200 PATIENCE=25 python step3_train_fast.py
FILTERS = int(os.environ.get("FILTERS", 64))
EPOCHS = int(os.environ.get("EPOCHS", 20))
PATIENCE = int(os.environ.get("PATIENCE", 6))
# Dropout inside the conv blocks (feeds into BatchNorm) makes validation scores collapse
# at inference time ("variance shift"), so it is OFF by default. Dropout in the FC layers stays.
DROPOUT_CONV = float(os.environ.get("DROPOUT_CONV", 0.0))
BN_MOMENTUM = float(os.environ.get("BN_MOMENTUM", 0.9))   # Keras default 0.99 lags badly on short runs


class OneHotSequence(utils.Sequence):
    """Feeds (seq_onehot, struct_onehot) -> label, one-hot encoding on the fly."""
    def __init__(self, seq_codes, struct_codes, y, batch_size=BATCH, shuffle=True):
        self.seq_codes, self.struct_codes, self.y = seq_codes, struct_codes, y
        self.batch_size, self.shuffle = batch_size, shuffle
        self.indices = np.arange(len(y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, i):
        idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        # code 4 (seq) / 3 (struct) means "pad or N" -> one-hot of that row is all zeros,
        # matching the paper's zero-vector padding exactly.
        xs = SEQ_ONEHOT[self.seq_codes[idx]]
        xt = STRUCT_ONEHOT[self.struct_codes[idx]]
        return (xs, xt), self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def dense_block(x, filters=FILTERS, kernel=5, n_layers=4, dropout=None, name=""):
    dropout = DROPOUT_CONV if dropout is None else dropout
    """4 conv layers; layer-1's output is skip-connected into every later layer."""
    def conv_unit(inp, i):
        h = layers.Conv1D(filters, kernel, padding="same", name=f"{name}_conv{i}")(inp)
        h = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_bn{i}")(h)
        h = layers.ReLU(name=f"{name}_relu{i}")(h)
        return layers.Dropout(dropout, name=f"{name}_drop{i}")(h) if dropout > 0 else h

    layer1 = conv_unit(x, 1)
    prev = layer1
    for i in range(2, n_layers + 1):
        merged = layers.Concatenate(name=f"{name}_cat{i}")([layer1, prev]) if i > 2 else layer1
        prev = conv_unit(merged, i)
    return layers.MaxPooling1D(4, name=f"{name}_pool")(prev)


def build_branch(input_dim, name):
    inp = layers.Input(shape=(MAXLEN, input_dim), name=f"{name}_input")
    h = layers.Conv1D(64, 5, padding="same", name=f"{name}_stem_conv")(inp)
    h = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_stem_bn")(h)
    h = layers.ReLU(name=f"{name}_stem_relu")(h)
    h = layers.MaxPooling1D(2, name=f"{name}_stem_pool")(h)
    h = dense_block(h, name=f"{name}_db1")
    h = dense_block(h, name=f"{name}_db2")
    h = layers.Conv1D(64, 5, padding="same", name=f"{name}_post_conv")(h)
    h = layers.MaxPooling1D(2, name=f"{name}_post_pool")(h)
    return inp, h


def build_model(n_classes):
    seq_in, seq_feat = build_branch(4, "seq")
    struct_in, struct_feat = build_branch(3, "struct")
    merged = layers.Concatenate(name="branch_concat")([seq_feat, struct_feat])
    h = layers.Conv1D(64, 3, padding="same", name="final_conv")(merged)
    h = layers.MaxPooling1D(2, name="final_pool")(h)
    h = layers.Flatten(name="flatten")(h)
    h = layers.Dropout(0.2)(layers.ReLU()(layers.Dense(256, name="fc1")(h)))
    h = layers.Dropout(0.2)(layers.ReLU()(layers.Dense(64, name="fc2")(h)))
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(h)
    model = models.Model([seq_in, struct_in], out, name="ncRDense")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    seq_codes = np.load(DATA / "nrc_train_seq_codes.npy")
    struct_codes = np.load(DATA / "nrc_train_struct_codes.npy")
    meta = pd.read_csv(DATA / "nrc_train_encoded_meta.csv")

    classes = sorted(meta["label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = utils.to_categorical(meta["label"].map(label_to_idx), num_classes=len(classes))
    pd.Series(classes).to_csv(DATA / "nrc_classes.csv", index=False, header=["label"])
    print(f"train: {len(meta)} sequences, {len(classes)} classes -> {classes}")

    n_val = max(1, int(0.1 * len(meta)))           # simple 90/10 split, matches paper's ratio
    rng = np.random.default_rng(0)
    val_idx = rng.choice(len(meta), n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(meta)), val_idx)

    train_gen = OneHotSequence(seq_codes[train_idx], struct_codes[train_idx], y[train_idx])
    val_gen = OneHotSequence(seq_codes[val_idx], struct_codes[val_idx], y[val_idx], shuffle=False)

    model = build_model(len(classes))
    model.summary()

    ckpt_path = DATA / "ncrdense_best.keras"
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        # If the laptop sleeps / you press Ctrl+C, re-running this script resumes
        # from the last finished epoch instead of starting over.
        callbacks.BackupAndRestore(str(DATA / "train_backup")),
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cbs, verbose=1)
    model.save(DATA / "ncrdense_final.keras")
    import shutil; shutil.rmtree(DATA / "train_backup", ignore_errors=True)   # run finished
    print(f"\nSaved: {ckpt_path} (best) and ncrdense_final.keras (last epoch)")


if __name__ == "__main__":
    main()
