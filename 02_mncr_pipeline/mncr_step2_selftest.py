"""
MncR - Step 2: self-test. Runs both pretrained models on the authors' 180-sequence test
set (which comes with its own GraphProt features) and checks you get ~97% accuracy,
as in the paper. If this passes, your TensorFlow/Keras setup and the sequence
preprocessing are correct, so any later problem is in the data, not the setup.

Run: python mncr_step2_selftest.py
"""
import numpy as np
import mncr_common as c


def main():
    ids, labs, seqs = c.read_fasta(c.DATA / "small_testset_30.fasta")
    truth = [l.strip() for l in open(c.DATA / "small_testset_30_labels.txt")]
    lines = c.read_feature_lines(c.DATA / "small_testset_30_graphprot.feature")
    assert len(seqs) == len(truth) == len(lines), "file lengths do not match"

    for name, feats in [("seqenc_fold8", None), ("mncr_fold7", lines)]:
        model = c.load_model(name)
        probs = c.predict(model, seqs, feats)
        pred = [c.CLASSES[i] for i in probs.argmax(1)]
        acc = np.mean([a == b for a, b in zip(pred, truth)])
        print(f"{name}: accuracy on {len(seqs)} test sequences = {acc:.3f}  (paper: >0.97)")
        if acc < 0.9:
            print("  -> too low; send me this output")


if __name__ == "__main__":
    main()
