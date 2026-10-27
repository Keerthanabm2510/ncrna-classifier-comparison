"""
MncR - Step 5: run the full MncR model (sequence + GraphProt graph features) next to SeqEnc.

Needs the .feature files from Step 4. Sets handled (any that are missing are skipped):
  rfam_benchmark (or rfam_benchmark_subset)  -> accuracy / F1 / MCC on the 4 shared classes
  mirna_precursors, mirna_mature_probes      -> % called miRNA

Run: python mncr_step5_mncr.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import mncr_common as c

FOUR = ["miRNA", "rRNA", "snoRNA", "tRNA"]


def main():
    mncr, seqenc = c.load_model("mncr_fold7"), c.load_model("seqenc_fold8")
    for name in ["rfam_benchmark_subset", "rfam_benchmark", "mirna_precursors", "mirna_mature_probes"]:
        fasta, feat = c.DATA / f"{name}.fasta", c.DATA / f"{name}.feature"
        if name == "rfam_benchmark":
            fasta = c.DATA / "rfam_testset_used_for_ncrdense.fasta"
        if not (fasta.exists() and feat.exists()):
            print(f"SKIP {name}: no {feat.name} yet (Step 4)"); continue
        ids, truth, seqs = c.read_fasta(fasta)
        lines = c.read_feature_lines(feat)
        assert len(lines) == len(seqs), f"{name}: {len(lines)} feature lines vs {len(seqs)} sequences"

        p_m = c.predict(mncr, seqs, lines)
        p_s = c.predict(seqenc, seqs)
        out = pd.DataFrame({"id": ids,
                            "mncr_class": [c.CLASSES[i] for i in p_m.argmax(1)], "mncr_confidence": p_m.max(1).round(3),
                            "seqenc_class": [c.CLASSES[i] for i in p_s.argmax(1)]})
        out["p_miRNA"] = p_m[:, c.CLASSES.index("miRNA")].round(3)
        if all(t for t in truth):
            out.insert(1, "true", truth)
        out.to_csv(c.DATA / f"{name}_mncr_predictions.csv", index=False)

        print(f"\n=== {name} ({len(out)} sequences) ===")
        if "true" in out and set(truth) <= set(FOUR):
            for tag, pred in [("MncR (sequence + graph)", out.mncr_class), ("SeqEnc (sequence only)", out.seqenc_class)]:
                p, r, f1, _ = precision_recall_fscore_support(truth, pred, labels=FOUR, average="macro", zero_division=0)
                print(f"{tag}: Accuracy {accuracy_score(truth, pred):.4f} | Recall {r:.4f} | Precision {p:.4f} | "
                      f"F1 {f1:.4f} | MCC {matthews_corrcoef(truth, pred):.4f}")
            print(classification_report(truth, out.mncr_class, labels=FOUR, digits=3, zero_division=0))
            cm = confusion_matrix(truth, out.mncr_class, labels=c.CLASSES)
            pd.DataFrame(cm, index=[f"true_{x}" for x in c.CLASSES], columns=[f"pred_{x}" for x in c.CLASSES]
                         ).to_csv(c.DATA / f"mncr_{name}_confusion_matrix.csv")
        else:
            print("MncR predictions:"); print(out.mncr_class.value_counts().to_string())
            print(f"called miRNA: MncR {(out.mncr_class == 'miRNA').mean():.1%} | SeqEnc {(out.seqenc_class == 'miRNA').mean():.1%}")


if __name__ == "__main__":
    main()
