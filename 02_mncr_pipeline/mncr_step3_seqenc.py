"""
MncR - Step 3: run the authors' pretrained SEQUENCE-ONLY model (SeqEnc) - no extra tools needed.

  A) Rfam benchmark (1,125 sequences, 4 classes: miRNA, rRNA, snoRNA, tRNA) - the same subset
     the MncR paper used to compare against ncRDense. Gives accuracy / F1 / MCC.
  B) Your GSE17681 sets: % of precursors / mature probes called miRNA.

MncR's classes are lncRNA, miRNA, rRNA, snRNA, snoRNA, tRNA. nRC's 13 classes are not all present,
so we score on the 4 classes both benchmarks share; a prediction of lncRNA/snRNA counts as wrong.

Run: python mncr_step3_seqenc.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import mncr_common as c

FOUR = ["miRNA", "rRNA", "snoRNA", "tRNA"]


def score_rfam(ids, truth, pred, tag):
    p, r, f1, _ = precision_recall_fscore_support(truth, pred, labels=FOUR, average="macro", zero_division=0)
    print(f"\n=== Rfam benchmark ({len(truth)} seqs, 4 classes), {tag} ===")
    print(f"Accuracy {accuracy_score(truth, pred):.4f} | Recall {r:.4f} | Precision {p:.4f} | "
          f"F1 {f1:.4f} | MCC {matthews_corrcoef(truth, pred):.4f}")
    print(classification_report(truth, pred, labels=FOUR, digits=3, zero_division=0))
    cm = confusion_matrix(truth, pred, labels=c.CLASSES)
    pd.DataFrame(cm, index=[f"true_{x}" for x in c.CLASSES], columns=[f"pred_{x}" for x in c.CLASSES]
                 ).to_csv(c.DATA / f"mncr_{tag}_rfam_confusion_matrix.csv")


def main():
    model = c.load_model("seqenc_fold8")

    ids, truth, seqs = c.read_fasta(c.DATA / "rfam_testset_used_for_ncrdense.fasta")
    probs = c.predict(model, seqs)
    pred = [c.CLASSES[i] for i in probs.argmax(1)]
    score_rfam(ids, truth, pred, "seqenc")
    pd.DataFrame({"id": ids, "true": truth, "predicted_class": pred, "confidence": probs.max(1).round(3)}
                 ).to_csv(c.DATA / "rfam_benchmark_seqenc_predictions.csv", index=False)

    for name in ["mirna_precursors", "mirna_mature_probes"]:
        path = c.DATA / f"{name}.fasta"
        if not path.exists():
            print(f"SKIP {name}: run step 1 first"); continue
        ids, _, seqs = c.read_fasta(path)
        probs = c.predict(model, seqs)
        out = pd.DataFrame({"id": ids, "predicted_class": [c.CLASSES[i] for i in probs.argmax(1)],
                            "confidence": probs.max(1).round(3),
                            "p_miRNA": probs[:, c.CLASSES.index("miRNA")].round(3)})
        out.to_csv(c.DATA / f"{name}_seqenc_predictions.csv", index=False)
        print(f"\n=== {name} ({len(out)} sequences), SeqEnc ===")
        print(out.predicted_class.value_counts().to_string())
        print(f"called miRNA: {(out.predicted_class == 'miRNA').mean():.1%}")


if __name__ == "__main__":
    main()
