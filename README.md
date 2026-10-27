# Comparison of RNAErnie, ncRDense and MncR for non-coding RNA classification on real GEO datasets

Scripts for the mini-project that compares three ncRNA classifiers on a shared reference benchmark and on public GEO datasets:

| Model | How it was used here |
|---|---|
| RNAErnie | Frozen pre-trained model (`multimolecule/rnaernie`) with a logistic-regression classifier trained on the nRC training set |
| ncRDense | Re-built in Keras (reduced CPU configuration) and trained on the nRC training set |
| MncR | The authors' pre-trained models (SeqEnc alone and the full model with GraphProt graph features) |

An equal-training-data control retrains the MncR SeqEnc network on nRC only.

**Datasets (GEO):** GSE17681 (miRNA), GSE209924 (snoRNA and a pool of six RNA types), GSE229667 (tRNA), GSE310677 (pool of six RNA types). GEO does not provide sequences, so sequences come from Ensembl release 116 (GRCh38 ncRNA FASTA), GtRNAdb (hg38), miRBase and the GPL9040 probe table.

## What is and is not in this repository

Only scripts are included. The data, trained models, embeddings and result files are not stored here because of their size. Download the GEO files and the nRC benchmark with the scripts or from their sources, and the MncR files with `02_mncr_pipeline/mncr_step1_download.py` (authors' repository: github.com/Stegobully/merged_ncRNAclassifier).

The scripts were written for one project folder. Each script has a `BASE` (or `MINI_PROJECT`) path near the top, with a default of `/home/keerthana/Documents/mini_project/2_project`. Set the environment variable `MINI_PROJECT=/your/path` or edit that line before running.

## Folders and run order

Within each folder run the scripts in the order of their step numbers.

| Folder | Content |
|---|---|
| `01_nrc_models` | nRC download, RNAErnie embeddings and classifier (steps 7–9), ncRDense structure prediction, encoding, training and evaluation, and RNAErnie embedding of FASTA files |
| `02_mncr_pipeline` | MncR download, self-test, SeqEnc, GraphProt graph features (separate environment) and the full model |
| `03_benchmark_and_equal_data_control` | The 1,125-sequence benchmark comparison, McNemar tests and figure, and the equal-training-data control (`seqenc_nrc_train.py`, then `seqenc_nrc_evaluate.py`) |
| `04_gse17681_mirna` | GPL9040 probes, mature and precursor miRNA sequences, model runs, analysis and McNemar tests |
| `05_gse209924_snorna_and_pool` | snoRNA-only test and the pooled six-type test for GSE209924 (steps 1–3 each) |
| `06_gse229667_trna` | tRNA test (steps 1–3) |
| `07_gse310677_pool` | Pooled six-type test for GSE310677 (steps 1–3) |

For each dataset, step 1 prepares the sequences, step 2 runs the three models, and step 3 analyses the predictions (recall, confusion matrices, McNemar tests with Holm correction, bootstrap intervals).

## Software

See `requirements.txt` for the versions used (Python 3.10.21, CPU only). GraphProt, RNAshapes and EDeN 0.4.2 were run in a separate conda environment, and the graph step was run in chunks of at most 30,000 nt (`MAX_NT` environment variable).

## Notes

- RNAErnie was not fine-tuned, ncRDense is a reduced re-implementation, and MncR used the authors' weights, so the set-ups are not identical (see the manuscript).
- Overlap with the nRC sets was checked by exact sequence match; overlap with the MncR training data was not assessed.
- No licence has been chosen yet.
