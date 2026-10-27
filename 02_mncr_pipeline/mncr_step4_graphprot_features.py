"""
MncR - Step 4: build the GraphProt graph features that the full MncR model needs.

MncR = sequence CNN + a network on 32,768 graph features of the predicted secondary structure.
Those features come from two external programs (RNAshapes -> fasta2shrep_gspan.pl, then EDeN),
both packaged in the bioconda "graphprot" package. Install in a SEPARATE environment so it
cannot disturb your TensorFlow environment (one-off, a few hundred MB):

    conda create -n graphprot -c conda-forge -c bioconda graphprot

Then, from your normal (mini_project) environment:

    python mncr_step4_graphprot_features.py --selftest     # recompute features for the 180 test seqs and
                                                           # compare with the authors' file (expect ~1.00)
    python mncr_step4_graphprot_features.py mirna_precursors
    python mncr_step4_graphprot_features.py mirna_mature_probes
    python mncr_step4_graphprot_features.py rfam_benchmark   # slow; see N_EVAL below

Output: data/<name>.feature (one line per sequence, same order as the FASTA).
Speed: GraphProt is slow on long sequences. Short hairpins (<150 nt) are fine; for the Rfam
benchmark you can score a random subset: N_EVAL=300 python mncr_step4_graphprot_features.py rfam_benchmark

NOTE: this step could not be run in the sandbox where the other scripts were tested
(RNAshapes is not installable there), so run --selftest first and send me the output.
The self-test uses short sequences only (the authors' full 180-sequence test set has ones up to 4,680 nt).
"""
import hashlib
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import mncr_common as c

ENV = os.environ.get("GRAPHPROT_ENV", "graphprot")
SOURCES = {
    "mirna_precursors": "mirna_precursors.fasta",
    "mirna_mature_probes": "mirna_mature_probes.fasta",
    "rfam_benchmark": "rfam_testset_used_for_ncrdense.fasta",
}


def _conda():
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not conda:
        sys.exit("conda not found on PATH")
    return conda


def conda_capture(cmd):
    """Run inside the graphprot env and return the output (used only for short lookups)."""
    return subprocess.run([_conda(), "run", "-n", ENV, "bash", "-c", cmd], capture_output=True, text=True)


def conda_run(cmd):
    """Run inside the graphprot env with LIVE output in your terminal (--no-capture-output),
    so you can see that it is working instead of staring at a silent screen."""
    return subprocess.run([_conda(), "run", "--no-capture-output", "-n", ENV, "bash", "-c", cmd])


def graphprot_dir():
    r = conda_capture('echo "$CONDA_PREFIX/libexec/graphprot"')
    d = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    if not d or not Path(d, "fasta2shrep_gspan.pl").exists():
        sys.exit(f"GraphProt not found in conda env '{ENV}' (looked in {d!r}). Install it first, see the top of this file.")
    return d


def make_features(ids, seqs, workdir, tag):
    gp = graphprot_dir()
    workdir.mkdir(parents=True, exist_ok=True)
    text = "".join(f">{i}\n{s.upper().replace('U', 'T')}\n" for i, s in zip(ids, seqs))   # DNA letters, as the authors used
    tag = f"{tag}_{hashlib.md5(text.encode()).hexdigest()[:8]}"       # same input -> same files -> reused, not recomputed
    fa, gspan = workdir / f"{tag}.fa", workdir / f"{tag}.gspan.gz"
    feat = Path(str(gspan) + ".feature")
    fa.write_text(text)
    longest = max(len(s) for s in seqs)
    print(f"{len(seqs)} sequences, longest {longest} nt (GraphProt time grows quickly with length)")

    if feat.exists() and feat.stat().st_size > 0:
        print(f"reusing existing {feat.name}")
    else:
        if not (gspan.exists() and gspan.stat().st_size > 100):
            t = time.time(); print("[1/2] RNAshapes structures -> graph (fasta2shrep_gspan.pl) ...", flush=True)
            r = conda_run(f"'{gp}/fasta2shrep_gspan.pl' -abstr -stdout -M 3 -wins '150,' -shift '25' "
                          f"-fasta '{fa}' -t 3 | gzip > '{gspan}'")
            if r.returncode != 0:
                gspan.unlink(missing_ok=True); sys.exit("fasta2shrep_gspan.pl failed (see messages above)")
            print(f"      done in {time.time() - t:.0f}s", flush=True)
        else:
            print(f"[1/2] reusing existing {gspan.name}")
        t = time.time(); print("[2/2] graph features (EDeN) ... this can take several minutes; Ctrl+C is safe, "
                              "step [1/2] is saved and will be reused", flush=True)
        r = conda_run(f"'{gp}/EDeN' -a FEATURE -i '{gspan}'")
        if not feat.exists():
            sys.exit("EDeN did not create the .feature file (see messages above)")
        print(f"      done in {time.time() - t:.0f}s", flush=True)
    n = sum(1 for _ in open(feat))
    if n != len(seqs):
        sys.exit(f"{n} feature lines but {len(seqs)} sequences - GraphProt dropped some; cannot continue")
    return feat


def selftest():
    """Recompute features for a few SHORT sequences from the authors' test set and compare with their file.
    (The full 180-sequence set contains sequences up to 4,680 nt, which would take very long.)"""
    ids, _, seqs = c.read_fasta(c.DATA / "small_testset_30.fasta")
    n = int(os.environ.get("SELFTEST_N", 40))
    short = [i for i in range(len(seqs)) if len(seqs[i]) <= 150][:n]
    print(f"self-test on {len(short)} sequences of <=150 nt (from the authors' set)")
    ours = c.read_feature_lines(make_features([ids[i] for i in short], [seqs[i] for i in short],
                                              c.DATA / "graphprot_tmp", "selftest"))
    all_theirs = c.read_feature_lines(c.DATA / "small_testset_30_graphprot.feature")
    theirs = [all_theirs[i] for i in short]
    cos = []
    for a, b in zip(c.feature_batch(ours), c.feature_batch(theirs)):
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    cos = np.array(cos)
    print(f"cosine similarity with the authors' features: mean {cos.mean():.4f}, min {cos.min():.4f} over {len(cos)} sequences")
    print("OK - your GraphProt install reproduces the authors' features" if cos.mean() > 0.99
          else "Different from the authors' features - send me this output before using it")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        return selftest()
    name = sys.argv[1]
    ids, labs, seqs = c.read_fasta(c.DATA / SOURCES[name])
    n_eval = int(os.environ.get("N_EVAL", 0))
    if n_eval and n_eval < len(ids):
        random.seed(0)
        keep = sorted(random.sample(range(len(ids)), n_eval))
        ids, labs, seqs = [ids[i] for i in keep], [labs[i] for i in keep], [seqs[i] for i in keep]
        name = f"{name}_subset"
        with open(c.DATA / f"{name}.fasta", "w") as f:                  # keeps the labels for scoring later
            for i, l, s in zip(ids, labs, seqs):
                f.write(f">{i} {l}\n{s}\n")
        print(f"using a random subset of {n_eval} sequences -> {name}.fasta")
    feat = make_features(ids, seqs, c.DATA / "graphprot_tmp", name)
    shutil.move(str(feat), c.DATA / f"{name}.feature")
    print(f"saved {c.DATA / (name + '.feature')} ({len(ids)} sequences)")


if __name__ == "__main__":
    main()
