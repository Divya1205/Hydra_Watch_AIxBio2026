#!/usr/bin/env python3
"""
prep_for_embedding.py

Takes Kraken2 outputs and prepares clean FASTA files for DNABERT-2.

What it does:
  1. Reads Kraken2 .out file to find which reads were classified as human (taxon 9606)
  2. Removes human-classified reads from the "classified" pile
  3. Subsamples non-human classified reads to ~50k (enough for Isolation Forest)
  4. Keeps ALL unclassified reads (these are anomaly candidates)
  5. Writes clean FASTA files ready for embedding

Usage:
  python prep_for_embedding.py SRR37006657
"""

import sys
import gzip
import random
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python prep_for_embedding.py <SRR_accession>")
    sys.exit(1)

ACC = sys.argv[1]
SUBSAMPLE_N = 50_000
HUMAN_TAXID = "9606"
random.seed(42)

# ── Step 1: Read Kraken2 output to identify human-classified reads
print(f"Reading Kraken2 output for {ACC}...")
kraken_out = f"{ACC}_kraken2.out"
human_read_ids = set()
n_classified = 0

with open(kraken_out) as f:
    for line in f:
        parts = line.strip().split("\t")
        if parts[0] == "C":  # C = Classified, U = Unclassified
            n_classified += 1
            taxid = parts[2]
            read_id = parts[1]
            if taxid == HUMAN_TAXID:
                human_read_ids.add(read_id)

print(f"  Total classified reads: {n_classified:,}")
print(f"  Human-classified reads: {len(human_read_ids):,} ({len(human_read_ids)/n_classified*100:.1f}%)")
print(f"  Non-human classified:   {n_classified - len(human_read_ids):,}")


def fastq_iter(path):
    """Yield (read_id, sequence) from FASTQ (handles .gz too)."""
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode) as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()  # +
            f.readline()  # quality
            read_id = header.lstrip("@").split()[0]
            yield read_id, seq


# ── Step 2: Process classified reads — exclude human, subsample
print(f"\nProcessing classified reads...")
classified_fq = f"{ACC}_classified_1.fastq"
non_human = []
for rid, seq in fastq_iter(classified_fq):
    if rid not in human_read_ids:
        non_human.append((rid, seq))

print(f"  Non-human classified reads available: {len(non_human):,}")

if len(non_human) > SUBSAMPLE_N:
    sample = random.sample(non_human, SUBSAMPLE_N)
    print(f"  Subsampled to {SUBSAMPLE_N:,}")
else:
    sample = non_human
    print(f"  Using all {len(sample):,} (fewer than {SUBSAMPLE_N:,})")

with open(f"{ACC}_classified_for_embedding.fasta", "w") as f:
    for rid, seq in sample:
        f.write(f">{rid}\n{seq}\n")
print(f"  Wrote {ACC}_classified_for_embedding.fasta")


# ── Step 3: Process unclassified reads — keep ALL
print(f"\nProcessing unclassified reads...")
unclassified_fq = f"{ACC}_unclassified_1.fastq"
n_ucl = 0
with open(f"{ACC}_unclassified_for_embedding.fasta", "w") as f:
    for rid, seq in fastq_iter(unclassified_fq):
        f.write(f">{rid}\n{seq}\n")
        n_ucl += 1

print(f"  Wrote {ACC}_unclassified_for_embedding.fasta  ({n_ucl:,} reads)")

print(f"\n=== Done ===")
print(f"Ready for embedding:")
print(f"  Classified (training):    {ACC}_classified_for_embedding.fasta")
print(f"  Unclassified (anomalies): {ACC}_unclassified_for_embedding.fasta")
