#!/usr/bin/env python3
"""
extract_clusters_for_blast.py

Extract top sequences from priority clusters (6, 4, 3) for NCBI web BLAST.

Run from the directory containing your FASTAs and results_tgvae/.

Output: One FASTA file per cluster, ready to paste into web BLAST.

Usage:
  python extract_clusters_for_blast.py
  python extract_clusters_for_blast.py --clusters 6 4 3 --top-n 5
"""

import argparse
from pathlib import Path
import pandas as pd
import os

# ── Path config — set HYDRAWATCH_ROOT env var, or override defaults below
PROJECT_ROOT = Path(os.environ.get("HYDRAWATCH_ROOT", "."))

EMBED_DIR = PROJECT_ROOT / "casper_data/ny_hospital_d/embeddings/25k/embeddings_combined"
RESULTS_DIR = PROJECT_ROOT / "results_tgvae"
CLUSTERS_TSV = RESULTS_DIR / "tgvae_top_anomalies_with_clusters.tsv"


# Map sample accession → FASTA file with the actual sequences
FASTA_MAP = {
    "SRR37006657": "SRR37006657_unclassified_250k.fasta",
    "SRR37006671": "SRR37006671_unclassified_250k.fasta",
    "SRR37006667": "SRR37006667_unclassified_250k.fasta",
}


def load_fasta(path):
    """Read a FASTA file into {read_id: sequence} dict."""
    seqs = {}
    rid, parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if rid is not None:
                    seqs[rid] = "".join(parts)
                rid = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if rid is not None:
            seqs[rid] = "".join(parts)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", nargs="+", type=int, default=[6, 4, 3],
                    help="Cluster IDs to extract (default: 6 4 3)")
    ap.add_argument("--top-n", type=int, default=5,
                    help="Top N reads per cluster (default: 5)")
    args = ap.parse_args()

    print(f"Loading {CLUSTERS_TSV}...")
    df = pd.read_csv(CLUSTERS_TSV, sep="\t")
    print(f"  Total reads: {len(df):,}")
    print(f"  Clusters present: {sorted(df['cluster'].unique())}")

    # Cache loaded FASTAs (don't re-read for every cluster)
    fasta_cache = {}

    # Combined output for one-shot BLAST
    combined_path = RESULTS_DIR / "all_priority_clusters_for_blast.fasta"
    combined_records = []

    for cluster_id in args.clusters:
        cdf = df[df["cluster"] == cluster_id].copy()
        if len(cdf) == 0:
            print(f"\nCluster {cluster_id}: NOT FOUND, skipping")
            continue

        cdf = cdf.sort_values("tgvae_error", ascending=False).head(args.top_n)
        print(f"\nCluster {cluster_id}: {len(cdf)} top reads")
        print(cdf[["sample", "timepoint", "read_id", "tgvae_error"]].to_string(index=False))

        # Load needed FASTAs lazily
        records = []
        for _, row in cdf.iterrows():
            sample = row["sample"]
            if sample not in fasta_cache:
                fa_path = Path(FASTA_MAP[sample])
                if not fa_path.exists():
                    print(f"  WARNING: {fa_path} not found, skipping {sample} reads")
                    fasta_cache[sample] = {}
                    continue
                print(f"  Loading {fa_path}...")
                fasta_cache[sample] = load_fasta(fa_path)

            seq = fasta_cache[sample].get(row["read_id"])
            if seq is None:
                print(f"  WARNING: {row['read_id']} not found in {sample} FASTA")
                continue

            header = f">{row['timepoint']}_cluster{cluster_id}_score{row['tgvae_error']:.1f}_{row['read_id']}"
            records.append(f"{header}\n{seq}")
            combined_records.append(f"{header}\n{seq}")

        # Per-cluster FASTA
        out_path = RESULTS_DIR / f"cluster{cluster_id}_for_blast.fasta"
        with open(out_path, "w") as f:
            f.write("\n".join(records) + "\n")
        print(f"  → {out_path}")

    # Combined FASTA (all priority clusters in one file)
    with open(combined_path, "w") as f:
        f.write("\n".join(combined_records) + "\n")
    print(f"\n→ Combined: {combined_path}  ({len(combined_records)} sequences)")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"1. Open: {combined_path}")
    print("2. Copy ALL contents")
    print("3. Go to: https://blast.ncbi.nlm.nih.gov/Blast.cgi")
    print("4. Click 'blastn' → paste into query box")
    print("5. Database: 'nt' (default), click BLAST")
    print("6. Wait 30-90 sec, then look at top hit per query")
    print()
    print("What to look for:")
    print("  - % Identity ≥ 70%, E-value ≤ 1e-5 → confident hit")
    print("  - Match a CASPER pathogen → validation (bucket A)")
    print("  - No good hit → novel candidate (bucket C, the discovery story)")


if __name__ == "__main__":
    main()
