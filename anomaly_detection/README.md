# Anomaly Detection

The TE-VAE (Transformer-encoder Variational Autoencoder) anomaly detection stage of the HydraWatch pipeline.

This stage takes DNABERT-2 embeddings (output of preprocessing) and produces:
1. A hybrid anomaly score per read
2. A clustering of the top 1% anomalies pooled across timepoints
3. Trajectory analysis (emerging / transient / declining clusters)
4. Figures for the report and deck
5. FASTA files of representative reads for BLAST validation

---

## Inputs

This stage expects DNABERT-2 embeddings from the preprocessing stage. Either:

- **Per-sample format** — 12 files (3 samples × 4 files) from the Kaggle embedding notebook:
  ```
  <accession>_classified_embeddings.npy
  <accession>_classified_ids.txt
  <accession>_unclassified_embeddings.npy
  <accession>_unclassified_ids.txt
  ```

- **Combined format** (what the scripts actually read):
  ```
  normal.npy           # concatenated classified embeddings, all 3 samples
  all.npy              # concatenated all reads (classified + unclassified)
  all_metadata.tsv     # row-aligned: read_id, sample, kind, timepoint
  ```

If you only have the per-sample files, merge them by stacking the `.npy` arrays in a fixed sample order and writing a metadata TSV with the four columns above.

Default expected location:
```
casper_data/ny_hospital_d/embeddings/25k/embeddings_combined/
```

The scripts also expect the unclassified-pool FASTAs from preprocessing for the BLAST extraction step:
```
SRR37006657_unclassified_250k.fasta
SRR37006671_unclassified_250k.fasta
SRR37006667_unclassified_250k.fasta
```

---

## Scripts in this folder

| Script | Purpose |
|:--|:--|
| `tevae_anomaly_detection_hybrid.py` | Trains TE-VAE on classified pool, scores all reads, saves hybrid scores |
| `tevae_plots.py` | Generates the three deck/report figures (distributions, trajectories, joint UMAP) |
| `replot_tevae_components.py` | Standalone script to regenerate the three-component distribution figure from existing scores (no retraining) |
| `extract_clusters_for_blast.py` | Extracts representative read sequences from priority clusters for NCBI web BLAST |

---

## Pipeline overview

```
DNABERT-2 embeddings (from preprocessing)
   │
   ▼
tevae_anomaly_detection_hybrid.py
   ├── trains TE-VAE on classified pool
   ├── computes reconstruction error per read
   ├── computes latent Mahalanobis distance per read
   └── outputs:
       ├── results_tevae/tevae_scores.tsv        (per-read scores)
       ├── results_tevae/tevae_threshold.txt     (threshold metadata)
       ├── results_tevae/fig_tevae_distribution_components.png
       └── results_tevae/fig_tevae_distribution.png
   │
   ▼
tevae_plots.py
   ├── reads tevae_scores.tsv
   ├── HDBSCAN-clusters top 1% anomalies pooled across timepoints
   └── outputs:
       ├── results_tevae/tevae_cluster_trajectories.tsv  (Table 1 in the report)
       ├── results_tevae/tevae_top_anomalies_with_clusters.tsv
       ├── results_tevae/fig_tevae_score_distributions.png
       ├── results_tevae/fig_tevae_cluster_trajectories.png  (Figure 4)
       └── results_tevae/fig_tevae_joint_umap.png            (Figure 5)
   │
   ▼
extract_clusters_for_blast.py
   └── extracts top-N reads per priority cluster (default: clusters 6, 4, 3; 5 reads each) →
       results_tevae/all_priority_clusters_for_blast.fasta
```

---

## Step 1 — Train TE-VAE and score all reads

```
python anomaly_detection/tevae_anomaly_detection_hybrid.py
```

What happens:
1. Loads embeddings + metadata
2. Standardises with z-scoring on the classified pool only
3. Trains the TE-VAE for 50 epochs (Adam, lr=1e-3, batch_size=256, β=0.1)
4. Computes per-read reconstruction error
5. Computes per-read latent Mahalanobis distance against the classified centroid
6. Combines into hybrid score: robust-z(recon) + robust-z(log(latent_mahal))
7. Sets threshold at μ+3σ on the classified hybrid scores
8. Saves scores + threshold metadata + the three-component distribution figure

Runtime: ~30–45 min on a CPU (Mac M1/M2), ~10 min on a GPU.

**TE-VAE architecture** (matches report §3.3):
- Encoder: linear(768→128) → 2× transformer self-attention blocks (4 heads each) → linear heads to μ, log σ²
- Latent dim: 32
- Decoder: linear → ReLU → linear → 768
- Loss: MSE reconstruction + β·KL with β=0.1 (down-weighted to prioritise reconstruction fidelity)

**Hybrid score** (matches report §3.4):
- Robust z-scored reconstruction error (median + MAD × 1.4826)
- Plus robust z-scored log-transformed latent Mahalanobis (the log compresses the heavy right tail)
- Equal weights, summed

---

## Step 2 — Generate cluster trajectories and figures

```
python anomaly_detection/tevae_plots.py
```

What happens:
1. Reads `tevae_scores.tsv`
2. Selects top 1% by hybrid score, pooled across all three timepoints
3. PCA-projects to 50 components
4. HDBSCAN clusters with min_cluster_size=30
5. Builds the cluster trajectory table (cluster × timepoint counts)
6. Generates Figure 3, Figure 4, and Figure 5 from the report

Runtime: ~3–5 min (UMAP is the slowest step).

**Outputs that go into the report and deck:**
- `tevae_cluster_trajectories.tsv` → Table 1 in the report
- `fig_tevae_score_distributions.png` → Figure 3
- `fig_tevae_cluster_trajectories.png` → Figure 4
- `fig_tevae_joint_umap.png` → Figure 5

---

## Step 3 (optional) — Regenerate the distribution figure

If you want to re-render the three-component figure (Figure 2 in the report) without retraining:

```
python anomaly_detection/replot_tevae_components.py
```

Reads `tevae_scores.tsv` and re-renders both `fig_tevae_distribution_components.png` (3-panel) and `fig_tevae_distribution.png` (single hybrid panel) at 300 DPI for report/deck use.

Use this when:
- You want to tweak figure formatting (colors, sizes, axis limits)
- You don't want to wait for a full retraining run
- You need the figure at a different DPI

---

## Step 4 — Extract sequences for BLAST validation

```
python anomaly_detection/extract_clusters_for_blast.py \
  --fasta-dir /path/to/unclassified/fastas
```

What happens:
1. Reads `tevae_top_anomalies_with_clusters.tsv`
2. For each priority cluster (default: 6, 4, 3), takes the top 5 reads by hybrid score
3. Pulls the matching sequences from the original 250K unclassified FASTAs
4. Writes per-cluster FASTAs and one combined FASTA

**Output:**
```
results_tevae/cluster6_for_blast.fasta
results_tevae/cluster4_for_blast.fasta
results_tevae/cluster3_for_blast.fasta
results_tevae/all_priority_clusters_for_blast.fasta   ← submit this to NCBI
```

Then submit the combined FASTA at https://blast.ncbi.nlm.nih.gov/Blast.cgi (blastn → nt → BLAST). Each query produces its own results section; for each one, look at the top hit. See report §3.6 for the three-bucket validation framework.

CLI options:
- `--clusters 6 4 3` — which cluster IDs to extract (default: 6, 4, 3)
- `--top-n 5` — top N reads per cluster (default: 5)
- `--fasta-dir <path>` — directory containing the `*_unclassified_250k.fasta` files

---

## Headline result

On the three-timepoint NY Hospital D pilot:

| Cluster | T1 | T2 | T3 | Growth | Pattern |
|:-:|:-:|:-:|:-:|:-:|:--|
| **6** | 284 | 122 | **3,506** | **×12.3** | Emerging — dominant signal |
| 3 | 0 | 0 | 31 | ×32 | Emerging — low mass |

The hybrid TE-VAE score separates classified from unclassified reads at 0.33% vs 55.6% flagged at μ+3σ. Three trajectory patterns are surfaced naturally by the framework: emerging (clusters 6, 3), transient (clusters 0, 1, 4 — T2-only), declining (clusters 2, 5 — T1-only).

BLAST validation of cluster 6 representative reads is queued (see report §4.5).

---

## Configuration

Most paths default to relative paths (current working directory). To run from outside the repo, set:

```
export HYDRAWATCH_ROOT=/path/to/your/data
```

The scripts check this environment variable and fall back to `.` if unset. Edit the `EMBED_DIR` and `RESULTS_DIR` constants at the top of each script if your folder layout differs.

---

## Reproducibility notes

- **Seed = 42** in TE-VAE training, PCA, HDBSCAN, UMAP, and reservoir sampling (in the BLAST extraction)
- **Float32 precision** throughout — float16 produces unstable Mahalanobis distances at this dimensionality
- **Pooled classified pool** — the TE-VAE is trained on classified embeddings from all three timepoints jointly to define one "site-normal" baseline (so anomaly scores are comparable across time)
- **Top 1% threshold for clustering** — fixed across timepoints; clusters are defined in the joint embedding space

---

## See also

- `../preprocessing/README.md` — how the embeddings were generated
- `../multi_view/` — DNA + protein proof of concept (Figure S2)
- `../report/HydraWatch_report.pdf` — full methodology and results
- `../README.md` — repo-level overview
```

