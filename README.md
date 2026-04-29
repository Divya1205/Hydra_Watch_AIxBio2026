# HydraWatch

**Embedding-based wastewater pathogen surveillance for federated hospital networks.**

AIxBio Hackathon · Track 2 · Apart Research · April 2026

> Local embeddings · Global signal · Reference-free early warning

---

## What this is

HydraWatch is a reference-free, privacy-preserving wastewater pathogen surveillance pipeline designed for federated hospital networks. Each hospital sequences its own sewershed, embeds reads with DNABERT-2, and trains a local Transformer-encoder Variational Autoencoder (TE-VAE) on the classified read pool to define a site-normal baseline. A hybrid anomaly score flags reads in the unclassified pool — the blind spot where novel pathogens hide because reference-based tools cannot see them. Anomalies are clustered with HDBSCAN and tracked across timepoints to surface emerging signals.

Cross-site detection happens by query, not data: hospitals exchange ~3 KB cluster centroids, never raw reads or read-level embeddings.

 **Full methodology and results:** [`submission/HydraWatch_report.pdf`](submission/HydraWatch_report.pdf)
 **Slide deck:** [`submission/HydraWatch_Track2_slides.pdf`](submission/HydraWatch_Track2_slides.pdf)

---

## Headline result

On a three-timepoint NY hospital sewershed pilot (CASPER PRJNA1247874, September–November 2025), joint HDBSCAN clustering surfaces a dominant emerging cluster:

| Cluster | T1 | T2 | T3 | Growth | Pattern |
|:-:|:-:|:-:|:-:|:-:|:--|
| **6** | 284 | 122 | **3,506** | **×12.3** | Emerging — dominant signal |
| 3 | 0 | 0 | 31 | ×32 | Emerging — low mass |

The hybrid TE-VAE score cleanly separates classified from unclassified reads (0.33% vs 55.6% flagged at μ + 3σ).

A multi-view (DNA + protein) proof of concept on a separate CASPER sample (SRR37006656) shows that 40 of the top 50 anomalous reads are flagged by both DNABERT-2 and ESM-2 — the views are complementary, not redundant.

BLAST validation of the emerging cluster is queued and will be reported in a follow-up.

---

## Pipeline overview

```
Wastewater FASTQ
   │
   ├── Trimmomatic (QC + trimming)
   ├── Kraken2 (reference classification)
   │     └── split into classified / unclassified pools
   │
   ├── Strip human reads + subsample (50K classified, 250K unclassified, R1 only)
   │
   ├── DNABERT-2 embedding (768-dim, frozen, mean-pooled, on Kaggle P100 GPU)
   │
   ├── TE-VAE training on classified pool
   │     └── hybrid anomaly score: robust z(recon) + robust z(log(latent Mahalanobis))
   │
   ├── HDBSCAN clustering on top 1% anomalies (pooled across timepoints)
   │     └── trajectory analysis: emerging / transient / declining
   │
   └── BLAST validation of representative reads (queued)
```

See `submission/HydraWatch_report.pdf` §3 for full methodology and Figure S1 for the architectural diagram.

---

## Repository structure

```
Hydra_Watch_AIxBio2026/
├── README.md                              ← you are here
├── requirements.txt                       ← Python dependencies
├── LICENSE                                ← MIT
│
├── preprocessing/                         ← Stages 1–5 of the pipeline
│   ├── README.md                          ← preprocessing walkthrough
│   ├── prep_for_embedding.py              ← strip human reads, subsample classified to 50K
│   ├── subsample_unclassified.py          ← reservoir-sample unclassified to 250K
│   └── generate-dnabert2-embeddings.ipynb ← DNABERT-2 inference on Kaggle GPU
│
├── anomaly_detection/                     ← Stages 6–8: TE-VAE + clustering + BLAST prep
│   ├── README.md                          ← anomaly detection walkthrough
│   ├── tevae_anomaly_detection_hybrid.py  ← TE-VAE training + hybrid scoring
│   ├── tevae_plots.py                     ← deck/report figures (3 + UMAP)
│   ├── replot_tevae_components.py         ← regenerate distribution figure
│   └── extract_clusters_for_blast.py      ← BLAST FASTA extraction
│
├── multi_view/                            ← §6.3: ESM-2 proof of concept
│   └── README.md                          ← multi-view extension notes
│
├── results/                               ← outputs (samples; full data on request)
│   ├── tevae_cluster_trajectories.tsv
│   ├── tevae_threshold.txt
│   └── figures/
│
├── data/                                  ← documentation only; raw data not committed
│   └── README.md                          ← how to download from SRA + Kaggle
│
└── submission/                            ← final hackathon deliverables
    ├── HydraWatch_report.pdf
    └── HydraWatch_Track2_slides.pdf
```

Each subfolder has its own README explaining what the scripts do and how to run them in order.

---

## Quick start

### 1. Set up the environment

```
git clone https://github.com/Divya1205/Hydra_Watch_AIxBio2026.git
cd Hydra_Watch_AIxBio2026
pip install -r requirements.txt
```

### 2. Get the data — three options

**Option A — full reproducibility, raw SRA reads:**
```
prefetch SRR37006657 SRR37006671 SRR37006667
fasterq-dump SRR37006657 SRR37006671 SRR37006667 --split-files
```
Then follow `preprocessing/README.md` from Step 1.

**Option B — skip preprocessing, use the published Kaggle dataset:**

🔗 https://www.kaggle.com/datasets/divyasitani/dataset-v3

Six preprocessed FASTAs ready for DNABERT-2 embedding. Attach to a Kaggle notebook and skip to `preprocessing/README.md` Step 5.

**Option C — skip preprocessing AND embedding:**

If a public embeddings dataset is available, download the 12 `.npy` files and place them in `casper_data/ny_hospital_d/embeddings/25k/embeddings_combined/`, then run the anomaly detection scripts directly.

### 3. Run the pipeline

```
# Preprocessing (see preprocessing/README.md for full details)
python preprocessing/prep_for_embedding.py SRR37006657   # repeat for SRR37006671, SRR37006667
python preprocessing/subsample_unclassified.py \
    SRR37006657_unclassified_for_embedding.fasta \
    SRR37006657_unclassified_250k.fasta \
    250000 --seed 42
# Then run preprocessing/generate-dnabert2-embeddings.ipynb on Kaggle GPU

# Anomaly detection (see anomaly_detection/README.md for full details)
python anomaly_detection/tevae_anomaly_detection_hybrid.py
python anomaly_detection/tevae_plots.py
python anomaly_detection/extract_clusters_for_blast.py \
    --fasta-dir <path/to/unclassified/fastas>
```

Outputs land in `results_tevae/` (anomaly scores, cluster trajectories, figures, BLAST-ready FASTAs).

---

## Method summary

| Stage | Tool | Notes |
|:--|:--|:--|
| QC + trimming | Trimmomatic | Standard parameters (LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:50) |
| Reference classification | Kraken2 | PlusPF database |
| Strip human + subsample classified | `prep_for_embedding.py` | Drops taxon 9606, subsamples to 50K, seed=42 |
| Subsample unclassified | `subsample_unclassified.py` | Vitter's Algorithm R, 250K reads, R1 only, seed=42 |
| Embedding | DNABERT-2 (117M, frozen) | 768-dim, mean-pooled, max 512 tokens |
| Anomaly model | TE-VAE | 32-dim latent, β = 0.1, 50 epochs |
| Anomaly score | Hybrid | Robust z(recon) + robust z(log(latent Mahalanobis)) |
| Threshold | μ + 3σ on classified scores | ~0.3% expected flag rate under Gaussianity |
| Clustering | HDBSCAN | min_cluster_size = 30, on 50 PCA components |
| Trajectory analysis | Per-cluster T1/T2/T3 counts | Emerging / transient / declining |
| Validation | NCBI web blastn | Queued for cluster 6 representative reads |

---

## Citation

If you use this work, please cite the project report:

> Sitani D, ElSayed M, Arrey F, Schutz H, Held S. *HydraWatch: Embedding-based wastewater pathogen surveillance for federated hospital networks.* AIxBio Hackathon Track 2, Apart Research, April 2026.

Underlying dataset:

> Justen LJ et al. (2026). *Deep untargeted wastewater metagenomic sequencing from sewersheds across the United States.* medRxiv 2026-03 (CASPER consortium). BioProject PRJNA1247874.

---

## Authors

| Author | Affiliation |
|:--|:--|
| Divya Sitani (lead) | Independent Researcher |
| Mohammed ElSayed | Helmut Schmidt Universität Hamburg |
| Frida Arrey | Independent Researcher |
| Hanna Schutz | Oxford Nanopore Technologies |
| Sascha Held | Swissbit AG |

With Apart Research.

---

## Limitations

HydraWatch is a hackathon-scale pilot. Key caveats:

- **BLAST validation is queued**, not yet completed for the TE-VAE clusters. The trajectory pattern (×12.3 emergence) is the embedding-space signal; sequence-level anchoring follows.
- **Single-site pilot.** The federated multi-site architecture is described and motivated, but only a single-site three-timepoint pilot has been run.
- **TE-VAE trained on classified embeddings.** The model's notion of "normal" inherits any biases of Kraken2's reference database.
- **Multi-view (ESM-2) is proof of concept only**, on a single sample separate from the main pilot. Full integration into the TE-VAE pipeline is future work.

See report §6.3 for full limitations and future work.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Built on top of the SecureBio CASPER initiative dataset (PRJNA1247874). HydraWatch is complementary to, not a replacement for, reference-based wastewater surveillance. The two layers cover different failure modes.
```



That's the last README. Repo is fully documented end-to-end. Submit and rest.
