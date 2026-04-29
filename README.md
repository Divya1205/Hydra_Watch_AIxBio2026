# HydraWatch

**Embedding-based wastewater pathogen surveillance for federated hospital networks.**

AIxBio Hackathon (Berlin node) · Track 2 · Apart Research · April 2026

> Local embeddings · Global signal · Reference-free early warning

---

## What this is

HydraWatch is a reference-free, privacy-preserving wastewater pathogen surveillance pipeline designed for federated hospital networks. Each hospital sequences its own sewershed, embeds reads with DNABERT-2, and trains a local Transformer-encoder Variational Autoencoder (TE-VAE) on the classified read pool to define a site-normal baseline. A hybrid anomaly score flags reads in the unclassified pool — the blind spot where novel pathogens hide because reference-based tools cannot see them. Anomalies are clustered with HDBSCAN and tracked across timepoints to surface emerging signals.

Cross-site detection happens by query, not data: hospitals exchange ~3 KB cluster centroids, never raw reads or read-level embeddings.

📄 **Full methodology and results:** [`report/HydraWatch_report.pdf`](report/HydraWatch_report.pdf)
🎯 **Slide deck:** [`report/HydraWatch_deck.pdf`](report/HydraWatch_deck.pdf)

---

## Headline result

On a three-timepoint NY hospital sewershed pilot (CASPER PRJNA1247874, Sep–Nov 2025), joint HDBSCAN clustering surfaces a dominant emerging cluster:

| Cluster | T1 | T2 | T3 | Growth | Pattern |
|:-:|:-:|:-:|:-:|:-:|:--|
| **6** | 284 | 122 | **3,506** | **×12.3** | Emerging — dominant signal |
| 3 | 0 | 0 | 31 | ×32 | Emerging — low mass |

The hybrid TE-VAE score cleanly separates classified from unclassified reads (0.33% vs 55.6% flagged at μ + 3σ).

A multi-view (DNA + protein) proof of concept on a separate CASPER sample shows that 40 of the top 50 anomalous reads are flagged by both DNABERT-2 and ESM-2 — the views are complementary, not redundant.

BLAST validation of the emerging cluster is queued and will be reported in a follow-up.

---

## Pipeline overview