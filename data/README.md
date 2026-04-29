# Data

Raw sequencing data is **not included** in this repository — it is too large for git and freely available from NCBI SRA.

This folder contains only this README, which documents how to retrieve the data and where it sits in the pipeline.

---

## Source

All sequencing reads come from the SecureBio CASPER initiative, BioProject **PRJNA1247874**:

> Justen LJ et al. (2026). *Deep untargeted wastewater metagenomic sequencing from sewersheds across the United States.* medRxiv 2026-03 (CASPER consortium).

---

## Pilot accessions (NY Hospital D, three timepoints)

The main HydraWatch pilot uses three runs from a single NY Hospital D sewershed:

| Label | Accession    | Collection      |
|:-:|:--|:--|
| T1    | SRR37006657  | September 2025  |
| T2    | SRR37006671  | October 2025    |
| T3    | SRR37006667  | November 2025   |

## Multi-view proof of concept (separate sample)

The DNA + protein multi-view extension (see report §6.3 and Figure S2) was piloted on a separate CASPER sample, not part of the main three-timepoint pilot:

| Accession    | Collection         |
|:--|:--|
| SRR37006656  | 5 November 2025    |

---

## How to download

### Option 1 — SRA Toolkit (raw FASTQ)

Install the toolkit:

```
# macOS
brew install sratoolkit

# Linux
sudo apt install sra-toolkit
```

Download the four runs:

```
prefetch SRR37006657 SRR37006671 SRR37006667 SRR37006656
fasterq-dump SRR37006657 SRR37006671 SRR37006667 SRR37006656 --split-files
```

This produces paired-end FASTQs (`<accession>_1.fastq` and `<accession>_2.fastq` per run). Compress with `gzip *.fastq` to save disk space.

For preprocessing the raw FASTQs (Trimmomatic → Kraken2 → subsample → DNABERT-2), see [`preprocessing/README.md`](../preprocessing/README.md).

### Option 2 — Skip preprocessing, use the published Kaggle dataset

The 6 preprocessed FASTAs (3 samples × classified + unclassified pools, after Trimmomatic + Kraken2 + human-read removal + subsampling) are already published:

🔗 **https://www.kaggle.com/datasets/divyasitani/dataset-v3**

Attach this dataset to a Kaggle notebook to skip directly to the DNABERT-2 embedding step (preprocessing Step 5).

### Option 3 — Batch metadata via SRA Run Selector

For collection dates, geographic data, sample types, and other BioSample attributes across the whole CASPER project:

🔗 **https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA1247874**

Click **Metadata** at the top to download `SraRunTable.csv`.

---

## Data sizes (approximate)

| Stage | Size per sample | Notes |
|:--|:--|:--|
| Raw FASTQ (paired-end, gzipped) | ~5–10 GB | After `gzip` |
| Trimmed FASTQ | ~5–10 GB | After Trimmomatic |
| Kraken2 classified pool (R1 only) | ~500 MB | Mostly human |
| Kraken2 unclassified pool (R1 only) | ~700 MB | The HydraWatch input |
| Subsampled FASTAs (50K classified + 250K unclassified) | ~50 MB | Published on Kaggle |
| DNABERT-2 embeddings per pool | 6–750 MB | float32 .npy |

---

## Where the data flows in the pipeline

```
NCBI SRA (PRJNA1247874)
   │
   ▼
preprocessing/  ────────►  Kaggle dataset: divyasitani/dataset-v3
   │
   ▼
DNABERT-2 (Kaggle GPU)
   │
   ▼
casper_data/ny_hospital_d/embeddings/25k/embeddings_combined/
   │
   ▼
anomaly_detection/  ────────►  results_tevae/
```

The `casper_data/` directory referenced by the anomaly detection scripts is **not** in this repo — it's where you place the embedding outputs locally after running the Kaggle notebook. See `preprocessing/README.md` Step 5f.

---

## License and citation

Underlying sequencing data is released by the CASPER consortium under their data-sharing terms (see PRJNA1247874 on NCBI). Cite:

> Justen LJ et al. (2026). *Deep untargeted wastewater metagenomic sequencing from sewersheds across the United States.* medRxiv 2026-03 (CASPER consortium).

The HydraWatch preprocessing pipeline and embeddings derived from this data are released under MIT (see [LICENSE](../LICENSE)).
```


