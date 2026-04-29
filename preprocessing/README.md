```markdown
Preprocessing Pipeline

End-to-end steps from raw SRA download to Kaggle-ready FASTAs.

This pipeline runs **once per sample** and produces the input files for the DNABERT-2 embedding step (which runs on Kaggle's P100 GPU).

---

## Pipeline overview

```
SRA FASTQ
   │
   ├── (1) Download from NCBI SRA
   ├── (2) Quality trim with Trimmomatic
   ├── (3) Classify with Kraken2 (PlusPF database)
   ├── (4) Strip human reads + subsample
   │       ├── classified pool → 50K non-human reads
   │       └── unclassified pool → 250K reservoir sample
   └── (5) Generate DNABERT-2 embeddings on Kaggle GPU
```

---

## Inputs

Three CASPER samples from NY Hospital D (BioProject **PRJNA1247874**):

| Label | Accession    | Collection      |
|:-:|:--|:--|
| T1    | SRR37006657  | September 2025  |
| T2    | SRR37006671  | October 2025    |
| T3    | SRR37006667  | November 2025   |

For the multi-view proof of concept (separate from main pilot):

| Accession    | Collection         |
|:--|:--|
| SRR37006656  | 5 November 2025    |

---

## Skip preprocessing — use the published Kaggle dataset

The 6 preprocessed FASTAs are already published on Kaggle:

🔗 **https://www.kaggle.com/datasets/divyasitani/dataset-v3**

If you just want to reproduce the embedding step or downstream analysis, attach this dataset to a Kaggle notebook and skip directly to **Step 5** below.

Steps 1–4 below document how the FASTAs in that dataset were generated, for reproducibility.

---

## Step 1 — Download raw FASTQ from SRA

Install the SRA Toolkit if needed:

```
# macOS
brew install sratoolkit

# Linux
sudo apt install sra-toolkit
```

Download and convert to FASTQ:

```
mkdir -p raw_fastq && cd raw_fastq

prefetch SRR37006657 SRR37006671 SRR37006667
fasterq-dump SRR37006657 SRR37006671 SRR37006667 --split-files
```

This produces paired-end files per sample:

```
SRR37006657_1.fastq    # R1 — forward reads
SRR37006657_2.fastq    # R2 — reverse reads
```

Compress to save disk space:

```
gzip *.fastq
```

---

## Step 2 — Quality trim with Trimmomatic

Install:

```
conda install -c bioconda trimmomatic
```

Run paired-end trimming:

```
mkdir -p trimmed

trimmomatic PE -phred33 \
  raw_fastq/SRR37006657_1.fastq.gz raw_fastq/SRR37006657_2.fastq.gz \
  trimmed/SRR37006657_1.trim.fastq.gz trimmed/SRR37006657_1.unpaired.fastq.gz \
  trimmed/SRR37006657_2.trim.fastq.gz trimmed/SRR37006657_2.unpaired.fastq.gz \
  ILLUMINACLIP:TruSeq3-PE.fa:2:30:10 \
  LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:50
```

**Parameters used:**
- `LEADING:3 TRAILING:3` — trim low-quality bases at read ends
- `SLIDINGWINDOW:4:15` — trim where average quality drops below 15 in a 4bp window
- `MINLEN:50` — drop reads shorter than 50bp after trimming

Repeat for SRR37006671 and SRR37006667.

---

## Step 3 — Classify with Kraken2

Download the Kraken2 PlusPF database (~70 GB, only needs to be done once):

```
mkdir -p kraken2_db && cd kraken2_db
wget https://genome-idx.s3.amazonaws.com/kraken/k2_pluspf_20240904.tar.gz
tar -xzf k2_pluspf_20240904.tar.gz
cd ..
```

Run classification:

```
mkdir -p kraken_output

kraken2 \
  --db kraken2_db \
  --paired \
  --output kraken_output/SRR37006657_kraken2.out \
  --report kraken_output/SRR37006657.report \
  --classified-out 'kraken_output/SRR37006657_classified#.fastq' \
  --unclassified-out 'kraken_output/SRR37006657_unclassified#.fastq' \
  trimmed/SRR37006657_1.trim.fastq.gz trimmed/SRR37006657_2.trim.fastq.gz
```

Outputs per sample:

```
SRR37006657_kraken2.out             # per-read classification (used by Step 4a)
SRR37006657.report                  # taxonomic summary
SRR37006657_classified_1.fastq      # R1 reads with a confident taxonomic hit
SRR37006657_classified_2.fastq      # R2 reads (paired)
SRR37006657_unclassified_1.fastq    # R1 reads Kraken2 could not classify
SRR37006657_unclassified_2.fastq    # R2 reads (paired)
```

Across CASPER samples, the unclassified pool is typically 30–40% of all reads. This is what HydraWatch operates on.

Repeat for the other two samples.

---

## Step 4 — Strip human reads and subsample

Two scripts run in sequence. We use only R1 reads (R2 is the reverse complement; embedding both would double-count fragments).

### 4a. Prep classified pool — strip human reads, subsample to 50K

`prep_for_embedding.py` reads the Kraken2 output, drops reads classified as human (taxon 9606) — these dominate the classified pool but are uninformative for site-normal definition — and subsamples the remaining non-human classified reads to 50K. It also writes the entire unclassified pool to a FASTA for downstream subsampling.

```
python preprocessing/prep_for_embedding.py SRR37006657
```

Inputs expected in the current directory:
- `SRR37006657_kraken2.out`
- `SRR37006657_classified_1.fastq`
- `SRR37006657_unclassified_1.fastq`

Outputs:
- `SRR37006657_classified_for_embedding.fasta` (50K non-human classified reads)
- `SRR37006657_unclassified_for_embedding.fasta` (full unclassified pool, FASTA form)

### 4b. Subsample unclassified pool to 250K

The unclassified pool is too large for full DNABERT-2 inference within the Kaggle GPU time budget. Reservoir-sample it down using Vitter's Algorithm R — constant memory, single pass, deterministic via fixed seed:

```
python preprocessing/subsample_unclassified.py \
  SRR37006657_unclassified_for_embedding.fasta \
  SRR37006657_unclassified_250k.fasta \
  250000 --seed 42
```

Repeat both 4a and 4b for SRR37006671 and SRR37006667.

**Final output per sample (these filenames matter — the Kaggle notebook expects exactly these suffixes):**

```
SRR37006657_classified_for_embedding.fasta   # 50K classified reads (R1, no human)
SRR37006657_unclassified_250k.fasta          # 250K unclassified reads (R1)
```

After this step you have **6 FASTA files total** (3 samples × 2 pools). These 6 files are what's published as the Kaggle dataset linked at the top of this README.

---

## Step 5 — Generate DNABERT-2 embeddings on Kaggle

The DNABERT-2 inference step runs on Kaggle's free P100 GPU because local CPU inference would take 10+ hours per sample. Embedding a 250K-read pool on P100 takes ~30–40 minutes.

### 5a. Use the published Kaggle dataset

The 6 preprocessed FASTAs are at:

🔗 **https://www.kaggle.com/datasets/divyasitani/dataset-v3**

(If you regenerated them locally via Steps 1–4, you can upload your own copy as a private dataset instead.)

### 5b. Open the embedding notebook on Kaggle

The notebook is in this repo at `preprocessing/generate-dnabert2-embeddings.ipynb`.

1. https://www.kaggle.com/code → **New Notebook** → import the `.ipynb`
2. **Settings → Accelerator: GPU P100** (T4 x2 also works)
3. **Settings → Internet: ON** (needed to download DNABERT-2 weights)
4. **Add data** → search `dataset-v3` (under user `divyasitani`) → **+ Add**

The notebook is already configured with the matching dataset path:

```python
INPUT_DIR = '/kaggle/input/datasets/divyasitani/dataset-v3'
```

### 5c. Edit the SAMPLES list before running

In Step 3 of the notebook, set:

```python
SAMPLES = [
    'SRR37006657',
    'SRR37006671',
    'SRR37006667',
]
```

### 5d. Run All

The notebook installs pinned dependencies (`transformers==4.29.2`, `tokenizers==0.13.3`, `einops`, `accelerate`), then loads DNABERT-2, then embeds all 6 FASTAs in sequence. Total time: ~2 hours on P100.

### 5e. Commit and download

1. **Save Version → Save & Run All (Commit)**
2. Once committed, go to the **Output** tab
3. Download all 12 files (4 per sample × 3 samples):
   - `SRR37006657_classified_embeddings.npy` (~6 MB)
   - `SRR37006657_classified_ids.txt`
   - `SRR37006657_unclassified_embeddings.npy` (~750 MB)
   - `SRR37006657_unclassified_ids.txt`
   - same 4 files for `SRR37006671` and `SRR37006667`

### 5f. Place the embeddings locally

Put all 12 files in:

```
casper_data/ny_hospital_d/embeddings/25k/embeddings_combined/
```

(or wherever `EMBED_DIR` in the anomaly detection scripts points).

The HydraWatch anomaly detection scripts read combined `all.npy`, `normal.npy`, and `all_metadata.tsv` from this folder. If you only have the per-sample files, merge them by stacking the `.npy` arrays in sample order and building a metadata TSV with columns `read_id`, `sample`, `kind`, `timepoint`.

---

## Summary: what feeds into the next stage

After preprocessing is done, the anomaly detection stage reads:

- **Classified embeddings** — used to train the TE-VAE site-normal model (50K reads × 3 samples = 150K rows)
- **Unclassified embeddings** — scored against the model (250K × 3 = 750K rows)
- **Unclassified FASTAs** (the original `*_unclassified_250k.fasta` files) — kept for sequence extraction during BLAST validation

See `anomaly_detection/` (or the repo root README) for the next steps.

---

## Datasets

| Dataset | URL | Contents |
|:--|:--|:--|
| Raw FASTQ | https://www.ncbi.nlm.nih.gov/sra (PRJNA1247874) | Original sequencing reads |
| Preprocessed FASTAs | https://www.kaggle.com/datasets/divyasitani/dataset-v3 | 6 trimmed + classified + non-human + 50K/250K-subsampled FASTAs |
| Embeddings | output of Step 5; can be re-published as a Kaggle dataset | DNABERT-2 768-dim vectors |

---

## Reproducibility notes

- **Seed = 42** everywhere subsampling is involved (both `prep_for_embedding.py` and `subsample_unclassified.py`)
- **R1 only** — never R2 (would double-count fragments)
- **Human reads removed** — taxon 9606 is dropped from the classified pool to focus the site-normal model on environmental + microbial biology
- **DNABERT-2 frozen** — no fine-tuning
- **Mean pooling** — averaging the final hidden state across all token positions (not [CLS] pooling, not max pooling)
- **Float32 precision** — float16 was tested but produced numerically unstable Mahalanobis distances downstream
- **Pinned dependencies on Kaggle** — `transformers==4.29.2`, `tokenizers==0.13.3` (newer transformers versions break compatibility with DNABERT-2's custom tokenizer)

---

## Troubleshooting

**`fasterq-dump` is slow.** Use `prefetch` first (downloads `.sra` files), then `fasterq-dump` is much faster than direct streaming.

**Trimmomatic adapter file not found.** The `TruSeq3-PE.fa` adapter file ships with Trimmomatic. Find it with:
```
find / -name "TruSeq3-PE.fa" 2>/dev/null
```
Pass the full path to the `ILLUMINACLIP:` option.

**Kraken2 database disk space.** PlusPF is ~70 GB. The smaller Standard-8 (~8 GB) database also works but classifies fewer reads, leaving more in the unclassified pool — which actually increases the dataset HydraWatch operates on.

**Kaggle dataset upload size limit.** Free Kaggle accounts allow up to 20 GB per dataset. Six FASTAs (3 × 50K classified + 3 × 250K unclassified) are well under this.

**Kaggle notebook fails on `pip install tokenizers`.** Restart the kernel after the install cell (Run → Restart & Run All from Step 3) and the pinned versions will load correctly.

**Sanity-check fails ("collapsed embeddings").** If `unique-norms` is fewer than 1000 in the Step 7 sanity check of the embedding notebook, the model probably failed to load properly. Restart the kernel and rerun.
```

