# Multi-view (DNA + protein) — proof of concept

Proof-of-concept extension showing that DNABERT-2 (nucleotide view) and ESM-2 (protein view) embeddings produce complementary anomaly signals. See report §6.3 and Figure S2.

---

## Files

- `pandemic_plug_and_play.ipynb` — end-to-end notebook running k-mer + DNABERT-2 + ESM-2 anomaly detection on a single sample, comparing the three signals and producing the unified pipeline summary figure.

---

## Sample

This proof of concept ran on a single CASPER sample, **separate from the main three-timepoint pilot**:

| Accession    | Collection         | Source |
|:--|:--|:--|
| SRR37006656  | 5 November 2025    | CASPER PRJNA1247874 |

The main HydraWatch pilot (T1/T2/T3) uses SRR37006657, SRR37006671, SRR37006667.

---

## Headline finding

Of the top 50 anomalous reads, **40 were flagged by both DNABERT-2 and ESM-2** — meaning the DNA and protein views are complementary, not redundant. BLAST triage of representative reads recovered known organisms (*Brevundimonas*, *Azospirillum*, *Gallid* sequences) at high identity, confirming the multi-view scoring surfaces real biological signal.

DNABERT-2 alone produced ~4× stronger separation between classified and unclassified pools than the k-mer baseline (separation delta −0.0515 vs −0.0124).

See Figure S2 in `submission/HydraWatch_report.pdf` for the full visual summary.

---

## What's not yet done

- **Integration into the TE-VAE pipeline.** The proof of concept used Isolation Forest as a simpler anomaly model. Adapting the TE-VAE to fuse DNA + protein representations is immediate next-step work.
- **Multi-timepoint extension.** This single-sample analysis needs to be run across the three pilot timepoints to confirm the complementarity holds longitudinally.

---

## See also

- `../preprocessing/README.md` — how the embeddings (DNA view) are generated
- `../anomaly_detection/README.md` — the main TE-VAE pipeline
- `../submission/HydraWatch_report.pdf` §6.3 + Figure S2 — full description
