#!/usr/bin/env python3
"""
replot_tevae_components.py

Regenerate the three-component distribution figure (recon error,
latent Mahalanobis, hybrid score) from the existing tevae_scores.tsv,
without retraining.

Outputs (high-res, report-ready):
  results_tevae/fig_tevae_distribution_components.png   - 3-panel
  results_tevae/fig_tevae_distribution.png              - hybrid only
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config
RESULTS_DIR = Path("/Users/sitani/Documents/pandemic_preparedness/results_tevae")
SCORES_TSV = RESULTS_DIR / "tevae_scores.tsv"
THRESHOLD_K = 3
DPI = 300

# ── Crisp small-size rendering
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def main():
    print(f"Loading {SCORES_TSV}...")
    meta = pd.read_csv(SCORES_TSV, sep="\t")
    print(f"  rows: {len(meta):,}")

    required = {"recon_error", "latent_mahal", "tevae_error", "kind"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"tevae_scores.tsv is missing columns: {missing}. "
                         f"Re-run the training script to regenerate it.")

    classified_mask = (meta["kind"] == "classified").values
    cl_scores = meta.loc[classified_mask, "tevae_error"].values
    threshold = float(cl_scores.mean() + THRESHOLD_K * cl_scores.std())
    print(f"  threshold (μ+{THRESHOLD_K}σ on classified): {threshold:.3f}")

    n_cl = int(classified_mask.sum())
    n_ucl = len(meta) - n_cl

    # ── Three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    components = [
        ("recon_error", "Reconstruction error", False),
        ("latent_mahal", "Latent Mahalanobis", False),
        ("tevae_error", "Hybrid score (z-recon + z-latent)", True),
    ]

    for ax, (col, title, is_hybrid) in zip(axes, components):
        cl = meta.loc[classified_mask, col].values
        ucl = meta.loc[~classified_mask, col].values

        ax.hist(cl, bins=80, alpha=0.6, label=f"Classified (n={n_cl:,})",
                color="#0D7377", density=True, edgecolor="white", linewidth=0.3)
        ax.hist(ucl, bins=80, alpha=0.6, label=f"Unclassified (n={n_ucl:,})",
                color="#C0392B", density=True, edgecolor="white", linewidth=0.3)

        if is_hybrid:
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                       label=f"Threshold = {threshold:.2f}")
            # Clip x-axis to remove extreme-outlier whitespace
            p995 = np.percentile(meta[col], 99.5)
            x_min = min(cl.min(), ucl.min())
            ax.set_xlim(x_min, p995)

        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    plt.suptitle("TE-VAE — three score components", fontsize=13, y=1.02)
    plt.tight_layout()
    out_components = RESULTS_DIR / "fig_tevae_distribution_components.png"
    plt.savefig(out_components, dpi=DPI)
    plt.close()
    print(f"  → {out_components}")

    # ── Single hybrid panel (for the deck)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cl = meta.loc[classified_mask, "tevae_error"].values
    ucl = meta.loc[~classified_mask, "tevae_error"].values
    ax.hist(cl, bins=80, alpha=0.6, label=f"Classified (n={n_cl:,})",
            color="#0D7377", density=True, edgecolor="white", linewidth=0.3)
    ax.hist(ucl, bins=80, alpha=0.6, label=f"Unclassified (n={n_ucl:,})",
            color="#C0392B", density=True, edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold (μ+{THRESHOLD_K}σ = {threshold:.2f})")
    p995 = np.percentile(meta["tevae_error"], 99.5)
    ax.set_xlim(min(cl.min(), ucl.min()), p995)
    ax.set_xlabel("TE-VAE hybrid anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("TE-VAE hybrid score: classified vs unclassified")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out_single = RESULTS_DIR / "fig_tevae_distribution.png"
    plt.savefig(out_single, dpi=DPI)
    plt.close()
    print(f"  → {out_single}")

    print("\nDone.")


if __name__ == "__main__":
    main()
