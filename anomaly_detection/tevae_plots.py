#!/usr/bin/env python3
"""
tevae_plots.py

Generate the 3 deck/report figures from TE-VAE scores:
  1. fig_tevae_score_distributions.png  - histogram per timepoint
  2. fig_tevae_cluster_trajectories.png - emerging clusters over time
  3. fig_tevae_joint_umap.png           - 3-panel joint UMAP

Inputs:
  results_tevae/tevae_scores.tsv       - per-read hybrid score (already produced)
  embeddings_combined/all.npy          - embeddings (for clustering + UMAP)

Edit BASE_DIR below to match your machine.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import os

# ── Path config — set HYDRAWATCH_ROOT env var, or override defaults below
PROJECT_ROOT = Path(os.environ.get("HYDRAWATCH_ROOT", "."))

EMBED_DIR = PROJECT_ROOT / "casper_data/ny_hospital_d/embeddings/25k/embeddings_combined"
RESULTS_DIR = PROJECT_ROOT / "results_tevae"

# ── Config
ANOMALY_PERCENTILE = 99
HDBSCAN_MIN_CLUSTER_SIZE = 30
N_PCA_FOR_CLUSTER = 50
SEED = 42
DPI = 300  # high-res for small-size readability in the report
np.random.seed(SEED)

# ── Global matplotlib settings for crisp small-size rendering
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def main():
    print("Loading TE-VAE scores + embeddings...")
    scores = pd.read_csv(RESULTS_DIR / "tevae_scores.tsv", sep="\t")
    all_emb = np.load(EMBED_DIR / "all.npy").astype(np.float32)
    meta = pd.read_csv(EMBED_DIR / "all_metadata.tsv", sep="\t")
    print(f"  scores: {len(scores):,}")
    print(f"  embeddings: {all_emb.shape}")

    # Re-attach scores to row-aligned meta
    meta = meta.merge(
        scores[["read_id", "sample", "kind", "tevae_error"]],
        on=["read_id", "sample", "kind"], how="left"
    )
    assert meta["tevae_error"].notna().all()

    ucl_mask = (meta["kind"] == "unclassified").values
    ucl_meta = meta[ucl_mask].copy().reset_index(drop=True)
    ucl_emb = all_emb[ucl_mask]
    print(f"  unclassified: {len(ucl_meta):,}")

    palette = ["#0D7377", "#C0392B", "#8B5CF6", "#E67E22", "#2980B9"]
    label_order = sorted(ucl_meta["timepoint"].unique())

    # ─────────────────────────────────────────────────────────────────────
    # FIG 1 — score distributions per timepoint
    # ─────────────────────────────────────────────────────────────────────
    print("\nFig 1: distributions...")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, tp in enumerate(label_order):
        data = ucl_meta[ucl_meta["timepoint"] == tp]["tevae_error"]
        acc = ucl_meta[ucl_meta["timepoint"] == tp]["sample"].iloc[0]
        ax.hist(data, bins=80, alpha=0.55, color=palette[i % len(palette)],
                label=f"{tp} ({acc})", density=True,
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("TE-VAE hybrid anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly score distribution by timepoint — TE-VAE")
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_tevae_score_distributions.png", dpi=DPI)
    plt.close()
    print("  fig_tevae_score_distributions.png")

    # ─────────────────────────────────────────────────────────────────────
    # Top 1% anomalies → cluster
    # ─────────────────────────────────────────────────────────────────────
    threshold = np.percentile(ucl_meta["tevae_error"].values, ANOMALY_PERCENTILE)
    high_mask = ucl_meta["tevae_error"].values >= threshold
    high_meta = ucl_meta[high_mask].copy().reset_index(drop=True)
    high_emb = ucl_emb[high_mask]
    print(f"\nTop 1% anomalies: {len(high_meta):,} reads (threshold {threshold:.4f})")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=N_PCA_FOR_CLUSTER, random_state=SEED)
    high_pcs = pca.fit_transform(high_emb)

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        cluster_labels = clusterer.fit_predict(high_pcs)
    except ImportError:
        from sklearn.cluster import KMeans
        n_kmeans = max(5, min(20, len(high_meta) // 100))
        cluster_labels = KMeans(n_clusters=n_kmeans, random_state=SEED, n_init=10) \
            .fit_predict(high_pcs)

    high_meta["cluster"] = cluster_labels
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  Clusters: {n_clusters}")

    # Trajectory table
    traj = high_meta.groupby(["cluster", "timepoint"]).size().unstack(fill_value=0)
    traj = traj.reindex(columns=label_order, fill_value=0)
    traj["total"] = traj.sum(axis=1)
    traj["mean_error"] = high_meta.groupby("cluster")["tevae_error"].mean()
    if len(label_order) >= 2:
        first, last = label_order[0], label_order[-1]
        traj["growth_ratio"] = (traj[last] + 1) / (traj[first] + 1)

    traj_clean = traj[traj.index != -1].sort_values(
        "growth_ratio" if "growth_ratio" in traj.columns else "total",
        ascending=False
    )
    traj_clean.to_csv(RESULTS_DIR / "tevae_cluster_trajectories.tsv", sep="\t")
    high_meta.sort_values(["cluster", "tevae_error"], ascending=[True, False]) \
             .to_csv(RESULTS_DIR / "tevae_top_anomalies_with_clusters.tsv",
                     sep="\t", index=False)
    print(f"  Saved cluster tables")

    # ─────────────────────────────────────────────────────────────────────
    # FIG 2 — cluster trajectories
    # ─────────────────────────────────────────────────────────────────────
    print("\nFig 2: trajectories...")
    if "growth_ratio" in traj.columns:
        emerging = traj_clean.head(8)
        fig, ax = plt.subplots(figsize=(8, 5))
        for cluster_id, row in emerging.iterrows():
            counts = [row[lbl] for lbl in label_order]
            ax.plot(label_order, counts, marker="o", linewidth=2, markersize=7,
                    label=f"Cluster {cluster_id} (×{row['growth_ratio']:.1f})")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Reads in cluster")
        ax.set_title(f"Top emerging anomaly clusters — TE-VAE ({first} → {last})")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9,
                  frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig_tevae_cluster_trajectories.png", dpi=DPI)
        plt.close()
        print("  fig_tevae_cluster_trajectories.png")
        print("\n  Top emerging clusters:")
        print(emerging[label_order + ["mean_error", "growth_ratio"]].to_string())

    # ─────────────────────────────────────────────────────────────────────
    # FIG 3 — joint UMAP, 3 panels
    # ─────────────────────────────────────────────────────────────────────
    print("\nFig 3: joint UMAP (slow, ~3-5 min)...")
    try:
        import umap
        rng = np.random.RandomState(SEED)
        cl_mask_all = (meta["kind"] == "classified").values
        cl_emb = all_emb[cl_mask_all]
        n_cl_backdrop = min(3000, len(cl_emb))
        cl_idx = rng.choice(len(cl_emb), n_cl_backdrop, replace=False)

        pcs_list, labels_list, scores_list, kind_list = [], [], [], []
        pcs_list.append(cl_emb[cl_idx])
        labels_list.append(np.array(["classified"] * n_cl_backdrop))
        scores_list.append(np.zeros(n_cl_backdrop))
        kind_list.append(np.array(["classified"] * n_cl_backdrop))

        per_sample_n = 8000
        for tp in label_order:
            tp_mask = (ucl_meta["timepoint"] == tp).values
            tp_emb = ucl_emb[tp_mask]
            tp_scores = ucl_meta[tp_mask]["tevae_error"].values
            n = min(per_sample_n, len(tp_emb))
            idx = rng.choice(len(tp_emb), n, replace=False)
            pcs_list.append(tp_emb[idx])
            labels_list.append(np.array([tp] * n))
            scores_list.append(tp_scores[idx])
            kind_list.append(np.array(["unclassified"] * n))

        combined = np.vstack(pcs_list)
        combined_labels = np.concatenate(labels_list)
        combined_scores = np.concatenate(scores_list)
        combined_kind = np.concatenate(kind_list)

        pca_umap = PCA(n_components=N_PCA_FOR_CLUSTER, random_state=SEED)
        combined_pcs = pca_umap.fit_transform(combined)

        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED, n_jobs=-1)
        coords = reducer.fit_transform(combined_pcs)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

        ax = axes[0]
        m = combined_kind == "classified"
        ax.scatter(coords[m, 0], coords[m, 1], c="#E2E8F0", s=3, alpha=0.4,
                   label="classified backdrop", linewidths=0)
        for i, tp in enumerate(label_order):
            mask = combined_labels == tp
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=palette[i % len(palette)], s=4, alpha=0.6,
                       label=tp, linewidths=0)
        ax.set_title("Unclassified reads by timepoint")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(frameon=True, framealpha=0.9, markerscale=2)

        ax = axes[1]
        m = combined_kind == "classified"
        ax.scatter(coords[m, 0], coords[m, 1], c="#E2E8F0", s=3, alpha=0.4,
                   linewidths=0)
        m = combined_kind == "unclassified"
        sc = ax.scatter(coords[m, 0], coords[m, 1], c=combined_scores[m],
                        s=4, alpha=0.75, cmap="plasma", linewidths=0,
                        vmin=np.percentile(combined_scores[m], 5),
                        vmax=np.percentile(combined_scores[m], 99))
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("TE-VAE hybrid score", fontsize=10)
        ax.set_title("Unclassified reads by anomaly score")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

        ax = axes[2]
        cm = combined_kind == "classified"
        um = combined_kind == "unclassified"
        ax.scatter(coords[cm, 0], coords[cm, 1], c="#94A3B8", s=3, alpha=0.4,
                   label="classified", linewidths=0)
        ax.scatter(coords[um, 0], coords[um, 1], c="#0D7377", s=4, alpha=0.55,
                   label="unclassified", linewidths=0)
        ax.set_title("Classified vs unclassified")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(frameon=True, framealpha=0.9, markerscale=2)

        plt.suptitle(f"Joint UMAP — TE-VAE method — {len(combined):,} reads",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig_tevae_joint_umap.png", dpi=DPI)
        plt.close()
        print("  fig_tevae_joint_umap.png")
    except ImportError:
        print("  Skipping UMAP (pip install umap-learn)")

    print(f"\nDone. Figures in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()