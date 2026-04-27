import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import ops as kops
from sklearn.preprocessing import StandardScaler

# ── Path config — set HYDRAWATCH_ROOT env var, or override defaults below
PROJECT_ROOT = Path(os.environ.get("HYDRAWATCH_ROOT", "."))

EMBED_DIR = PROJECT_ROOT / "casper_data/ny_hospital_d/embeddings/25k/embeddings_combined"
RESULTS_DIR = PROJECT_ROOT / "results_tevae"
AE_SCORES = PROJECT_ROOT / "results_ae/ae_scores.tsv"
MAHAL_SCORES = PROJECT_ROOT / "results_v3/all_unclassified_scores.tsv"

EPOCHS = 50
BATCH_SIZE = 256
LATENT_DIM = 32
HIDDEN_DIM = 128
KL_WEIGHT = 0.1         # bumped up for tighter normal distribution
SEED = 42
THRESHOLD_K = 3

# ── Hybrid score weights (tweak to taste)
W_RECON = 1.0    # reconstruction error contribution
W_LATENT = 1.0   # latent Mahalanobis distance contribution

tf.random.set_seed(SEED)
np.random.seed(SEED)
RESULTS_DIR.mkdir(exist_ok=True)


class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + kops.exp(0.5 * logvar) * eps


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm1 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 2, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        a = self.attn(x, x)
        x = self.norm1(x + a)
        f = self.ffn(x)
        return self.norm2(x + f)


def build_tevae(input_dim):
    class TE-VAE(Model):
        def __init__(self, input_dim, **kwargs):
            super().__init__(**kwargs)
            self.input_proj = layers.Dense(HIDDEN_DIM, activation="relu")
            self.tblock1 = TransformerBlock(HIDDEN_DIM, num_heads=4)
            self.tblock2 = TransformerBlock(HIDDEN_DIM, num_heads=4)
            self.mu_head = layers.Dense(LATENT_DIM)
            self.logvar_head = layers.Dense(LATENT_DIM)
            self.sampler = Sampling()
            self.decoder = tf.keras.Sequential([
                layers.Dense(HIDDEN_DIM, activation="relu"),
                layers.Dense(input_dim),
            ])
            self.recon_tracker = tf.keras.metrics.Mean(name="recon")
            self.kl_tracker = tf.keras.metrics.Mean(name="kl")
            self.total_tracker = tf.keras.metrics.Mean(name="loss")

        @property
        def metrics(self):
            return [self.total_tracker, self.recon_tracker, self.kl_tracker]

        def encode(self, x):
            h = self.input_proj(x)
            h = kops.expand_dims(h, axis=1)
            h = self.tblock1(h)
            h = self.tblock2(h)
            h = kops.squeeze(h, axis=1)
            return self.mu_head(h), self.logvar_head(h)

        def call(self, x, training=False):
            mu, logvar = self.encode(x)
            z = self.sampler([mu, logvar])
            return self.decoder(z)

        def compute_loss_components(self, x):
            mu, logvar = self.encode(x)
            z = self.sampler([mu, logvar])
            recon = self.decoder(z)
            recon_loss = kops.mean(kops.square(x - recon))
            kl_loss = -0.5 * kops.mean(
                1 + logvar - kops.square(mu) - kops.exp(logvar)
            )
            return recon_loss, kl_loss, recon

        def train_step(self, data):
            x = data[0] if isinstance(data, tuple) else data
            with tf.GradientTape() as tape:
                recon_loss, kl_loss, _ = self.compute_loss_components(x)
                total = recon_loss + KL_WEIGHT * kl_loss
            grads = tape.gradient(total, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.total_tracker.update_state(total)
            self.recon_tracker.update_state(recon_loss)
            self.kl_tracker.update_state(kl_loss)
            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            x = data[0] if isinstance(data, tuple) else data
            recon_loss, kl_loss, _ = self.compute_loss_components(x)
            total = recon_loss + KL_WEIGHT * kl_loss
            self.total_tracker.update_state(total)
            self.recon_tracker.update_state(recon_loss)
            self.kl_tracker.update_state(kl_loss)
            return {m.name: m.result() for m in self.metrics}

        def encode_mu(self, x, batch_size=512):
            """Get latent means in batches (no sampling)."""
            mus = []
            for i in range(0, len(x), batch_size):
                xb = x[i:i + batch_size]
                mu, _ = self.encode(xb)
                mus.append(mu.numpy())
            return np.concatenate(mus, axis=0)

    model = TE-VAE(input_dim)
    model.build((None, input_dim))
    model.compile(optimizer="adam")
    return model


def main():
    print("Loading embeddings...")
    normal = np.load(EMBED_DIR / "normal.npy").astype(np.float32)
    all_emb = np.load(EMBED_DIR / "all.npy").astype(np.float32)
    meta = pd.read_csv(EMBED_DIR / "all_metadata.tsv", sep="\t")
    print(f"  normal: {normal.shape}")
    print(f"  all:    {all_emb.shape}")

    scaler = StandardScaler()
    normal_z = scaler.fit_transform(normal).astype(np.float32)
    all_z = scaler.transform(all_emb).astype(np.float32)

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(normal_z))
    split = int(0.8 * len(idx))
    train_x = normal_z[idx[:split]]
    val_x = normal_z[idx[split:]]
    print(f"  train: {len(train_x):,}  val: {len(val_x):,}")

    print("\nBuilding TE-VAE...")
    model = build_tevae(input_dim=normal_z.shape[1])
    model.summary()

    print("\nTraining (classified only)...")
    model.fit(
        train_x, train_x,
        validation_data=(val_x, val_x),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # ── Reconstruction error
    print("\nComputing reconstruction error...")
    recon = model.predict(all_z, batch_size=BATCH_SIZE * 2, verbose=1)
    recon_error = np.mean((all_z - recon) ** 2, axis=1).astype(np.float32)

    # ── Latent space Mahalanobis (compute on classified mu, score everything)
    print("\nComputing latent-space Mahalanobis distance...")
    print("  Encoding classified pool...")
    mu_classified = model.encode_mu(normal_z, batch_size=BATCH_SIZE * 2)
    print("  Encoding all reads...")
    mu_all = model.encode_mu(all_z, batch_size=BATCH_SIZE * 2)

    # Fit Mahalanobis on classified mu
    cl_mean = mu_classified.mean(axis=0)
    cl_cov = np.cov(mu_classified, rowvar=False)
    # Add small ridge for numerical stability
    cl_cov += np.eye(cl_cov.shape[0]) * 1e-4
    cl_cov_inv = np.linalg.inv(cl_cov)

    diffs = mu_all - cl_mean
    latent_mahal = np.sqrt(np.einsum("ij,jk,ik->i", diffs, cl_cov_inv, diffs)).astype(np.float32)

    # ── Hybrid score: z-normalize each, then weighted sum
    # Normalize each component using classified pool stats (so "normal" sits near 0)
    classified_mask = (meta["kind"] == "classified").values
    re_cl = recon_error[classified_mask]
    lm_cl = latent_mahal[classified_mask]

    # re_z = (recon_error - re_cl.mean()) / (re_cl.std() + 1e-8)
    # lm_z = (latent_mahal - lm_cl.mean()) / (lm_cl.std() + 1e-8)
    # With this:
    def robust_z(x, ref):
       med = np.median(ref)
       mad = np.median(np.abs(ref - med)) * 1.4826  # scale to ~std for normal
       return (x - med) / (mad + 1e-8)

    re_z = robust_z(recon_error, re_cl)

    # lm_z = robust_z(latent_mahal, lm_cl)
    # hybrid_score = W_RECON * re_z + W_LATENT * lm_z
    latent_mahal_log = np.log1p(latent_mahal)
    lm_cl_log = latent_mahal_log[classified_mask]
    lm_z = robust_z(latent_mahal_log, lm_cl_log)
    hybrid_score = W_RECON * re_z + W_LATENT * lm_z
    meta = meta.copy()
    meta["recon_error"] = recon_error
    meta["latent_mahal"] = latent_mahal
    meta["tevae_error"] = hybrid_score   # keep the column name for downstream compat

    # ── Threshold
    cl_scores = hybrid_score[classified_mask]
    cl_mean_s = float(cl_scores.mean())
    cl_std_s = float(cl_scores.std())
    threshold = cl_mean_s + THRESHOLD_K * cl_std_s
    meta["is_anomaly"] = (hybrid_score > threshold).astype(int)

    n_total_anom = int(meta["is_anomaly"].sum())
    n_ucl_anom = int(meta[meta["kind"] == "unclassified"]["is_anomaly"].sum())
    n_cl_anom = int(meta[meta["kind"] == "classified"]["is_anomaly"].sum())
    n_ucl_total = int((meta["kind"] == "unclassified").sum())
    n_cl_total = int(classified_mask.sum())

    print(f"\nHybrid score = {W_RECON}*z(recon) + {W_LATENT}*z(latent_mahal)")
    print(f"Threshold (mean + {THRESHOLD_K}σ on classified): {threshold:.4f}")
    print(f"  Classified flagged:     {n_cl_anom:,} ({100*n_cl_anom/n_cl_total:.3f}%)")
    print(f"  Unclassified flagged:   {n_ucl_anom:,} ({100*n_ucl_anom/n_ucl_total:.3f}%)")

    out_path = RESULTS_DIR / "tevae_scores.tsv"
    meta.sort_values("tevae_error", ascending=False).to_csv(
        out_path, sep="\t", index=False
    )
    print(f"\nSaved: {out_path}")

    with open(RESULTS_DIR / "tevae_threshold.txt", "w") as f:
        f.write(f"TE-VAE hybrid anomaly score\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Score = {W_RECON} * z(recon_error) + {W_LATENT} * z(latent_mahal)\n")
        f.write(f"Threshold: mean + {THRESHOLD_K} * std (on classified)\n")
        f.write(f"  threshold = {threshold:.4f}\n\n")
        f.write(f"KL_WEIGHT = {KL_WEIGHT}\n\n")
        f.write(f"Total flagged:        {n_total_anom:,} / {len(meta):,}\n")
        f.write(f"Classified flagged:   {n_cl_anom:,} ({100*n_cl_anom/n_cl_total:.3f}%)\n")
        f.write(f"Unclassified flagged: {n_ucl_anom:,} ({100*n_ucl_anom/n_ucl_total:.3f}%)\n")

    # ── Distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    components = [
        ("recon_error", "Reconstruction error", "fig_tevae_recon.png"),
        ("latent_mahal", "Latent Mahalanobis", "fig_tevae_latent.png"),
        ("tevae_error", "Hybrid score (z-recon + z-latent)", "fig_tevae_hybrid.png"),
    ]

    for ax, (col, title, _) in zip(axes, components):
        cl = meta[meta["kind"] == "classified"][col]
        ucl = meta[meta["kind"] == "unclassified"][col]
        ax.hist(cl, bins=80, alpha=0.6, label=f"Classified (n={len(cl):,})",
                color="#0D7377", density=True)
        ax.hist(ucl, bins=80, alpha=0.6, label=f"Unclassified (n={len(ucl):,})",
                color="#C0392B", density=True)
        if col == "tevae_error":
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                       label=f"Threshold = {threshold:.2f}")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=9)

    plt.suptitle("TE-VAE — three score components", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_tevae_distribution_components.png", dpi=150)
    plt.close()
    print("  fig_tevae_distribution_components.png")

    # Single hybrid plot too (for the deck)
    fig, ax = plt.subplots(figsize=(9, 5))
    cl = meta[meta["kind"] == "classified"]["tevae_error"]
    ucl = meta[meta["kind"] == "unclassified"]["tevae_error"]
    ax.hist(cl, bins=80, alpha=0.6, label=f"Classified (n={len(cl):,})",
            color="#0D7377", density=True)
    ax.hist(ucl, bins=80, alpha=0.6, label=f"Unclassified (n={len(ucl):,})",
            color="#C0392B", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold (μ+{THRESHOLD_K}σ = {threshold:.2f})")
    ax.set_xlabel("TE-VAE hybrid anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("TE-VAE hybrid score: classified vs unclassified")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_tevae_distribution.png", dpi=150)
    plt.close()
    print("  fig_tevae_distribution.png")

    print("\nDone.")
    print(f"Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
