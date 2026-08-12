#!/usr/bin/env python3
"""
Embedding Geometry Visualization (Task 5.2).

Creates:
1. t-SNE / UMAP plots of CLIP embeddings colored by domain
2. t-SNE / UMAP plots of CLAP embeddings colored by domain
3. Before/after ExMCR projection comparison
4. Gramian volume heatmaps for matched vs mismatched triples
5. PCA variance explained plots

Usage:
    python scripts/visualize_embeddings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
OUTPUT_DIR = PROJECT_ROOT / "figures"


def collect_embeddings(samples: list) -> dict:
    """Collect all embeddings for visualization."""
    from src.embeddings.aligned_embeddings import AlignedEmbedder
    from src.config.settings import EXMCR_WEIGHTS_PATH

    embedder = AlignedEmbedder(target_dim=512)

    # Try loading ExMCR
    exmcr = None
    if EXMCR_WEIGHTS_PATH.exists():
        from src.embeddings.space_alignment import ExMCRProjector
        exmcr = ExMCRProjector(weights_path=str(EXMCR_WEIGHTS_PATH))
        if exmcr.is_identity:
            exmcr = None

    data = {
        "text_clip": [], "text_clap": [],
        "image": [], "audio": [],
        "audio_projected": [],  # ExMCR projected
        "domains": [], "conditions": [],
        "sample_ids": [],
    }

    for i, s in enumerate(samples):
        data["text_clip"].append(embedder.embed_text(s["prompt_text"]).squeeze())
        data["text_clap"].append(embedder.embed_text_for_audio(s["prompt_text"]).squeeze())
        data["image"].append(embedder.embed_image(s["image_path"]).squeeze())
        data["audio"].append(embedder.embed_audio(s["audio_path"]).squeeze())

        if exmcr is not None:
            data["audio_projected"].append(exmcr.project_audio(data["audio"][-1]).squeeze())
        else:
            data["audio_projected"].append(data["audio"][-1])

        data["domains"].append(s.get("domain", "unknown"))
        data["conditions"].append(s.get("condition", "unknown"))
        data["sample_ids"].append(s["sample_id"])
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}", end="\r")

    print()

    # Stack arrays
    for key in ["text_clip", "text_clap", "image", "audio", "audio_projected"]:
        data[key] = np.stack(data[key])

    return data


def plot_tsne_umap(data: dict, output_dir: Path):
    """Create t-SNE and UMAP visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Try UMAP, fall back to t-SNE only
    try:
        from umap import UMAP
        has_umap = True
    except ImportError:
        has_umap = False
        print("  UMAP not installed, using t-SNE only")

    domain_colors = {
        "nature": "#2ca02c",
        "urban": "#d62728",
        "water": "#1f77b4",
        "mixed": "#ff7f0e",
        "other": "#9467bd",
        "unknown": "#7f7f7f",
    }

    domains = data["domains"]
    colors = [domain_colors.get(d, "#7f7f7f") for d in domains]

    modalities = {
        "CLIP text": data["text_clip"],
        "CLIP image": data["image"],
        "CLAP text": data["text_clap"],
        "CLAP audio": data["audio"],
    }

    # t-SNE plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Embedding Geometry (t-SNE)", fontsize=14, fontweight="bold")

    for idx, (name, embs) in enumerate(modalities.items()):
        ax = axes[idx // 2][idx % 2]
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(embs) - 1))
        coords = tsne.fit_transform(embs)

        for domain in sorted(set(domains)):
            mask = [d == domain for d in domains]
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=domain_colors.get(domain, "#7f7f7f"),
                label=domain, alpha=0.7, s=60, edgecolors="white", linewidths=0.5,
            )
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_dir / "fig_embedding_tsne.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_embedding_tsne.pdf")

    # UMAP plots
    if has_umap:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Embedding Geometry (UMAP)", fontsize=14, fontweight="bold")

        for idx, (name, embs) in enumerate(modalities.items()):
            ax = axes[idx // 2][idx % 2]
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(10, len(embs) - 1))
            coords = reducer.fit_transform(embs)

            for domain in sorted(set(domains)):
                mask = [d == domain for d in domains]
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=domain_colors.get(domain, "#7f7f7f"),
                    label=domain, alpha=0.7, s=60, edgecolors="white", linewidths=0.5,
                )
            ax.set_title(name, fontsize=12)
            ax.legend(fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fig.savefig(output_dir / "fig_embedding_umap.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: fig_embedding_umap.pdf")


def plot_exmcr_comparison(data: dict, output_dir: Path):
    """Before/after ExMCR projection comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ExMCR Projection: Audio → CLIP Space", fontsize=14, fontweight="bold")

    domain_colors = {
        "nature": "#2ca02c", "urban": "#d62728",
        "water": "#1f77b4", "mixed": "#ff7f0e",
    }
    domains = data["domains"]

    # Before: Image (CLIP) + Audio (CLAP) — different spaces
    combined_before = np.vstack([data["image"], data["audio"]])
    labels_before = ["image"] * len(data["image"]) + ["audio"] * len(data["audio"])
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(combined_before) - 1))
    coords_before = tsne.fit_transform(combined_before)

    n = len(data["image"])
    axes[0].scatter(coords_before[:n, 0], coords_before[:n, 1],
                    c="steelblue", label="Image (CLIP)", marker="o", alpha=0.7, s=60)
    axes[0].scatter(coords_before[n:, 0], coords_before[n:, 1],
                    c="coral", label="Audio (CLAP)", marker="^", alpha=0.7, s=60)
    axes[0].set_title("Before ExMCR\n(Different Spaces)", fontsize=11)
    axes[0].legend()
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # After: Image (CLIP) + Audio (ExMCR→CLIP)
    combined_after = np.vstack([data["image"], data["audio_projected"]])
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=min(15, len(combined_after) - 1))
    coords_after = tsne2.fit_transform(combined_after)

    axes[1].scatter(coords_after[:n, 0], coords_after[:n, 1],
                    c="steelblue", label="Image (CLIP)", marker="o", alpha=0.7, s=60)
    axes[1].scatter(coords_after[n:, 0], coords_after[n:, 1],
                    c="coral", label="Audio (ExMCR→CLIP)", marker="^", alpha=0.7, s=60)
    axes[1].set_title("After ExMCR\n(Shared CLIP Space)", fontsize=11)
    axes[1].legend()
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    fig.savefig(output_dir / "fig_exmcr_comparison.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_exmcr_comparison.pdf")


def plot_gram_heatmaps(data: dict, output_dir: Path):
    """Gramian volume heatmaps for matched vs mismatched triples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.coherence.gram_volume import gram_volume_2d

    n = len(data["image"])

    # Text-Image similarity matrix
    sim_ti = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_ti[i, j] = 1.0 - gram_volume_2d(data["text_clip"][i], data["image"][j])

    # Text-Audio similarity matrix
    sim_ta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_ta[i, j] = 1.0 - gram_volume_2d(data["text_clap"][i], data["audio"][j])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Gramian Coherence Matrices", fontsize=14, fontweight="bold")

    im1 = axes[0].imshow(sim_ti, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Text-Image Gram Coherence", fontsize=11)
    axes[0].set_xlabel("Image index")
    axes[0].set_ylabel("Text index")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(sim_ta, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Text-Audio Gram Coherence", fontsize=11)
    axes[1].set_xlabel("Audio index")
    axes[1].set_ylabel("Text index")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_gram_heatmaps.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_gram_heatmaps.pdf")


def plot_pca_variance(data: dict, output_dir: Path):
    """PCA variance explained plots for each modality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    modalities = {
        "CLIP text": data["text_clip"],
        "CLIP image": data["image"],
        "CLAP text": data["text_clap"],
        "CLAP audio": data["audio"],
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, embs in modalities.items():
        n_comp = min(embs.shape[0], embs.shape[1], 50)
        pca = PCA(n_components=n_comp)
        pca.fit(embs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, n_comp + 1), cumvar, label=name, linewidth=2)

    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=11)
    ax.set_title("PCA Variance Explained by Modality", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_pca_variance.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_pca_variance.pdf")


def plot_gram_volume_distributions(data: dict, output_dir: Path):
    """Gramian volume distributions for baseline vs perturbation conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
    from src.coherence.gram_volume import gram_volume_2d

    n = len(data["image"])

    # Compute per-sample gram volumes (diagonal = matched pairs)
    gram_ti = np.array([
        gram_volume_2d(data["text_clip"][i], data["image"][i]) for i in range(n)
    ])
    gram_ta = np.array([
        gram_volume_2d(data["text_clap"][i], data["audio"][i]) for i in range(n)
    ])

    # Try loading RQ1 data for condition labels
    rq1_path = PROJECT_ROOT / "runs" / "rq1" / "rq1_results.json"
    conditions = data["conditions"]
    unique_conditions = sorted(set(conditions))

    has_condition_split = len(unique_conditions) > 1 and "unknown" not in unique_conditions

    if not has_condition_split:
        # Try loading from RQ1 results
        if rq1_path.exists():
            try:
                with open(rq1_path) as f:
                    rq1 = json.load(f)
                if "results" in rq1:
                    for entry in rq1["results"]:
                        sid = entry.get("sample_id", "")
                        cond = entry.get("condition", "unknown")
                        if sid in data["sample_ids"]:
                            idx = data["sample_ids"].index(sid)
                            conditions[idx] = cond
                    unique_conditions = sorted(set(conditions))
                    has_condition_split = len(unique_conditions) > 1
            except Exception as e:
                print(f"  Warning: Could not parse RQ1 data: {e}")

    if not has_condition_split:
        # Fallback: use diagonal (matched) vs off-diagonal (mismatched) from heatmap
        print("  No condition labels found; using matched vs mismatched (off-diagonal) as proxy")
        matched_ti = gram_ti  # diagonal values
        matched_ta = gram_ta

        # Compute some off-diagonal values for comparison
        offdiag_ti = []
        offdiag_ta = []
        rng = np.random.RandomState(42)
        for i in range(n):
            j = rng.choice([k for k in range(n) if k != i])
            offdiag_ti.append(gram_volume_2d(data["text_clip"][i], data["image"][j]))
            offdiag_ta.append(gram_volume_2d(data["text_clap"][i], data["audio"][j]))
        offdiag_ti = np.array(offdiag_ti)
        offdiag_ta = np.array(offdiag_ta)

        group_a_ti, group_b_ti = matched_ti, offdiag_ti
        group_a_ta, group_b_ta = matched_ta, offdiag_ta
        label_a, label_b = "Matched", "Mismatched"
    else:
        # Split by condition: baseline vs perturbation
        baseline_conds = {"baseline", "matched", "original"}
        mask_base = np.array([c.lower() in baseline_conds for c in conditions])
        mask_pert = ~mask_base

        if mask_base.sum() == 0 or mask_pert.sum() == 0:
            # All same group — use first vs rest of unique conditions
            mask_base = np.array([c == unique_conditions[0] for c in conditions])
            mask_pert = ~mask_base
            label_a = unique_conditions[0]
            label_b = "other"
        else:
            label_a, label_b = "Baseline", "Perturbation"

        group_a_ti, group_b_ti = gram_ti[mask_base], gram_ti[mask_pert]
        group_a_ta, group_b_ta = gram_ta[mask_base], gram_ta[mask_pert]

    def cohens_d(x, y):
        pooled_std = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)
        if pooled_std == 0:
            return 0.0
        return (np.mean(x) - np.mean(y)) / pooled_std

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gramian Volume Distributions", fontsize=14, fontweight="bold")

    for ax, ga, gb, title in [
        (axes[0], group_a_ti, group_b_ti, "Text-Image Gram Volume"),
        (axes[1], group_a_ta, group_b_ta, "Text-Audio Gram Volume"),
    ]:
        bins = np.linspace(
            min(ga.min(), gb.min()) - 0.02,
            max(ga.max(), gb.max()) + 0.02,
            20,
        )
        ax.hist(ga, bins=bins, alpha=0.5, label=label_a, color="#2ca02c", edgecolor="white")
        ax.hist(gb, bins=bins, alpha=0.5, label=label_b, color="#d62728", edgecolor="white")

        ks_stat, ks_p = ks_2samp(ga, gb)
        d = cohens_d(ga, gb)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Gram Volume")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.text(
            0.97, 0.95,
            f"Cohen's d = {d:.3f}\nKS p = {ks_p:.4f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(output_dir / "fig_gram_volume_distributions.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_gram_volume_distributions.pdf")


def plot_coherence_vs_human(data: dict, output_dir: Path):
    """Scatter plots of coherence metrics vs human ratings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from src.coherence.gram_volume import gram_volume_2d

    # Try loading human ratings
    human_path = PROJECT_ROOT / "runs" / "rq3" / "rq3_human_scores.json"
    human_path_alt = PROJECT_ROOT / "runs" / "rq3" / "rq3_results.json"
    cmsci_path = PROJECT_ROOT / "runs" / "rq3" / "rq3_cmsci_results.json"

    human_scores = None
    for hp in [human_path, human_path_alt]:
        if hp.exists():
            try:
                with open(hp) as f:
                    human_data = json.load(f)
                # Format 1: {"S001": {"weighted_score": {"mean": ...}}} (our format)
                first_key = next(iter(human_data), "")
                if first_key.startswith("S") and isinstance(human_data.get(first_key), dict):
                    human_scores = {}
                    for sid, entry in human_data.items():
                        ws = entry.get("weighted_score", {})
                        hmean = ws.get("mean") if isinstance(ws, dict) else None
                        if hmean is not None:
                            human_scores[sid] = float(hmean)
                # Format 2: {"results": [{"sample_id": ..., "human_mean": ...}]}
                elif "results" in human_data:
                    human_scores = {}
                    for entry in human_data["results"]:
                        sid = entry.get("sample_id", "")
                        hmean = entry.get("human_mean", entry.get("human_score", None))
                        if sid and hmean is not None:
                            human_scores[sid] = float(hmean)
                # Format 3: {"samples": [...]}
                elif "samples" in human_data:
                    human_scores = {}
                    for entry in human_data["samples"]:
                        sid = entry.get("sample_id", "")
                        hmean = entry.get("human_mean", entry.get("human_score", None))
                        if sid and hmean is not None:
                            human_scores[sid] = float(hmean)
                if human_scores:
                    break
            except Exception as e:
                print(f"  Warning: Could not parse {hp.name}: {e}")

    if not human_scores:
        print("  Warning: No human rating data found; skipping coherence vs human plot")
        return

    # Load cMSCI results if available
    cmsci_scores = {}
    msci_scores = {}
    if cmsci_path.exists():
        try:
            with open(cmsci_path) as f:
                cmsci_data = json.load(f)
            for entry in cmsci_data.get("results", cmsci_data.get("samples", [])):
                sid = entry.get("sample_id", "")
                cmsci_scores[sid] = entry.get("cmsci_f", entry.get("cmsci", None))
                msci_scores[sid] = entry.get("msci", entry.get("msci_a", None))
        except Exception as e:
            print(f"  Warning: Could not parse cMSCI results: {e}")

    # Align samples
    n = len(data["image"])
    sample_ids = data["sample_ids"]
    conditions = data["conditions"]

    # Compute gram coherence for matched samples
    gram_coh = []
    human_vals = []
    cmsci_vals = []
    msci_vals = []
    cond_list = []
    valid_mask = []

    for i in range(n):
        sid = sample_ids[i]
        if sid not in human_scores:
            continue
        gram_ti = 1.0 - gram_volume_2d(data["text_clip"][i], data["image"][i])
        gram_ta = 1.0 - gram_volume_2d(data["text_clap"][i], data["audio"][i])
        gram_coh.append((gram_ti + gram_ta) / 2.0)
        human_vals.append(human_scores[sid])
        cmsci_vals.append(cmsci_scores.get(sid, None))
        msci_vals.append(msci_scores.get(sid, None))
        cond_list.append(conditions[i])

    if len(human_vals) < 3:
        print("  Warning: Too few matched samples for scatter; skipping")
        return

    gram_coh = np.array(gram_coh)
    human_vals = np.array(human_vals)

    condition_colors = {
        "baseline": "#2ca02c", "matched": "#2ca02c", "original": "#2ca02c",
        "wrong_image": "#d62728", "wrong_audio": "#1f77b4",
        "perturbation": "#d62728", "unknown": "#7f7f7f",
    }

    colors = [condition_colors.get(c.lower(), "#7f7f7f") for c in cond_list]

    n_panels = 1
    has_cmsci = any(v is not None for v in cmsci_vals)
    has_msci = any(v is not None for v in msci_vals)
    if has_cmsci:
        n_panels += 1
    if has_msci:
        n_panels += 1

    fig, axes = plt.subplots(1, max(n_panels, 3), figsize=(5 * max(n_panels, 3), 5))
    if max(n_panels, 3) == 1:
        axes = [axes]
    fig.suptitle("Coherence Metrics vs Human Ratings", fontsize=14, fontweight="bold")

    # Panel 1: Gram coherence
    ax = axes[0]
    ax.scatter(human_vals, gram_coh, c=colors, alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
    rho, pval = spearmanr(human_vals, gram_coh)
    ax.set_xlabel("Human Mean Rating", fontsize=10)
    ax.set_ylabel("Gramian Coherence", fontsize=10)
    ax.set_title("Gramian Coherence vs Human", fontsize=11)
    ax.text(
        0.03, 0.97, f"rho = {rho:.3f}\np = {pval:.4f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    # Add legend for conditions
    for cond, color in condition_colors.items():
        if cond in [c.lower() for c in cond_list]:
            ax.scatter([], [], c=color, label=cond, s=40)
    ax.legend(fontsize=7, loc="lower right")

    # Panel 2: cMSCI variant F
    panel_idx = 1
    if has_cmsci:
        ax = axes[panel_idx]
        cmsci_arr = np.array([v if v is not None else np.nan for v in cmsci_vals])
        valid = ~np.isnan(cmsci_arr)
        ax.scatter(human_vals[valid], cmsci_arr[valid], c=[colors[i] for i in range(len(valid)) if valid[i]],
                   alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
        if valid.sum() >= 3:
            rho2, pval2 = spearmanr(human_vals[valid], cmsci_arr[valid])
        else:
            rho2, pval2 = np.nan, np.nan
        ax.set_xlabel("Human Mean Rating", fontsize=10)
        ax.set_ylabel("cMSCI (Variant F)", fontsize=10)
        ax.set_title("cMSCI Variant F vs Human", fontsize=11)
        ax.text(
            0.03, 0.97, f"rho = {rho2:.3f}\np = {pval2:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
        panel_idx += 1
    else:
        axes[1].text(0.5, 0.5, "cMSCI data\nnot available", ha="center", va="center",
                     fontsize=11, transform=axes[1].transAxes, color="gray")
        axes[1].set_title("cMSCI Variant F vs Human", fontsize=11)
        panel_idx = 2

    # Panel 3: MSCI variant A
    if has_msci:
        ax = axes[panel_idx]
        msci_arr = np.array([v if v is not None else np.nan for v in msci_vals])
        valid = ~np.isnan(msci_arr)
        ax.scatter(human_vals[valid], msci_arr[valid], c=[colors[i] for i in range(len(valid)) if valid[i]],
                   alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
        if valid.sum() >= 3:
            rho3, pval3 = spearmanr(human_vals[valid], msci_arr[valid])
        else:
            rho3, pval3 = np.nan, np.nan
        ax.set_xlabel("Human Mean Rating", fontsize=10)
        ax.set_ylabel("MSCI (Variant A)", fontsize=10)
        ax.set_title("MSCI Variant A vs Human", fontsize=11)
        ax.text(
            0.03, 0.97, f"rho = {rho3:.3f}\np = {pval3:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
    else:
        ax_last = axes[panel_idx] if panel_idx < len(axes) else axes[-1]
        ax_last.text(0.5, 0.5, "MSCI data\nnot available", ha="center", va="center",
                     fontsize=11, transform=ax_last.transAxes, color="gray")
        ax_last.set_title("MSCI Variant A vs Human", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_coherence_vs_human.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_coherence_vs_human.pdf")


def plot_3way_gramian_heatmap(data: dict, output_dir: Path):
    """3-way Gramian volume strip/swarm plot grouped by condition or domain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.coherence.gram_volume import gram_volume_3d

    n = len(data["image"])

    # Compute 3-way gram volumes (text + image + audio_projected)
    gram_3way = np.array([
        gram_volume_3d(data["text_clip"][i], data["image"][i], data["audio_projected"][i])
        for i in range(n)
    ])

    conditions = data["conditions"]
    unique_conditions = sorted(set(conditions))

    # Decide grouping: condition if varied, else domain
    if len(unique_conditions) <= 1 or all(c == "unknown" for c in conditions):
        grouping = data["domains"]
        group_label = "Domain"
    else:
        grouping = conditions
        group_label = "Condition"

    unique_groups = sorted(set(grouping))

    group_colors = {
        "nature": "#2ca02c", "urban": "#d62728", "water": "#1f77b4",
        "mixed": "#ff7f0e", "other": "#9467bd", "unknown": "#7f7f7f",
        "baseline": "#2ca02c", "matched": "#2ca02c", "original": "#2ca02c",
        "wrong_image": "#d62728", "wrong_audio": "#1f77b4",
        "perturbation": "#d62728",
    }

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(unique_groups) * 2), 6))
    fig.suptitle("3-Way Gramian Volume by " + group_label, fontsize=14, fontweight="bold")

    positions = []
    for gi, group in enumerate(unique_groups):
        mask = [g == group for g in grouping]
        vals = gram_3way[mask]
        color = group_colors.get(group.lower(), "#7f7f7f")

        # Jittered strip plot
        jitter = np.random.RandomState(42).normal(0, 0.06, size=len(vals))
        ax.scatter(
            np.full(len(vals), gi) + jitter, vals,
            c=color, alpha=0.6, s=50, edgecolors="white", linewidths=0.5,
            zorder=3,
        )
        # Mean + std bar
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        ax.errorbar(gi, mean_val, yerr=std_val, fmt="D", color="black",
                     markersize=8, capsize=6, capthick=1.5, zorder=4)
        ax.text(
            gi, mean_val + std_val + 0.02,
            f"{mean_val:.3f}\n+/-{std_val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
        positions.append(gi)

    ax.set_xticks(positions)
    ax.set_xticklabels(unique_groups, fontsize=10)
    ax.set_xlabel(group_label, fontsize=11)
    ax.set_ylabel("3-Way Gram Volume", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_gram_3way_strip.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_gram_3way_strip.pdf")


def plot_geometric_diagram(output_dir: Path):
    """Conceptual diagram illustrating Gramian volume geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Gramian Volume: Geometric Intuition", fontsize=14, fontweight="bold")

    # --- Left panel: High Coherence (low volume) ---
    ax = axes[0]
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.3, 1.2)
    ax.set_aspect("equal")
    ax.set_title("High Coherence", fontsize=12, fontweight="bold", color="#2ca02c")

    origin = (0, 0)
    # Nearly parallel arrows
    arrows_high = [
        (0.0, 0.0, 1.2, 0.75),
        (0.0, 0.0, 1.25, 0.65),
        (0.0, 0.0, 1.15, 0.80),
    ]
    arrow_colors = ["#1f77b4", "#d62728", "#ff7f0e"]
    arrow_labels = ["Text", "Image", "Audio"]

    for (x0, y0, x1, y1), color, label in zip(arrows_high, arrow_colors, arrow_labels):
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="->,head_width=6,head_length=4",
            color=color, linewidth=2.5, zorder=3,
        )
        ax.add_patch(arrow)
        ax.text(x1 + 0.03, y1 + 0.03, label, fontsize=9, color=color, fontweight="bold")

    # Shaded parallelogram (small area)
    poly_high = Polygon(
        [origin, (1.2, 0.75), (1.2 + 1.15, 0.75 + 0.80), (1.15, 0.80)],
        alpha=0.15, facecolor="#2ca02c", edgecolor="#2ca02c", linewidth=1.5, linestyle="--",
    )
    ax.add_patch(poly_high)
    ax.text(0.7, 0.15, "Low Gramian\nVolume", fontsize=10, ha="center",
            style="italic", color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Right panel: Low Coherence (high volume) ---
    ax = axes[1]
    ax.set_xlim(-0.2, 1.8)
    ax.set_ylim(-0.5, 1.4)
    ax.set_aspect("equal")
    ax.set_title("Low Coherence", fontsize=12, fontweight="bold", color="#d62728")

    arrows_low = [
        (0.0, 0.0, 1.4, 0.15),
        (0.0, 0.0, 0.5, 1.2),
        (0.0, 0.0, 1.2, 0.9),
    ]

    for (x0, y0, x1, y1), color, label in zip(arrows_low, arrow_colors, arrow_labels):
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="->,head_width=6,head_length=4",
            color=color, linewidth=2.5, zorder=3,
        )
        ax.add_patch(arrow)
        ax.text(x1 + 0.03, y1 + 0.03, label, fontsize=9, color=color, fontweight="bold")

    # Shaded parallelogram (large area)
    poly_low = Polygon(
        [origin, (1.4, 0.15), (1.4 + 0.5, 0.15 + 1.2), (0.5, 1.2)],
        alpha=0.15, facecolor="#d62728", edgecolor="#d62728", linewidth=1.5, linestyle="--",
    )
    ax.add_patch(poly_low)
    ax.text(0.9, -0.25, "High Gramian\nVolume", fontsize=10, ha="center",
            style="italic", color="#d62728",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_geometric_diagram.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_geometric_diagram.pdf")


def plot_complementarity_visualization(data: dict, output_dir: Path):
    """Scatter: image-audio coherence vs complementarity, both against human ratings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from src.coherence.gram_volume import gram_volume_2d

    # Try loading human ratings
    human_scores = None
    for hp in [
        PROJECT_ROOT / "runs" / "rq3" / "rq3_human_scores.json",
        PROJECT_ROOT / "runs" / "rq3" / "rq3_results.json",
    ]:
        if hp.exists():
            try:
                with open(hp) as f:
                    hdata = json.load(f)
                # Format 1: {"S001": {"weighted_score": {"mean": ...}}}
                first_key = next(iter(hdata), "")
                if first_key.startswith("S") and isinstance(hdata.get(first_key), dict):
                    human_scores = {}
                    for sid, entry in hdata.items():
                        ws = entry.get("weighted_score", {})
                        hmean = ws.get("mean") if isinstance(ws, dict) else None
                        if hmean is not None:
                            human_scores[sid] = float(hmean)
                else:
                    entries = hdata.get("results", hdata.get("samples", []))
                    human_scores = {}
                    for entry in entries:
                        sid = entry.get("sample_id", "")
                        hmean = entry.get("human_mean", entry.get("human_score", None))
                        if sid and hmean is not None:
                            human_scores[sid] = float(hmean)
                if human_scores:
                    break
            except Exception as e:
                print(f"  Warning: Could not parse {hp.name}: {e}")

    if not human_scores:
        print("  Warning: No human rating data found; skipping complementarity plot")
        return

    n = len(data["image"])
    sample_ids = data["sample_ids"]
    conditions = data["conditions"]

    gram_ia_coh = []
    gram_ia_vol = []
    human_vals = []
    cond_list = []

    for i in range(n):
        sid = sample_ids[i]
        if sid not in human_scores:
            continue
        vol = gram_volume_2d(data["image"][i], data["audio_projected"][i])
        gram_ia_vol.append(vol)
        gram_ia_coh.append(1.0 - vol)
        human_vals.append(human_scores[sid])
        cond_list.append(conditions[i])

    if len(human_vals) < 3:
        print("  Warning: Too few matched samples; skipping complementarity plot")
        return

    gram_ia_coh = np.array(gram_ia_coh)
    gram_ia_vol = np.array(gram_ia_vol)
    human_vals = np.array(human_vals)

    condition_colors = {
        "baseline": "#2ca02c", "matched": "#2ca02c", "original": "#2ca02c",
        "wrong_image": "#d62728", "wrong_audio": "#1f77b4",
        "perturbation": "#d62728", "unknown": "#7f7f7f",
    }
    colors = [condition_colors.get(c.lower(), "#7f7f7f") for c in cond_list]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Image-Audio: Coherence vs Complementarity", fontsize=14, fontweight="bold")

    # Left: coherence (1 - volume) vs human
    ax = axes[0]
    ax.scatter(human_vals, gram_ia_coh, c=colors, alpha=0.7, s=60,
               edgecolors="white", linewidths=0.5)
    rho1, p1 = spearmanr(human_vals, gram_ia_coh)
    ax.set_xlabel("Human Mean Rating", fontsize=10)
    ax.set_ylabel("Image-Audio Gram Coherence (1 - vol)", fontsize=10)
    ax.set_title("Coherence vs Human\n(expect negative rho)", fontsize=11)
    ax.text(
        0.03, 0.97, f"rho = {rho1:.3f}\np = {p1:.4f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    for cond, color in condition_colors.items():
        if cond in [c.lower() for c in cond_list]:
            ax.scatter([], [], c=color, label=cond, s=40)
    ax.legend(fontsize=7, loc="lower right")

    # Right: volume (complementarity) vs human
    ax = axes[1]
    ax.scatter(human_vals, gram_ia_vol, c=colors, alpha=0.7, s=60,
               edgecolors="white", linewidths=0.5)
    rho2, p2 = spearmanr(human_vals, gram_ia_vol)
    ax.set_xlabel("Human Mean Rating", fontsize=10)
    ax.set_ylabel("Image-Audio Gram Volume (complementarity)", fontsize=10)
    ax.set_title("Complementarity vs Human\n(expect positive rho)", fontsize=11)
    ax.text(
        0.03, 0.97, f"rho = {rho2:.3f}\np = {p2:.4f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )
    ax.text(
        0.97, 0.03, "complementary != redundant",
        transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
        horizontalalignment="right", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(output_dir / "fig_complementarity.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_complementarity.pdf")


def main():
    print("=" * 70)
    print("Embedding Geometry Visualization")
    print("=" * 70)

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]
    print(f"  {len(samples)} samples")

    print("\n--- Collecting Embeddings ---")
    data = collect_embeddings(samples)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating t-SNE/UMAP Plots ---")
    plot_tsne_umap(data, OUTPUT_DIR)

    print("\n--- Generating ExMCR Comparison ---")
    plot_exmcr_comparison(data, OUTPUT_DIR)

    print("\n--- Generating Gramian Heatmaps ---")
    plot_gram_heatmaps(data, OUTPUT_DIR)

    print("\n--- Generating PCA Variance Plots ---")
    plot_pca_variance(data, OUTPUT_DIR)

    print("\n--- Generating Gramian Volume Distributions ---")
    plot_gram_volume_distributions(data, OUTPUT_DIR)

    print("\n--- Generating Coherence vs Human Scatter ---")
    plot_coherence_vs_human(data, OUTPUT_DIR)

    print("\n--- Generating 3-Way Gramian Strip Plot ---")
    plot_3way_gramian_heatmap(data, OUTPUT_DIR)

    print("\n--- Generating Geometric Diagram ---")
    plot_geometric_diagram(OUTPUT_DIR)

    print("\n--- Generating Complementarity Visualization ---")
    plot_complementarity_visualization(data, OUTPUT_DIR)

    print(f"\n  All figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
