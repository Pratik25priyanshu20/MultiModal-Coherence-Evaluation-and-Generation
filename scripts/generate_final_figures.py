"""Generate the 6 final paper figures (fig1_final.pdf – fig6_final.pdf)."""

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
FIG_DIR = PROJECT / "figures"

SAMPLES_PATH = PROJECT / "runs" / "rq3" / "rq3_samples.json"


def _load_evaluation_data():
    """Load samples + human scores + fresh cMSCI scores (same as full evaluation)."""
    # Force single-clip CLAP before engine import (consistent with index)
    import src.embeddings.aligned_embeddings as _ae
    _ae.AUDIO_USE_WINDOWED = False

    from scripts.optimize_cmsci import load_human_scores
    from src.coherence.cmsci_engine import CalibratedCoherenceEngine
    from src.config.settings import (
        CMSCI_CALIBRATION_PATH,
        EXMCR_WEIGHTS_PATH,
        BRIDGE_WEIGHTS_PATH,
        PROB_CLIP_ADAPTER_PATH,
        PROB_CLAP_ADAPTER_PATH,
    )

    with open(SAMPLES_PATH) as f:
        all_samples = json.load(f)["samples"]

    human_scores = load_human_scores()

    engine = CalibratedCoherenceEngine(
        calibration_path=str(CMSCI_CALIBRATION_PATH) if CMSCI_CALIBRATION_PATH.exists() else None,
        exmcr_weights_path=str(EXMCR_WEIGHTS_PATH) if EXMCR_WEIGHTS_PATH.exists() else None,
        bridge_path=str(BRIDGE_WEIGHTS_PATH) if BRIDGE_WEIGHTS_PATH.exists() else None,
        prob_clip_adapter_path=str(PROB_CLIP_ADAPTER_PATH) if PROB_CLIP_ADAPTER_PATH.exists() else None,
        prob_clap_adapter_path=str(PROB_CLAP_ADAPTER_PATH) if PROB_CLAP_ADAPTER_PATH.exists() else None,
        negative_bank_enabled=True,
    )

    # Force single-clip CLAP (matching the full evaluation that produced rho=0.519)
    engine.embedder._use_windowed = False

    paired = []
    for s in all_samples:
        sid = s["sample_id"]
        if sid not in human_scores:
            continue

        result = engine.evaluate(
            text=s["prompt_text"],
            image_path=s.get("image_path"),
            audio_path=s.get("audio_path"),
            domain=s.get("domain", ""),
        )

        cmsci = result["variant_scores"]["F_full_cmsci"] or result["cmsci"]
        human_val = human_scores[sid]["weighted_score"]["mean"]

        paired.append({
            "sample_id": sid,
            "condition": s.get("condition", "baseline"),
            "domain": s.get("domain", ""),
            "human_score": human_val,
            "cmsci_score": cmsci,
            "st_i": result["scores"]["st_i"],
            "st_a": result["scores"]["st_a"],
        })

    print(f"  Loaded {len(paired)} paired samples")
    return paired


# ── Fig 1: copy geometric diagram ─────────────────────────────────────────
def fig1_geometric_diagram():
    src = FIG_DIR / "fig_geometric_diagram.pdf"
    dst = FIG_DIR / "fig1_final.pdf"
    shutil.copy2(src, dst)
    print(f"Fig 1: copied {src.name} → {dst.name}")


# ── Fig 2: cMSCI vs Human scatter (main result) ──────────────────────────
def fig2_cmsci_vs_human(paired):
    colors = {"baseline": "#2ecc71", "wrong_image": "#e74c3c", "wrong_audio": "#3498db"}
    labels = {"baseline": "Baseline", "wrong_image": "Wrong Image", "wrong_audio": "Wrong Audio"}

    fig, ax = plt.subplots(figsize=(7, 6))

    for cond in ["baseline", "wrong_image", "wrong_audio"]:
        xs = [s["human_score"] for s in paired if s["condition"] == cond]
        ys = [s["cmsci_score"] for s in paired if s["condition"] == cond]
        ax.scatter(xs, ys, c=colors[cond], label=labels[cond],
                   s=60, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)

    # Regression line over all points
    all_h = [s["human_score"] for s in paired]
    all_c = [s["cmsci_score"] for s in paired]
    rho, p = stats.spearmanr(all_h, all_c)

    z = np.polyfit(all_h, all_c, 1)
    xline = np.linspace(min(all_h) - 0.02, max(all_h) + 0.02, 100)
    yline = np.polyval(z, xline)
    ax.plot(xline, yline, color="#7f8c8d", linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)

    ax.set_xlabel("Human Mean Coherence Rating", fontsize=12)
    ax.set_ylabel("cMSCI Score", fontsize=12)
    ax.set_title("cMSCI vs Human Coherence Judgments", fontsize=14, fontweight="bold")

    # Annotation box
    textstr = f"Spearman ρ = {rho:.3f} (p = {p:.4f})\nN = {len(paired)}, k = 3 raters"
    props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="#bdc3c7", alpha=0.9)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props)

    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG_DIR / "fig2_final.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 2: cMSCI vs Human scatter (ρ={rho:.3f}, p={p:.4f}) → {out.name}")


# ── Fig 3: Baseline comparison bar chart ──────────────────────────────────
def fig3_baseline_comparison():
    with open(PROJECT / "runs" / "full_evaluation" / "full_evaluation.json") as f:
        data = json.load(f)

    results = data["results"]

    # Build list: (name, rho, significant)
    methods = []
    name_map = {
        "cMSCI": "cMSCI (ours)",
        "VLM_Judge": "VLM-as-Judge (LLaVA-7B)",
        "CCA": "CCA",
        "cosine_znorm": "Cosine + z-norm",
        "concatenated_cosine": "Concatenated cosine",
        "BLIPScore": "BLIPScore + CLAPScore",
        "MSCI": "MSCI (cosine baseline)",
        "raw_cosine": "Raw cosine",
        "CLIPScore": "CLIPScore + CLAPScore",
        "RegCCA": "Regularized CCA",
        "retrieval_rank": "Retrieval rank",
    }

    for key, display in name_map.items():
        if key in results:
            r = results[key]
            sig = r.get("significant", r.get("p", 1.0) < 0.05)
            methods.append((display, r["rho"], sig))

    # Sort by rho descending
    methods.sort(key=lambda x: x[1], reverse=True)

    names = [m[0] for m in methods]
    rhos = [m[1] for m in methods]
    sigs = [m[2] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(range(len(names)), rhos, height=0.65,
                   color=["#2980b9" if s else "#bdc3c7" for s in sigs],
                   edgecolor="white", linewidth=0.5)

    # Highlight cMSCI
    bars[0].set_color("#e74c3c")
    bars[0].set_edgecolor("#c0392b")
    bars[0].set_linewidth(1.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ with Human Ratings", fontsize=12)
    ax.set_title("Baseline Comparison: Correlation with Human Coherence Judgments",
                 fontsize=13, fontweight="bold")

    # Add value labels
    for i, (rho, sig) in enumerate(zip(rhos, sigs)):
        marker = " *" if sig else ""
        ax.text(rho + 0.008, i, f"{rho:.3f}{marker}", va="center", fontsize=9,
                fontweight="bold" if i == 0 else "normal")

    # Significance threshold line
    ax.axvline(x=0, color="gray", linewidth=0.5)

    # Legend
    sig_patch = mpatches.Patch(color="#2980b9", label="Significant (p < 0.05)")
    ns_patch = mpatches.Patch(color="#bdc3c7", label="Not significant")
    ours_patch = mpatches.Patch(color="#e74c3c", label="cMSCI (ours)")
    ax.legend(handles=[ours_patch, sig_patch, ns_patch], loc="lower right", fontsize=9)

    ax.set_xlim(-0.05, 0.62)
    ax.grid(True, axis="x", alpha=0.2)

    plt.tight_layout()
    out = FIG_DIR / "fig3_final.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 3: Baseline comparison ({len(methods)} methods) → {out.name}")


# ── Fig 4: copy ablation effect sizes ─────────────────────────────────────
def fig4_ablation():
    src = FIG_DIR / "fig14_cmsci_ablation_f.pdf"
    dst = FIG_DIR / "fig4_final.pdf"
    shutil.copy2(src, dst)
    print(f"Fig 4: copied {src.name} → {dst.name}")


# ── Fig 5: copy score distributions ──────────────────────────────────────
def fig5_distributions():
    src = FIG_DIR / "fig15_cmsci_distributions_f.pdf"
    dst = FIG_DIR / "fig5_final.pdf"
    shutil.copy2(src, dst)
    print(f"Fig 5: copied {src.name} → {dst.name}")


# ── Fig 6: Channel decomposition (expanded data) ─────────────────────────
def fig6_channel_decomposition():
    # Per-condition means from Table 9 (expanded training data results)
    conditions = ["Baseline", "Wrong Image", "Wrong Audio"]
    st_i = [0.242, 0.146, 0.239]
    st_a = [0.531, 0.519, 0.217]
    cmsci = [0.444, 0.407, 0.223]

    x = np.arange(len(conditions))
    width = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: stacked bar chart (st_i + st_a)
    bars1 = ax1.bar(x, st_i, width, label="Text–Image (CLIP)", color="#e74c3c", alpha=0.85)
    bars2 = ax1.bar(x, st_a, width, bottom=st_i, label="Text–Audio (CLAP)", color="#3498db", alpha=0.85)

    # Add value labels
    for i in range(len(conditions)):
        ax1.text(x[i], st_i[i] / 2, f"{st_i[i]:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax1.text(x[i], st_i[i] + st_a[i] / 2, f"{st_a[i]:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=11)
    ax1.set_ylabel("Cosine Similarity", fontsize=12)
    ax1.set_title("Channel Decomposition\n(Expanded Training Data)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_ylim(0, 0.85)
    ax1.grid(True, axis="y", alpha=0.2)

    # Annotations for key changes
    ax1.annotate("↓40%", xy=(1, 0.146), xytext=(1.35, 0.08),
                fontsize=9, color="#c0392b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
    ax1.annotate("↓59%", xy=(2, 0.239 + 0.217), xytext=(2.35, 0.55),
                fontsize=9, color="#2471a3", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2471a3", lw=1.2))

    # Right panel: cMSCI scores
    bar_colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars3 = ax2.bar(x, cmsci, width, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1)

    for i in range(len(conditions)):
        ax2.text(x[i], cmsci[i] + 0.015, f"{cmsci[i]:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=11)
    ax2.set_ylabel("cMSCI Score", fontsize=12)
    ax2.set_title("cMSCI Per-Condition Scores\n(Expanded Training Data)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 0.55)
    ax2.grid(True, axis="y", alpha=0.2)

    # Score gap annotation
    ax2.annotate(f"Δ = {cmsci[0] - cmsci[2]:.3f}\n(audio detected)",
                xy=(2, cmsci[2]), xytext=(2.4, 0.35),
                fontsize=9, color="#2c3e50", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2))

    plt.tight_layout()
    out = FIG_DIR / "fig6_final.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 6: Channel decomposition (expanded data) → {out.name}")


if __name__ == "__main__":
    print("Generating 6 final paper figures...\n")

    # Load evaluation data once (shared by fig2 and fig6)
    print("Loading evaluation data & computing fresh cMSCI scores...")
    paired = _load_evaluation_data()

    fig1_geometric_diagram()
    fig2_cmsci_vs_human(paired)
    fig3_baseline_comparison()
    fig4_ablation()
    fig5_distributions()
    fig6_channel_decomposition()
    print("\nDone. All figures saved to figures/fig[1-6]_final.pdf")
