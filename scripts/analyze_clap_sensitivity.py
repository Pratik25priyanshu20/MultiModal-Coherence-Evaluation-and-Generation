#!/usr/bin/env python3
"""
CLAP Sensitivity vs Generator Quality Analysis.

Investigates whether CLAP audio embeddings are less discriminative than
CLIP image embeddings, and whether audio quality predicts CLAP scores.

Produces:
1. Channel-wise effect sizes (Cohen's d) for CLIP vs CLAP perturbation sensitivity
2. Per-domain CLAP discrimination (AUC for matched vs mismatched)
3. Audio quality correlation with st_a scores
4. Four PDF figures

Usage:
    python scripts/analyze_clap_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "runs" / "clap_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Palette
COL_CLIP = "#e74c3c"
COL_CLAP = "#3498db"
COL_BASELINE = "#2ecc71"
COL_MISMATCH = "#e67e22"


# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════

def load_rq1_results() -> List[Dict[str, Any]]:
    """Load RQ1 experiment results.

    RQ1 data has 270 runs (30 prompts x 3 seeds x 3 conditions:
    baseline, wrong_image, wrong_audio). Each run has st_i (CLIP
    text-image cosine sim) and st_a (CLAP text-audio cosine sim),
    plus condition and domain fields.

    Returns:
        List of per-run result dicts.
    """
    # Primary path
    primary = PROJECT_ROOT / "runs" / "rq1" / "rq1_results.json"
    if primary.exists():
        with open(primary) as f:
            data = json.load(f)
        print(f"  Loaded {len(data['results'])} runs from {primary}")
        return data["results"]

    # Fallback: glob for rq1*.json
    import glob
    candidates = sorted(
        PROJECT_ROOT.glob("runs/rq1*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No RQ1 results found. Expected runs/rq1/rq1_results.json"
        )
    with open(candidates[0]) as f:
        data = json.load(f)
    results = data.get("results", data)
    if isinstance(results, dict):
        results = list(results.values())
    print(f"  Loaded {len(results)} runs from {candidates[0]}")
    return results


def load_rq3_samples() -> List[Dict[str, Any]]:
    """Load RQ3 samples for audio quality correlation analysis.

    Returns:
        List of sample dicts with audio_path, st_a, etc.
    """
    path = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
    if not path.exists():
        print(f"  WARNING: {path} not found. Skipping quality analysis.")
        return []
    with open(path) as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"  Loaded {len(samples)} RQ3 samples from {path}")
    return samples


# ══════════════════════════════════════════════════════════════
# Analysis Functions
# ══════════════════════════════════════════════════════════════

def _aggregate_by_prompt(
    results: List[Dict[str, Any]],
    condition: str,
    score_key: str,
) -> Dict[str, float]:
    """Average score_key across seeds for a given condition, per prompt."""
    bucket: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        if r.get("condition") == condition and r.get(score_key) is not None:
            bucket[r["prompt_id"]].append(r[score_key])
    return {pid: float(np.mean(vals)) for pid, vals in bucket.items()}


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (pooled standard deviation).

    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_channel_effect_sizes(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute channel-wise effect sizes for perturbation sensitivity.

    For CLIP: Cohen's d between baseline st_i and wrong_image st_i
    For CLAP: Cohen's d between baseline st_a and wrong_audio st_a

    Also computes KS test statistics for distributional differences.

    Returns:
        Dict with clip_d, clap_d, clip_ks, clap_ks and supporting stats.
    """
    # Aggregate per-prompt means (average across seeds)
    clip_baseline = _aggregate_by_prompt(results, "baseline", "st_i")
    clip_mismatch = _aggregate_by_prompt(results, "wrong_image", "st_i")
    clap_baseline = _aggregate_by_prompt(results, "baseline", "st_a")
    clap_mismatch = _aggregate_by_prompt(results, "wrong_audio", "st_a")

    # Align prompts
    clip_prompts = sorted(set(clip_baseline) & set(clip_mismatch))
    clap_prompts = sorted(set(clap_baseline) & set(clap_mismatch))

    clip_base_arr = np.array([clip_baseline[p] for p in clip_prompts])
    clip_mis_arr = np.array([clip_mismatch[p] for p in clip_prompts])
    clap_base_arr = np.array([clap_baseline[p] for p in clap_prompts])
    clap_mis_arr = np.array([clap_mismatch[p] for p in clap_prompts])

    # Cohen's d (baseline - mismatch; positive = baseline higher)
    clip_d = _cohens_d(clip_base_arr, clip_mis_arr)
    clap_d = _cohens_d(clap_base_arr, clap_mis_arr)

    # KS test
    clip_ks_stat, clip_ks_p = sp_stats.ks_2samp(clip_base_arr, clip_mis_arr)
    clap_ks_stat, clap_ks_p = sp_stats.ks_2samp(clap_base_arr, clap_mis_arr)

    # Paired t-test for within-prompt differences
    clip_t, clip_t_p = sp_stats.ttest_rel(clip_base_arr, clip_mis_arr)
    clap_t, clap_t_p = sp_stats.ttest_rel(clap_base_arr, clap_mis_arr)

    return {
        "clip_d": round(clip_d, 4),
        "clap_d": round(clap_d, 4),
        "clip_ks_stat": round(float(clip_ks_stat), 4),
        "clip_ks_p": round(float(clip_ks_p), 6),
        "clap_ks_stat": round(float(clap_ks_stat), 4),
        "clap_ks_p": round(float(clap_ks_p), 6),
        "clip_ttest_t": round(float(clip_t), 4),
        "clip_ttest_p": round(float(clip_t_p), 6),
        "clap_ttest_t": round(float(clap_t), 4),
        "clap_ttest_p": round(float(clap_t_p), 6),
        "clip_baseline_mean": round(float(np.mean(clip_base_arr)), 4),
        "clip_baseline_std": round(float(np.std(clip_base_arr, ddof=1)), 4),
        "clip_mismatch_mean": round(float(np.mean(clip_mis_arr)), 4),
        "clip_mismatch_std": round(float(np.std(clip_mis_arr, ddof=1)), 4),
        "clap_baseline_mean": round(float(np.mean(clap_base_arr)), 4),
        "clap_baseline_std": round(float(np.std(clap_base_arr, ddof=1)), 4),
        "clap_mismatch_mean": round(float(np.mean(clap_mis_arr)), 4),
        "clap_mismatch_std": round(float(np.std(clap_mis_arr, ddof=1)), 4),
        "n_clip_prompts": len(clip_prompts),
        "n_clap_prompts": len(clap_prompts),
        # Raw arrays for plotting (not serialised)
        "_clip_base": clip_base_arr,
        "_clip_mis": clip_mis_arr,
        "_clap_base": clap_base_arr,
        "_clap_mis": clap_mis_arr,
    }


def compute_domain_auc(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute per-domain AUC for CLIP and CLAP discrimination.

    For each domain, computes AUC for binary classification:
    label 1 = matched (baseline), label 0 = mismatched (wrong_*).

    Returns:
        Dict keyed by domain with clip_auc and clap_auc.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # Collect per-run scores by domain and condition
    domains = sorted({r["domain"] for r in results if r.get("domain")})
    domain_results = {}

    for dom in domains:
        dom_runs = [r for r in results if r.get("domain") == dom]

        # CLIP: baseline st_i (matched=1) vs wrong_image st_i (mismatched=0)
        clip_scores = []
        clip_labels = []
        for r in dom_runs:
            if r["condition"] == "baseline" and r.get("st_i") is not None:
                clip_scores.append(r["st_i"])
                clip_labels.append(1)
            elif r["condition"] == "wrong_image" and r.get("st_i") is not None:
                clip_scores.append(r["st_i"])
                clip_labels.append(0)

        # CLAP: baseline st_a (matched=1) vs wrong_audio st_a (mismatched=0)
        clap_scores = []
        clap_labels = []
        for r in dom_runs:
            if r["condition"] == "baseline" and r.get("st_a") is not None:
                clap_scores.append(r["st_a"])
                clap_labels.append(1)
            elif r["condition"] == "wrong_audio" and r.get("st_a") is not None:
                clap_scores.append(r["st_a"])
                clap_labels.append(0)

        clip_auc = None
        clap_auc = None
        clip_fpr, clip_tpr = None, None
        clap_fpr, clap_tpr = None, None

        if len(set(clip_labels)) == 2 and len(clip_labels) >= 4:
            clip_auc = roc_auc_score(clip_labels, clip_scores)
            clip_fpr, clip_tpr, _ = roc_curve(clip_labels, clip_scores)

        if len(set(clap_labels)) == 2 and len(clap_labels) >= 4:
            clap_auc = roc_auc_score(clap_labels, clap_scores)
            clap_fpr, clap_tpr, _ = roc_curve(clap_labels, clap_scores)

        domain_results[dom] = {
            "clip_auc": round(float(clip_auc), 4) if clip_auc is not None else None,
            "clap_auc": round(float(clap_auc), 4) if clap_auc is not None else None,
            "clip_n": len(clip_labels),
            "clap_n": len(clap_labels),
            "_clip_fpr": clip_fpr,
            "_clip_tpr": clip_tpr,
            "_clap_fpr": clap_fpr,
            "_clap_tpr": clap_tpr,
        }

    # Also compute overall AUC
    all_clip_scores, all_clip_labels = [], []
    all_clap_scores, all_clap_labels = [], []
    for r in results:
        if r["condition"] == "baseline" and r.get("st_i") is not None:
            all_clip_scores.append(r["st_i"])
            all_clip_labels.append(1)
        elif r["condition"] == "wrong_image" and r.get("st_i") is not None:
            all_clip_scores.append(r["st_i"])
            all_clip_labels.append(0)
        if r["condition"] == "baseline" and r.get("st_a") is not None:
            all_clap_scores.append(r["st_a"])
            all_clap_labels.append(1)
        elif r["condition"] == "wrong_audio" and r.get("st_a") is not None:
            all_clap_scores.append(r["st_a"])
            all_clap_labels.append(0)

    from sklearn.metrics import roc_curve as _roc_curve

    overall_clip_auc = roc_auc_score(all_clip_labels, all_clip_scores)
    overall_clap_auc = roc_auc_score(all_clap_labels, all_clap_scores)
    overall_clip_fpr, overall_clip_tpr, _ = _roc_curve(all_clip_labels, all_clip_scores)
    overall_clap_fpr, overall_clap_tpr, _ = _roc_curve(all_clap_labels, all_clap_scores)

    domain_results["_overall"] = {
        "clip_auc": round(float(overall_clip_auc), 4),
        "clap_auc": round(float(overall_clap_auc), 4),
        "_clip_fpr": overall_clip_fpr,
        "_clip_tpr": overall_clip_tpr,
        "_clap_fpr": overall_clap_fpr,
        "_clap_tpr": overall_clap_tpr,
    }

    return domain_results


def compute_quality_correlation(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Correlate audio quality metrics with st_a (CLAP) scores.

    For each RQ3 sample with an audio_path, runs AudioAnalyzer to obtain
    RMS dB and spectral flatness, then computes Spearman correlations
    with st_a.

    Returns:
        Dict with correlation stats and raw arrays for plotting.
    """
    if not samples:
        return {
            "rms_rho": None,
            "rms_p": None,
            "flatness_rho": None,
            "flatness_p": None,
            "n": 0,
        }

    from src.embeddings.audio_analysis import AudioAnalyzer

    analyzer = AudioAnalyzer()

    rms_vals = []
    flatness_vals = []
    sta_vals = []
    sample_ids = []

    for i, s in enumerate(samples):
        audio_path = s.get("audio_path")
        st_a = s.get("st_a")
        if audio_path is None or st_a is None:
            continue

        audio_file = Path(audio_path)
        if not audio_file.is_absolute():
            audio_file = PROJECT_ROOT / audio_file
        if not audio_file.exists():
            print(f"    Skip {s.get('sample_id', i)}: audio not found at {audio_file}")
            continue

        try:
            report = analyzer.analyze(str(audio_file))
            rms_vals.append(report.rms_db)
            flatness_vals.append(report.spectral_flatness_mean)
            sta_vals.append(st_a)
            sample_ids.append(s.get("sample_id", f"S{i:03d}"))
        except Exception as e:
            print(f"    Skip {s.get('sample_id', i)}: analysis error: {e}")
            continue

        print(f"  [{i+1}/{len(samples)}] analyzed {s.get('sample_id', '')}", end="\r")
    print()

    if len(sta_vals) < 3:
        print("  WARNING: Too few valid audio samples for correlation analysis.")
        return {
            "rms_rho": None,
            "rms_p": None,
            "flatness_rho": None,
            "flatness_p": None,
            "n": len(sta_vals),
        }

    rms_arr = np.array(rms_vals)
    flat_arr = np.array(flatness_vals)
    sta_arr = np.array(sta_vals)

    rms_rho, rms_p = sp_stats.spearmanr(rms_arr, sta_arr)
    flat_rho, flat_p = sp_stats.spearmanr(flat_arr, sta_arr)

    return {
        "rms_rho": round(float(rms_rho), 4),
        "rms_p": round(float(rms_p), 6),
        "flatness_rho": round(float(flat_rho), 4),
        "flatness_p": round(float(flat_p), 6),
        "n": len(sta_vals),
        "rms_mean": round(float(np.mean(rms_arr)), 2),
        "rms_std": round(float(np.std(rms_arr, ddof=1)), 2),
        "flatness_mean": round(float(np.mean(flat_arr)), 4),
        "flatness_std": round(float(np.std(flat_arr, ddof=1)), 4),
        "_rms": rms_arr,
        "_flatness": flat_arr,
        "_st_a": sta_arr,
        "_sample_ids": sample_ids,
    }


# ══════════════════════════════════════════════════════════════
# Plotting Functions
# ══════════════════════════════════════════════════════════════

def plot_clap_distributions(effect_sizes: Dict[str, Any]) -> Path:
    """Plot overlapping histograms of CLIP and CLAP score distributions.

    Two side-by-side panels:
      Left:  st_i for baseline vs wrong_image (annotated with Cohen's d)
      Right: st_a for baseline vs wrong_audio (annotated with Cohen's d)

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: CLIP ──
    ax = axes[0]
    clip_base = effect_sizes["_clip_base"]
    clip_mis = effect_sizes["_clip_mis"]
    bins_clip = np.linspace(
        min(clip_base.min(), clip_mis.min()) - 0.02,
        max(clip_base.max(), clip_mis.max()) + 0.02,
        25,
    )
    ax.hist(clip_base, bins=bins_clip, alpha=0.6, color=COL_BASELINE,
            label="Baseline (matched)", edgecolor="white", linewidth=0.5)
    ax.hist(clip_mis, bins=bins_clip, alpha=0.6, color=COL_CLIP,
            label="Wrong image (mismatched)", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("CLIP text-image similarity ($s_{t,i}$)")
    ax.set_ylabel("Count (prompts)")
    ax.set_title("CLIP Channel Sensitivity")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.annotate(
        f"Cohen's $d$ = {effect_sizes['clip_d']:.2f}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )

    # ── Right: CLAP ──
    ax = axes[1]
    clap_base = effect_sizes["_clap_base"]
    clap_mis = effect_sizes["_clap_mis"]
    bins_clap = np.linspace(
        min(clap_base.min(), clap_mis.min()) - 0.02,
        max(clap_base.max(), clap_mis.max()) + 0.02,
        25,
    )
    ax.hist(clap_base, bins=bins_clap, alpha=0.6, color=COL_BASELINE,
            label="Baseline (matched)", edgecolor="white", linewidth=0.5)
    ax.hist(clap_mis, bins=bins_clap, alpha=0.6, color=COL_CLAP,
            label="Wrong audio (mismatched)", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("CLAP text-audio similarity ($s_{t,a}$)")
    ax.set_ylabel("Count (prompts)")
    ax.set_title("CLAP Channel Sensitivity")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.annotate(
        f"Cohen's $d$ = {effect_sizes['clap_d']:.2f}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )

    plt.tight_layout()
    out_path = FIG_DIR / "fig_clap_distributions.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_clap_roc(domain_auc: Dict[str, Any]) -> Path:
    """Plot ROC curves for overall CLIP vs CLAP discrimination.

    Uses the overall (all-domain) ROC curves, annotated with AUC in legend.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))

    overall = domain_auc["_overall"]

    # CLIP ROC
    clip_fpr = overall["_clip_fpr"]
    clip_tpr = overall["_clip_tpr"]
    clip_auc = overall["clip_auc"]
    ax.plot(clip_fpr, clip_tpr, color=COL_CLIP, linewidth=2.0,
            label=f"CLIP (AUC = {clip_auc:.3f})")

    # CLAP ROC
    clap_fpr = overall["_clap_fpr"]
    clap_tpr = overall["_clap_tpr"]
    clap_auc = overall["clap_auc"]
    ax.plot(clap_fpr, clap_tpr, color=COL_CLAP, linewidth=2.0,
            label=f"CLAP (AUC = {clap_auc:.3f})")

    # Chance line
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0,
            label="Chance", alpha=0.6)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Matched vs Mismatched Discrimination")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path = FIG_DIR / "fig_clap_roc.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_clap_domains(domain_auc: Dict[str, Any]) -> Path:
    """Plot grouped bar chart of per-domain AUC for CLIP vs CLAP.

    Domains on x-axis, AUC on y-axis, two bars per domain.

    Returns:
        Path to saved figure.
    """
    # Filter to actual domains (skip _overall and entries with None AUC)
    domains = [
        d for d in sorted(domain_auc.keys())
        if not d.startswith("_")
        and domain_auc[d].get("clip_auc") is not None
        and domain_auc[d].get("clap_auc") is not None
    ]

    clip_aucs = [domain_auc[d]["clip_auc"] for d in domains]
    clap_aucs = [domain_auc[d]["clap_auc"] for d in domains]
    domain_labels = [d.capitalize() for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_clip = ax.bar(x - width / 2, clip_aucs, width, label="CLIP (text-image)",
                       color=COL_CLIP, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_clap = ax.bar(x + width / 2, clap_aucs, width, label="CLAP (text-audio)",
                       color=COL_CLAP, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar in bars_clip:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars_clap:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Domain")
    ax.set_ylabel("AUC (matched vs mismatched)")
    ax.set_title("Per-Domain Discrimination: CLIP vs CLAP")
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = FIG_DIR / "fig_clap_domains.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_clap_quality(quality_corr: Dict[str, Any]) -> Path:
    """Plot scatter plots of audio quality vs CLAP scores.

    Two side-by-side panels:
      Left:  RMS dB vs st_a (with Spearman rho annotation)
      Right: Spectral flatness vs st_a (with Spearman rho annotation)

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    rms_arr = quality_corr.get("_rms")
    flat_arr = quality_corr.get("_flatness")
    sta_arr = quality_corr.get("_st_a")

    if rms_arr is None or len(rms_arr) < 3:
        # Placeholder if no data
        for ax in axes:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color="gray")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        plt.tight_layout()
        out_path = FIG_DIR / "fig_clap_quality.pdf"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path} (placeholder — no data)")
        return out_path

    # ── Left: RMS dB vs st_a ──
    ax = axes[0]
    ax.scatter(rms_arr, sta_arr, alpha=0.7, s=50, color=COL_CLAP, edgecolor="white",
               linewidth=0.5)

    # Trend line
    if len(rms_arr) >= 3:
        z = np.polyfit(rms_arr, sta_arr, 1)
        x_line = np.linspace(rms_arr.min(), rms_arr.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="gray", linestyle="--",
                linewidth=1.0, alpha=0.7)

    rms_rho = quality_corr["rms_rho"]
    rms_p = quality_corr["rms_p"]
    sig_str = f"p = {rms_p:.4f}" if rms_p >= 0.0001 else f"p < 0.0001"
    ax.annotate(
        f"Spearman $\\rho$ = {rms_rho:.3f}\n{sig_str}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=10, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )
    ax.set_xlabel("RMS Energy (dB)")
    ax.set_ylabel("CLAP text-audio similarity ($s_{t,a}$)")
    ax.set_title("Audio Loudness vs CLAP Score")

    # ── Right: Spectral flatness vs st_a ──
    ax = axes[1]
    ax.scatter(flat_arr, sta_arr, alpha=0.7, s=50, color=COL_CLAP, edgecolor="white",
               linewidth=0.5)

    # Trend line
    if len(flat_arr) >= 3:
        z = np.polyfit(flat_arr, sta_arr, 1)
        x_line = np.linspace(flat_arr.min(), flat_arr.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="gray", linestyle="--",
                linewidth=1.0, alpha=0.7)

    flat_rho = quality_corr["flatness_rho"]
    flat_p = quality_corr["flatness_p"]
    sig_str = f"p = {flat_p:.4f}" if flat_p >= 0.0001 else f"p < 0.0001"
    ax.annotate(
        f"Spearman $\\rho$ = {flat_rho:.3f}\n{sig_str}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=10, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )
    ax.set_xlabel("Spectral Flatness (mean)")
    ax.set_ylabel("CLAP text-audio similarity ($s_{t,a}$)")
    ax.set_title("Audio Spectral Flatness vs CLAP Score")

    plt.tight_layout()
    out_path = FIG_DIR / "fig_clap_quality.pdf"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════
# Serialisation Helpers
# ══════════════════════════════════════════════════════════════

def _make_serialisable(obj: Any) -> Any:
    """Recursively strip numpy arrays and non-JSON types for output."""
    if isinstance(obj, dict):
        return {
            k: _make_serialisable(v)
            for k, v in obj.items()
            if not k.startswith("_")
        }
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CLAP Sensitivity vs Generator Quality Analysis")
    print("=" * 70)

    # ── 1. Load data ──
    print("\n--- Loading RQ1 Results ---")
    results = load_rq1_results()

    print("\n--- Loading RQ3 Samples ---")
    rq3_samples = load_rq3_samples()

    # ── 2. Channel effect sizes ──
    print("\n--- Computing Channel Effect Sizes ---")
    effect_sizes = compute_channel_effect_sizes(results)
    print(f"  CLIP Cohen's d:  {effect_sizes['clip_d']:.4f}  "
          f"(KS stat={effect_sizes['clip_ks_stat']:.3f}, p={effect_sizes['clip_ks_p']:.4f})")
    print(f"  CLAP Cohen's d:  {effect_sizes['clap_d']:.4f}  "
          f"(KS stat={effect_sizes['clap_ks_stat']:.3f}, p={effect_sizes['clap_ks_p']:.4f})")

    # ── 3. Domain AUC ──
    print("\n--- Computing Per-Domain AUC ---")
    domain_auc = compute_domain_auc(results)
    for dom in sorted(domain_auc.keys()):
        if dom.startswith("_"):
            continue
        clip_a = domain_auc[dom]["clip_auc"]
        clap_a = domain_auc[dom]["clap_auc"]
        clip_str = f"{clip_a:.3f}" if clip_a is not None else "N/A"
        clap_str = f"{clap_a:.3f}" if clap_a is not None else "N/A"
        print(f"  {dom:>8s}:  CLIP AUC = {clip_str},  CLAP AUC = {clap_str}")
    overall = domain_auc["_overall"]
    print(f"  {'Overall':>8s}:  CLIP AUC = {overall['clip_auc']:.3f},  "
          f"CLAP AUC = {overall['clap_auc']:.3f}")

    # ── 4. Audio quality correlation ──
    print("\n--- Computing Audio Quality Correlation ---")
    quality_corr = compute_quality_correlation(rq3_samples)
    if quality_corr["rms_rho"] is not None:
        print(f"  RMS dB vs st_a:          rho = {quality_corr['rms_rho']:.4f}, "
              f"p = {quality_corr['rms_p']:.4f}")
        print(f"  Spectral flat. vs st_a:  rho = {quality_corr['flatness_rho']:.4f}, "
              f"p = {quality_corr['flatness_p']:.4f}")
        print(f"  n = {quality_corr['n']} samples analysed")
    else:
        print("  Skipped (no valid audio samples)")

    # ── 5. Generate figures ──
    print("\n--- Generating Figures ---")
    plot_clap_distributions(effect_sizes)
    plot_clap_roc(domain_auc)
    plot_clap_domains(domain_auc)
    plot_clap_quality(quality_corr)

    # ── 6. Save JSON results ──
    output = {
        "effect_sizes": _make_serialisable(effect_sizes),
        "domain_auc": _make_serialisable(domain_auc),
        "quality_correlation": _make_serialisable(quality_corr),
        "summary": {
            "clip_effect_size_d": effect_sizes["clip_d"],
            "clap_effect_size_d": effect_sizes["clap_d"],
            "sensitivity_ratio": (
                round(abs(effect_sizes["clip_d"]) / max(abs(effect_sizes["clap_d"]), 1e-6), 2)
            ),
            "clip_overall_auc": overall["clip_auc"],
            "clap_overall_auc": overall["clap_auc"],
            "rms_rho": quality_corr.get("rms_rho"),
            "flatness_rho": quality_corr.get("flatness_rho"),
            "n_rq1_runs": len(results),
            "n_rq3_quality_samples": quality_corr["n"],
        },
    }

    out_path = OUTPUT_DIR / "clap_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── 7. Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35s} {'CLIP':>10s} {'CLAP':>10s}")
    print("-" * 55)
    print(f"{'Cohen d (matched vs mismatched)':<35s} "
          f"{effect_sizes['clip_d']:>10.3f} {effect_sizes['clap_d']:>10.3f}")
    print(f"{'KS statistic':<35s} "
          f"{effect_sizes['clip_ks_stat']:>10.3f} {effect_sizes['clap_ks_stat']:>10.3f}")
    print(f"{'KS p-value':<35s} "
          f"{effect_sizes['clip_ks_p']:>10.4f} {effect_sizes['clap_ks_p']:>10.4f}")
    print(f"{'Paired t-test p-value':<35s} "
          f"{effect_sizes['clip_ttest_p']:>10.4f} {effect_sizes['clap_ttest_p']:>10.4f}")
    print(f"{'Overall AUC':<35s} "
          f"{overall['clip_auc']:>10.3f} {overall['clap_auc']:>10.3f}")
    ratio = abs(effect_sizes["clip_d"]) / max(abs(effect_sizes["clap_d"]), 1e-6)
    print(f"\n  CLIP/CLAP sensitivity ratio (|d|): {ratio:.2f}x")

    if abs(effect_sizes["clip_d"]) > abs(effect_sizes["clap_d"]):
        print("  --> CLIP is MORE sensitive to perturbation than CLAP")
    elif abs(effect_sizes["clap_d"]) > abs(effect_sizes["clip_d"]):
        print("  --> CLAP is MORE sensitive to perturbation than CLIP")
    else:
        print("  --> CLIP and CLAP show similar sensitivity")

    print("\nDone.")


if __name__ == "__main__":
    main()
