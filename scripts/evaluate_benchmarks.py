#!/usr/bin/env python3
"""
Public Benchmark Evaluation.

Evaluates multimodal coherence methods on AudioCaps and VGGSound benchmarks.
Creates matched (text+correct audio) and mismatched (text+random audio) pairs,
then measures discrimination ability via AUC.

Methods evaluated:
1. CLAP cosine similarity (text-audio baseline)
2. CLIPScore (text-image proxy via text-text in CLIP space)
3. Raw cosine similarity
4. Gramian coherence (our method)
5. cMSCI full pipeline (if calibration available)

Usage:
    python scripts/evaluate_benchmarks.py --audiocaps --max-samples 50
    python scripts/evaluate_benchmarks.py --vggsound --max-samples 100
    python scripts/evaluate_benchmarks.py --all --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coherence.gram_volume import gram_volume_2d, normalized_gram_coherence
from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity

logger = logging.getLogger(__name__)

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"
RUNS_DIR = PROJECT_ROOT / "runs" / "benchmarks"
CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "cmsci_calibration.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_benchmark_manifest(dataset: str) -> List[Dict[str, Any]]:
    """Load manifest.json for a benchmark dataset.

    Args:
        dataset: One of "audiocaps" or "vggsound".

    Returns:
        List of entry dicts, each with at least "caption" and "audio_path".

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
    manifest_path = BENCHMARK_DIR / dataset / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run 'python scripts/download_benchmarks.py --{dataset}' first."
        )

    with open(manifest_path) as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        raise ValueError(f"Manifest at {manifest_path} contains no entries.")

    # Filter to entries whose audio files actually exist on disk
    valid = []
    missing = 0
    for e in entries:
        audio_path = e.get("audio_path", "")
        if audio_path and Path(audio_path).exists():
            valid.append(e)
        else:
            missing += 1

    if missing > 0:
        logger.warning(
            "Skipped %d entries with missing audio files in %s manifest",
            missing, dataset,
        )

    return valid


# ---------------------------------------------------------------------------
# Pair creation
# ---------------------------------------------------------------------------


def create_matched_mismatched_pairs(
    entries: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Tuple[str, str, int]]:
    """Create matched and mismatched (caption, audio_path, label) pairs.

    For each entry:
      - Matched pair: (caption, own audio_path, label=1)
      - Mismatched pair: (caption, some other audio_path, label=0)

    The mismatched assignment is a derangement (no entry paired with itself).

    Args:
        entries: List of manifest entries with "caption" and "audio_path".
        seed: Random seed for reproducibility.

    Returns:
        List of (caption, audio_path, label) tuples.  Length = 2 * len(entries).
    """
    rng = np.random.RandomState(seed)
    n = len(entries)
    if n < 2:
        raise ValueError("Need at least 2 entries to create mismatched pairs.")

    captions = [e["caption"] for e in entries]
    audio_paths = [e["audio_path"] for e in entries]

    # Build a derangement: permute indices such that perm[i] != i for all i
    indices = np.arange(n)
    perm = indices.copy()
    for _ in range(100):  # retry limit (derangement is almost always found)
        rng.shuffle(perm)
        fixed_points = np.where(perm == indices)[0]
        if len(fixed_points) == 0:
            break
        # Fix collisions: swap each fixed point with its neighbor
        for fp in fixed_points:
            swap_target = (fp + 1) % n
            perm[fp], perm[swap_target] = perm[swap_target], perm[fp]
        # Re-check
        if np.all(perm != indices):
            break

    # If derangement still has fixed points (extremely unlikely), force-swap
    fixed = np.where(perm == indices)[0]
    for fp in fixed:
        swap_target = (fp + 1) % n
        perm[fp], perm[swap_target] = perm[swap_target], perm[fp]

    pairs: List[Tuple[str, str, int]] = []
    for i in range(n):
        # Matched
        pairs.append((captions[i], audio_paths[i], 1))
        # Mismatched
        pairs.append((captions[i], audio_paths[perm[i]], 0))

    return pairs


# ---------------------------------------------------------------------------
# Evaluation methods
# ---------------------------------------------------------------------------


def evaluate_clap_cosine(
    pairs: List[Tuple[str, str, int]],
    embedder: AlignedEmbedder,
) -> List[float]:
    """CLAP cosine similarity between text_for_audio and audio embeddings.

    This is the natural text-audio baseline since CLAP encodes both modalities
    into the same 512-d space.

    Args:
        pairs: List of (caption, audio_path, label).
        embedder: AlignedEmbedder instance.

    Returns:
        List of cosine similarity scores (same order as pairs).
    """
    scores: List[float] = []
    for caption, audio_path, _ in pairs:
        try:
            emb_text = embedder.embed_text_for_audio(caption)
            emb_audio = embedder.embed_audio(audio_path)
            score = float(cosine_similarity(emb_text, emb_audio))
        except Exception as e:
            logger.warning("CLAP cosine failed for '%s': %s", audio_path, e)
            score = 0.0
        scores.append(score)
    return scores


def evaluate_clip_text_similarity(
    pairs: List[Tuple[str, str, int]],
    embedder: AlignedEmbedder,
) -> List[float]:
    """CLIP text-only self-similarity (proxy baseline).

    Embeds the caption via CLIP text encoder. Since there is no image in these
    benchmarks, this measures how similar the caption's CLIP text embedding is
    to itself -- a constant per caption. Included for completeness to show that
    CLIP text-only carries no discrimination signal for audio matching.

    For a slightly more meaningful signal, we compute the cosine between the
    CLIP text embedding and the CLAP text embedding of the same caption. This
    measures cross-space text consistency.

    Args:
        pairs: List of (caption, audio_path, label).
        embedder: AlignedEmbedder instance.

    Returns:
        List of cross-space text similarity scores.
    """
    scores: List[float] = []
    for caption, _, _ in pairs:
        try:
            emb_clip = embedder.embed_text(caption)
            emb_clap = embedder.embed_text_for_audio(caption)
            # Cross-space cosine (CLIP text vs CLAP text -- different spaces,
            # so this is not geometrically meaningful but serves as a constant
            # control baseline).
            score = float(cosine_similarity(emb_clip, emb_clap))
        except Exception as e:
            logger.warning("CLIP text similarity failed for caption: %s", e)
            score = 0.0
        scores.append(score)
    return scores


def evaluate_gram_coherence(
    pairs: List[Tuple[str, str, int]],
    embedder: AlignedEmbedder,
) -> List[float]:
    """Gramian volume coherence between CLAP text and audio embeddings.

    Coherence = 1 - gram_volume_2d(text_clap, audio).
    Higher coherence means more aligned.

    Args:
        pairs: List of (caption, audio_path, label).
        embedder: AlignedEmbedder instance.

    Returns:
        List of Gramian coherence scores.
    """
    scores: List[float] = []
    for caption, audio_path, _ in pairs:
        try:
            emb_text = embedder.embed_text_for_audio(caption)
            emb_audio = embedder.embed_audio(audio_path)
            volume = gram_volume_2d(emb_text, emb_audio)
            coherence = normalized_gram_coherence(volume)
            score = float(coherence)
        except Exception as e:
            logger.warning("Gram coherence failed for '%s': %s", audio_path, e)
            score = 0.0
        scores.append(score)
    return scores


def evaluate_cmsci(
    pairs: List[Tuple[str, str, int]],
    calibration_path: Optional[str] = None,
) -> Optional[List[float]]:
    """Full cMSCI pipeline evaluation (text + audio only, no image).

    Uses CalibratedCoherenceEngine if calibration data is available.
    Returns None if the engine cannot be initialized (missing calibration).

    Args:
        pairs: List of (caption, audio_path, label).
        calibration_path: Path to calibration JSON. Defaults to project artifact.

    Returns:
        List of cMSCI scores, or None if calibration is unavailable.
    """
    cal_path = calibration_path or str(CALIBRATION_PATH)
    if not Path(cal_path).exists():
        logger.info(
            "Calibration file not found at %s; skipping cMSCI evaluation.", cal_path,
        )
        return None

    try:
        from src.coherence.cmsci_engine import CalibratedCoherenceEngine
        engine = CalibratedCoherenceEngine(
            calibration_path=cal_path,
            negative_bank_enabled=False,  # No negative bank for benchmarks
        )
    except Exception as e:
        logger.warning("Could not initialize CalibratedCoherenceEngine: %s", e)
        return None

    scores: List[float] = []
    for caption, audio_path, _ in pairs:
        try:
            result = engine.evaluate(
                text=caption,
                image_path=None,
                audio_path=audio_path,
                n_mc_samples=0,
            )
            score = result.get("cmsci") or result.get("msci") or 0.0
            scores.append(float(score))
        except Exception as e:
            logger.warning("cMSCI failed for '%s': %s", audio_path, e)
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    labels: List[int],
    scores: List[float],
    method_name: str,
) -> Dict[str, Any]:
    """Compute discrimination metrics: AUC, best-threshold accuracy, mean scores.

    Args:
        labels: Ground-truth binary labels (1=matched, 0=mismatched).
        scores: Predicted scores (higher = more coherent).
        method_name: Human-readable name of the method.

    Returns:
        Dict with auc, accuracy, threshold, mean_matched, mean_mismatched.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    labels_arr = np.array(labels, dtype=int)
    scores_arr = np.array(scores, dtype=float)

    # Handle degenerate cases
    if len(np.unique(labels_arr)) < 2:
        logger.warning("Only one class present for %s; AUC undefined.", method_name)
        return {
            "method": method_name,
            "auc": float("nan"),
            "accuracy": float("nan"),
            "threshold": float("nan"),
            "mean_matched": float(np.mean(scores_arr[labels_arr == 1])) if np.any(labels_arr == 1) else float("nan"),
            "mean_mismatched": float(np.mean(scores_arr[labels_arr == 0])) if np.any(labels_arr == 0) else float("nan"),
            "n_pairs": len(labels_arr),
        }

    if np.std(scores_arr) < 1e-12:
        logger.warning("Constant scores for %s; AUC = 0.5.", method_name)
        return {
            "method": method_name,
            "auc": 0.5,
            "accuracy": 0.5,
            "threshold": float(scores_arr[0]),
            "mean_matched": float(np.mean(scores_arr[labels_arr == 1])),
            "mean_mismatched": float(np.mean(scores_arr[labels_arr == 0])),
            "n_pairs": len(labels_arr),
        }

    auc = float(roc_auc_score(labels_arr, scores_arr))

    # Best threshold via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    predictions = (scores_arr >= best_threshold).astype(int)
    accuracy = float(np.mean(predictions == labels_arr))

    # F1 at the Youden-optimal threshold
    from sklearn.metrics import f1_score
    f1 = float(f1_score(labels_arr, predictions, zero_division=0))

    # FPR at 95% TPR (smallest FPR achieving TPR >= 0.95)
    tpr_mask = tpr >= 0.95
    fpr_at_95tpr = float(np.min(fpr[tpr_mask])) if np.any(tpr_mask) else float("nan")

    mean_matched = float(np.mean(scores_arr[labels_arr == 1]))
    mean_mismatched = float(np.mean(scores_arr[labels_arr == 0]))

    # Cohen's d (matched vs mismatched), pooled SD
    s_m = scores_arr[labels_arr == 1]
    s_u = scores_arr[labels_arr == 0]
    pooled_sd = np.sqrt((s_m.var(ddof=1) + s_u.var(ddof=1)) / 2.0)
    cohens_d = float((mean_matched - mean_mismatched) / pooled_sd) if pooled_sd > 0 else float("nan")

    return {
        "method": method_name,
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "fpr_at_95_tpr": round(fpr_at_95tpr, 4) if not np.isnan(fpr_at_95tpr) else None,
        "cohens_d": round(cohens_d, 4) if not np.isnan(cohens_d) else None,
        "threshold": round(best_threshold, 4),
        "mean_matched": round(mean_matched, 4),
        "mean_mismatched": round(mean_mismatched, 4),
        "n_pairs": len(labels_arr),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_benchmark_roc(
    results: Dict[str, Dict[str, Any]],
    all_labels: List[int],
    all_scores: Dict[str, List[float]],
    output_dir: Path,
    dataset_name: str = "benchmark",
) -> None:
    """Plot ROC curves for all evaluated methods.

    Args:
        results: Dict mapping method_name -> metrics dict.
        all_labels: Ground-truth labels (shared across methods).
        all_scores: Dict mapping method_name -> list of scores.
        output_dir: Directory to save the figure.
        dataset_name: Name used in the figure title and filename.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        logger.warning("matplotlib or sklearn not available; skipping ROC plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    labels_arr = np.array(all_labels, dtype=int)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (method_name, scores) in enumerate(all_scores.items()):
        scores_arr = np.array(scores, dtype=float)
        if np.std(scores_arr) < 1e-12:
            continue  # Skip constant-score methods in ROC plot
        fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
        auc_val = results[method_name]["auc"]
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{method_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC: Matched vs Mismatched Discrimination ({dataset_name})", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"fig_benchmark_roc_{dataset_name}.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ROC figure saved: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate_dataset(
    dataset: str,
    max_samples: int,
    embedder: AlignedEmbedder,
) -> Dict[str, Any]:
    """Run full evaluation on a single benchmark dataset.

    Args:
        dataset: "audiocaps" or "vggsound".
        max_samples: Maximum number of manifest entries to use.
        embedder: Shared AlignedEmbedder instance.

    Returns:
        Dict with per-method metrics, raw scores, and labels.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {dataset}")
    print(f"{'=' * 60}")

    # Load data
    entries = load_benchmark_manifest(dataset)
    if len(entries) > max_samples:
        entries = entries[:max_samples]
    print(f"  Loaded {len(entries)} entries (max_samples={max_samples})")

    # Create pairs
    pairs = create_matched_mismatched_pairs(entries, seed=42)
    labels = [label for _, _, label in pairs]
    n_matched = sum(labels)
    n_mismatched = len(labels) - n_matched
    print(f"  Created {len(pairs)} pairs ({n_matched} matched, {n_mismatched} mismatched)")

    # Evaluate each method
    method_scores: Dict[str, List[float]] = {}
    method_results: Dict[str, Dict[str, Any]] = {}

    # 1. CLAP cosine
    print("  [1/4] CLAP cosine similarity...")
    t0 = time.time()
    clap_scores = evaluate_clap_cosine(pairs, embedder)
    elapsed = time.time() - t0
    method_scores["CLAP_cosine"] = clap_scores
    method_results["CLAP_cosine"] = compute_metrics(labels, clap_scores, "CLAP_cosine")
    print(f"         AUC={method_results['CLAP_cosine']['auc']:.4f}  ({elapsed:.1f}s)")

    # 2. CLIP text similarity (control baseline)
    print("  [2/4] CLIP text cross-space similarity (control)...")
    t0 = time.time()
    clip_scores = evaluate_clip_text_similarity(pairs, embedder)
    elapsed = time.time() - t0
    method_scores["CLIP_text_xspace"] = clip_scores
    method_results["CLIP_text_xspace"] = compute_metrics(labels, clip_scores, "CLIP_text_xspace")
    print(f"         AUC={method_results['CLIP_text_xspace']['auc']:.4f}  ({elapsed:.1f}s)")

    # 3. Gramian coherence
    print("  [3/4] Gramian coherence...")
    t0 = time.time()
    gram_scores = evaluate_gram_coherence(pairs, embedder)
    elapsed = time.time() - t0
    method_scores["Gram_coherence"] = gram_scores
    method_results["Gram_coherence"] = compute_metrics(labels, gram_scores, "Gram_coherence")
    print(f"         AUC={method_results['Gram_coherence']['auc']:.4f}  ({elapsed:.1f}s)")

    # 4. cMSCI (conditional)
    print("  [4/4] cMSCI full pipeline...")
    t0 = time.time()
    cmsci_scores = evaluate_cmsci(pairs)
    elapsed = time.time() - t0
    if cmsci_scores is not None:
        method_scores["cMSCI"] = cmsci_scores
        method_results["cMSCI"] = compute_metrics(labels, cmsci_scores, "cMSCI")
        print(f"         AUC={method_results['cMSCI']['auc']:.4f}  ({elapsed:.1f}s)")
    else:
        print(f"         Skipped (no calibration)  ({elapsed:.1f}s)")

    # Print summary table
    print(f"\n  {'Method':<22} {'AUC':>6} {'Acc':>6} {'Matched':>8} {'Mismatched':>10}")
    print(f"  {'-' * 56}")
    for name, metrics in method_results.items():
        print(
            f"  {name:<22} {metrics['auc']:>6.4f} {metrics['accuracy']:>6.4f} "
            f"{metrics['mean_matched']:>8.4f} {metrics['mean_mismatched']:>10.4f}"
        )

    # Plot ROC
    plot_benchmark_roc(
        method_results, labels, method_scores,
        output_dir=RUNS_DIR, dataset_name=dataset,
    )

    return {
        "dataset": dataset,
        "n_entries": len(entries),
        "n_pairs": len(pairs),
        "methods": method_results,
        "scores": {k: [round(s, 6) for s in v] for k, v in method_scores.items()},
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate coherence methods on public benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate_benchmarks.py --audiocaps --max-samples 50\n"
            "  python scripts/evaluate_benchmarks.py --vggsound --max-samples 100\n"
            "  python scripts/evaluate_benchmarks.py --all --max-samples 5000\n"
        ),
    )
    parser.add_argument("--audiocaps", action="store_true", help="Evaluate on AudioCaps")
    parser.add_argument("--vggsound", action="store_true", help="Evaluate on VGGSound")
    parser.add_argument("--all", action="store_true", help="Evaluate on all benchmarks")
    parser.add_argument(
        "--max-samples", type=int, default=100,
        help="Max manifest entries per dataset (default: 100)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: runs/benchmarks/benchmark_evaluation.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    datasets: List[str] = []
    if args.audiocaps or args.all:
        datasets.append("audiocaps")
    if args.vggsound or args.all:
        datasets.append("vggsound")

    if not datasets:
        print("No dataset specified. Use --audiocaps, --vggsound, or --all")
        parser.print_help()
        return

    print("=" * 70)
    print("Public Benchmark Evaluation")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Max samples per dataset: {args.max_samples}")
    print("=" * 70)

    # Initialize shared embedder once (loads CLIP + CLAP models)
    print("\nLoading embedding models...")
    t0 = time.time()
    embedder = AlignedEmbedder(target_dim=512)
    print(f"  Models loaded in {time.time() - t0:.1f}s")

    # Evaluate each dataset
    all_results: Dict[str, Any] = {}
    for dataset in datasets:
        try:
            result = evaluate_dataset(dataset, args.max_samples, embedder)
            all_results[dataset] = result
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            continue
        except Exception as e:
            logger.error("Evaluation failed for %s: %s", dataset, e, exc_info=True)
            continue

    # Save combined results
    if all_results:
        output_path = Path(args.output) if args.output else RUNS_DIR / "benchmark_evaluation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip raw scores/labels from the saved JSON to keep it manageable
        save_data = {}
        for ds_name, ds_result in all_results.items():
            save_data[ds_name] = {
                "dataset": ds_result["dataset"],
                "n_entries": ds_result["n_entries"],
                "n_pairs": ds_result["n_pairs"],
                "methods": ds_result["methods"],
            }

        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
