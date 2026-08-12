#!/usr/bin/env python3
"""
USE CASE 1: Offline Batch Quality Gate using cMSCI v1 on AudioCaps.

Simulates a CI/CD quality gate where a production pipeline generates
multimodal bundles and we must automatically flag incoherent ones.

Creates matched (correct caption + correct audio) and mismatched
(wrong caption + correct audio) pairs, then evaluates discrimination
ability of:
  1. cMSCI v1 (full Variant F pipeline)
  2. CLAP cosine (raw MSCI-style baseline)
  3. Gramian coherence (geometric baseline)
  4. CLIPScore proxy (text-only control — expected to fail)
  5. Median threshold baseline

Metrics computed per method:
  - AUC (ROC), Youden's J best threshold, Precision/Recall/F1
  - Precision@90% recall, TPR at bottom-10% flag, FPR@95% TPR

Saves full results to runs/use_case_experiments/usecase1_batch_qa.json.

Usage:
    python scripts/usecase1_batch_qa.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    CMSCI_CALIBRATION_PATH,
    EXMCR_WEIGHTS_PATH,
    BRIDGE_WEIGHTS_PATH,
    PROB_CLIP_ADAPTER_PATH,
    PROB_CLAP_ADAPTER_PATH,
)
from src.coherence.gram_volume import gram_volume_2d, normalized_gram_coherence
from src.embeddings.similarity import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("usecase1")

MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmarks" / "audiocaps" / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "use_case_experiments"
OUTPUT_PATH = OUTPUT_DIR / "usecase1_batch_qa.json"

SEED = 42


# ──────────────────────────────────────────────────────────────────────────
# Data Loading + Pair Creation
# ──────────────────────────────────────────────────────────────────────────


def load_audiocaps_entries() -> List[Dict[str, Any]]:
    """Load AudioCaps manifest and filter to entries with existing audio."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    entries = data.get("entries", [])
    valid = [e for e in entries if Path(e.get("audio_path", "")).exists()]
    logger.info("Loaded %d AudioCaps entries (%d valid with audio on disk)", len(entries), len(valid))
    return valid


def create_pairs(entries: List[Dict[str, Any]], seed: int = SEED) -> Tuple[List[Tuple[str, str, int]], np.ndarray]:
    """Create matched (label=1) and mismatched (label=0) pairs via derangement.

    Returns (pairs, derangement_indices) where pairs is a list of
    (caption, audio_path, label) tuples.
    """
    rng = np.random.RandomState(seed)
    n = len(entries)
    captions = [e["caption"] for e in entries]
    audio_paths = [e["audio_path"] for e in entries]

    # Build derangement (no element maps to itself)
    indices = np.arange(n)
    perm = indices.copy()
    for _ in range(200):
        rng.shuffle(perm)
        if np.all(perm != indices):
            break
    # Fix any remaining fixed points
    fixed = np.where(perm == indices)[0]
    for fp in fixed:
        swap = (fp + 1) % n
        perm[fp], perm[swap] = perm[swap], perm[fp]
    assert np.all(perm != indices), "Derangement failed"

    pairs: List[Tuple[str, str, int]] = []
    for i in range(n):
        pairs.append((captions[i], audio_paths[i], 1))          # matched
        pairs.append((captions[i], audio_paths[perm[i]], 0))     # mismatched

    logger.info("Created %d pairs (%d matched + %d mismatched)", len(pairs), n, n)
    return pairs, perm


# ──────────────────────────────────────────────────────────────────────────
# Scoring Methods
# ──────────────────────────────────────────────────────────────────────────


def score_cmsci_v1(pairs: List[Tuple[str, str, int]]) -> Tuple[List[float], float]:
    """Score all pairs using full cMSCI v1 (Variant F) engine.

    For text-audio-only samples, cMSCI degrades gracefully: it uses
    whatever channels are available (text-audio Gramian, CLAP z-norm,
    CLAP contrastive margin, CLAP probabilistic adapter).

    Returns (scores, elapsed_seconds).
    """
    from src.coherence.cmsci_engine import CalibratedCoherenceEngine

    engine = CalibratedCoherenceEngine(
        calibration_path=str(CMSCI_CALIBRATION_PATH),
        exmcr_weights_path=str(EXMCR_WEIGHTS_PATH),
        bridge_path=str(BRIDGE_WEIGHTS_PATH),
        prob_clip_adapter_path=str(PROB_CLIP_ADAPTER_PATH),
        prob_clap_adapter_path=str(PROB_CLAP_ADAPTER_PATH),
    )

    scores: List[float] = []
    t0 = time.time()

    for idx, (caption, audio_path, _) in enumerate(pairs):
        try:
            result = engine.evaluate(
                text=caption,
                image_path=None,
                audio_path=audio_path,
                n_mc_samples=50,   # reduced for speed; 50 is sufficient for CI
            )
            score = result.get("cmsci")
            if score is None:
                score = result.get("msci", 0.0)
            scores.append(float(score))
        except Exception as e:
            logger.warning("cMSCI failed for pair %d: %s", idx, e)
            scores.append(0.0)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  cMSCI: %d/%d pairs (%.1fs)", idx + 1, len(pairs), elapsed)

    elapsed = time.time() - t0
    logger.info("cMSCI v1 scoring complete: %d pairs in %.1fs (%.2f pairs/s)",
                len(pairs), elapsed, len(pairs) / elapsed)
    return scores, elapsed


def score_clap_cosine(pairs: List[Tuple[str, str, int]], embedder) -> Tuple[List[float], float]:
    """Raw CLAP cosine similarity (MSCI-equivalent text-audio baseline)."""
    scores: List[float] = []
    t0 = time.time()
    for caption, audio_path, _ in pairs:
        try:
            emb_t = embedder.embed_text_for_audio(caption)
            emb_a = embedder.embed_audio(audio_path)
            scores.append(float(cosine_similarity(emb_t, emb_a)))
        except Exception as e:
            logger.warning("CLAP cosine failed: %s", e)
            scores.append(0.0)
    elapsed = time.time() - t0
    return scores, elapsed


def score_gram_coherence(pairs: List[Tuple[str, str, int]], embedder) -> Tuple[List[float], float]:
    """Gramian volume coherence (1 - volume) between CLAP text and audio."""
    scores: List[float] = []
    t0 = time.time()
    for caption, audio_path, _ in pairs:
        try:
            emb_t = embedder.embed_text_for_audio(caption)
            emb_a = embedder.embed_audio(audio_path)
            volume = gram_volume_2d(emb_t, emb_a)
            scores.append(float(normalized_gram_coherence(volume)))
        except Exception as e:
            logger.warning("Gramian failed: %s", e)
            scores.append(0.0)
    elapsed = time.time() - t0
    return scores, elapsed


def score_clip_text_proxy(pairs: List[Tuple[str, str, int]], embedder) -> Tuple[List[float], float]:
    """CLIPScore proxy: CLIP text embedding norm (no audio signal).

    Since AudioCaps has no images, CLIPScore degrades to a text-only signal.
    This is a control baseline expected to yield AUC near 0.5.
    """
    scores: List[float] = []
    t0 = time.time()
    for caption, audio_path, _ in pairs:
        try:
            emb_clip = embedder.embed_text(caption)
            emb_clap = embedder.embed_text_for_audio(caption)
            # Cross-space cosine (CLIP text vs CLAP text) — geometrically
            # meaningless but provides a constant-ish control score
            scores.append(float(cosine_similarity(emb_clip, emb_clap)))
        except Exception as e:
            logger.warning("CLIP text proxy failed: %s", e)
            scores.append(0.0)
    elapsed = time.time() - t0
    return scores, elapsed


def score_median_threshold(labels: List[int], reference_scores: List[float]) -> List[float]:
    """Median threshold baseline: scores above median => matched.

    Uses reference_scores (CLAP cosine) to compute threshold.
    Returns the same reference_scores (the threshold is applied at evaluation).
    """
    return reference_scores  # Threshold applied in metrics


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────


def compute_full_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    method_name: str,
) -> Dict[str, Any]:
    """Compute comprehensive QA-gate metrics.

    Metrics:
      - AUC (ROC)
      - Best threshold via Youden's J
      - Precision, Recall, F1 at best threshold
      - Precision@90% recall
      - TPR when flagging bottom 10% as incoherent
      - FPR at 95% TPR
      - Mean/std for matched and mismatched
      - Cohen's d (effect size)
    """
    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve,
        precision_score, recall_score, f1_score,
    )

    matched = scores[labels == 1]
    mismatched = scores[labels == 0]

    # Handle constant scores (e.g., CLIPScore control)
    if np.std(scores) < 1e-12:
        return {
            "method": method_name,
            "auc": 0.5,
            "best_threshold": float(scores[0]),
            "precision_at_best": 0.5,
            "recall_at_best": 1.0,
            "f1_at_best": 0.6667,
            "precision_at_90_recall": 0.5,
            "fpr_at_95_tpr": 1.0,
            "tpr_bottom_10pct_flag": 0.0,
            "threshold_10pct": float(scores[0]),
            "n_flagged_bottom10": 0,
            "n_flagged_correct_bottom10": 0,
            "mean_matched": float(np.mean(matched)),
            "std_matched": float(np.std(matched, ddof=1)) if len(matched) > 1 else 0.0,
            "mean_mismatched": float(np.mean(mismatched)),
            "std_mismatched": float(np.std(mismatched, ddof=1)) if len(mismatched) > 1 else 0.0,
            "cohens_d": 0.0,
            "n_pairs": len(labels),
            "n_matched": int(np.sum(labels == 1)),
            "n_mismatched": int(np.sum(labels == 0)),
            "note": "Constant scores — no discriminative signal",
        }

    # AUC
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.5

    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Youden's J best threshold
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    preds_best = (scores >= best_threshold).astype(int)

    prec_best = float(precision_score(labels, preds_best, zero_division=0))
    rec_best = float(recall_score(labels, preds_best, zero_division=0))
    f1_best = float(f1_score(labels, preds_best, zero_division=0))

    # Precision@90% recall
    # precision_recall_curve returns recall in DECREASING order.
    # We want the highest precision achievable while recall >= 0.90.
    prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
    prec_at_90 = None
    for p, r in zip(prec_curve, rec_curve):
        if r >= 0.90:
            if prec_at_90 is None or p > prec_at_90:
                prec_at_90 = float(p)
    if prec_at_90 is None:
        prec_at_90 = float(prec_curve[0])  # fallback to lowest-threshold precision

    # FPR at 95% TPR
    fpr_at_95tpr = None
    for fp, tp in zip(fpr, tpr):
        if tp >= 0.95:
            fpr_at_95tpr = float(fp)
            break
    if fpr_at_95tpr is None:
        fpr_at_95tpr = 1.0

    # TPR when flagging bottom 10% as incoherent
    # Bottom 10% by score => predicted incoherent (label=0)
    threshold_10pct = float(np.percentile(scores, 10))
    flagged = scores <= threshold_10pct
    # Of flagged items, what fraction are truly mismatched?
    n_flagged = int(np.sum(flagged))
    n_flagged_correct = int(np.sum(flagged & (labels == 0)))
    # TPR = fraction of all mismatched that we caught
    n_total_mismatched = int(np.sum(labels == 0))
    tpr_bottom10 = n_flagged_correct / max(n_total_mismatched, 1)

    # Cohen's d
    pooled_std = np.sqrt((np.var(matched, ddof=1) + np.var(mismatched, ddof=1)) / 2)
    cohens_d = float((np.mean(matched) - np.mean(mismatched)) / max(pooled_std, 1e-12))

    return {
        "method": method_name,
        "auc": round(auc, 4),
        "best_threshold": round(best_threshold, 6),
        "precision_at_best": round(prec_best, 4),
        "recall_at_best": round(rec_best, 4),
        "f1_at_best": round(f1_best, 4),
        "precision_at_90_recall": round(prec_at_90, 4),
        "fpr_at_95_tpr": round(fpr_at_95tpr, 4),
        "tpr_bottom_10pct_flag": round(tpr_bottom10, 4),
        "threshold_10pct": round(threshold_10pct, 6),
        "n_flagged_bottom10": n_flagged,
        "n_flagged_correct_bottom10": n_flagged_correct,
        "mean_matched": round(float(np.mean(matched)), 6),
        "std_matched": round(float(np.std(matched, ddof=1)), 6),
        "mean_mismatched": round(float(np.mean(mismatched)), 6),
        "std_mismatched": round(float(np.std(mismatched, ddof=1)), 6),
        "cohens_d": round(cohens_d, 4),
        "n_pairs": len(labels),
        "n_matched": int(np.sum(labels == 1)),
        "n_mismatched": int(np.sum(labels == 0)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("USE CASE 1: Offline Batch Quality Gate — cMSCI v1 on AudioCaps")
    print("=" * 70)
    print()

    # 1. Load data
    entries = load_audiocaps_entries()
    n_samples = len(entries)
    pairs, derangement = create_pairs(entries)
    labels = np.array([label for _, _, label in pairs], dtype=int)

    print(f"\nDataset: AudioCaps (text + audio only, no images)")
    print(f"Samples: {n_samples}")
    print(f"Total pairs: {len(pairs)} ({n_samples} matched + {n_samples} mismatched)")
    print()

    all_results: Dict[str, Any] = {}
    all_scores: Dict[str, List[float]] = {}
    all_times: Dict[str, float] = {}

    # 2. cMSCI v1 (full pipeline)
    print("-" * 50)
    print("[1/4] cMSCI v1 (full Variant F pipeline)...")
    print("-" * 50)
    cmsci_scores, cmsci_time = score_cmsci_v1(pairs)
    all_scores["cMSCI_v1"] = cmsci_scores
    all_times["cMSCI_v1"] = cmsci_time

    # We need an embedder for the baselines — reuse from cMSCI engine implicitly
    # but create a standalone one for baselines to keep timing clean
    print("\nLoading embedder for baselines...")
    from src.embeddings.aligned_embeddings import AlignedEmbedder
    embedder = AlignedEmbedder(target_dim=512)

    # 3. CLAP cosine (MSCI equivalent)
    print("-" * 50)
    print("[2/4] CLAP cosine similarity (MSCI baseline)...")
    print("-" * 50)
    clap_scores, clap_time = score_clap_cosine(pairs, embedder)
    all_scores["CLAP_cosine"] = clap_scores
    all_times["CLAP_cosine"] = clap_time

    # 4. Gramian coherence
    print("-" * 50)
    print("[3/4] Gramian volume coherence...")
    print("-" * 50)
    gram_scores, gram_time = score_gram_coherence(pairs, embedder)
    all_scores["Gram_coherence"] = gram_scores
    all_times["Gram_coherence"] = gram_time

    # 5. CLIPScore proxy (control)
    print("-" * 50)
    print("[4/4] CLIPScore text-only proxy (control)...")
    print("-" * 50)
    clip_scores, clip_time = score_clip_text_proxy(pairs, embedder)
    all_scores["CLIPScore_proxy"] = clip_scores
    all_times["CLIPScore_proxy"] = clip_time

    # 6. Median threshold (uses CLAP cosine scores with median split)
    # Scored identically to CLAP cosine; the difference is in interpretation
    all_scores["Median_threshold"] = clap_scores  # same scores, different threshold strategy

    # ──────────────────────────────────────────────────────────────
    # Compute metrics for each method
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    method_metrics: Dict[str, Dict[str, Any]] = {}
    for method_name, scores_list in all_scores.items():
        scores_arr = np.array(scores_list, dtype=float)
        metrics = compute_full_metrics(labels, scores_arr, method_name)
        if method_name in all_times:
            metrics["elapsed_seconds"] = round(all_times[method_name], 2)
        method_metrics[method_name] = metrics

    # For median threshold: override threshold to be the median
    median_val = float(np.median(all_scores["CLAP_cosine"]))
    method_metrics["Median_threshold"]["threshold_override"] = round(median_val, 6)
    # Recompute precision/recall/f1 at median threshold
    from sklearn.metrics import precision_score, recall_score, f1_score
    median_preds = (np.array(all_scores["Median_threshold"]) >= median_val).astype(int)
    method_metrics["Median_threshold"]["precision_at_median"] = round(
        float(precision_score(labels, median_preds, zero_division=0)), 4)
    method_metrics["Median_threshold"]["recall_at_median"] = round(
        float(recall_score(labels, median_preds, zero_division=0)), 4)
    method_metrics["Median_threshold"]["f1_at_median"] = round(
        float(f1_score(labels, median_preds, zero_division=0)), 4)

    # Print summary table
    print(f"\n{'Method':<22} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'d':>7} {'P@90R':>7} {'FPR@95':>8}")
    print("-" * 82)
    for name, m in method_metrics.items():
        if name == "Median_threshold":
            continue  # Print separately
        print(f"{name:<22} {m['auc']:>7.4f} {m['f1_at_best']:>7.4f} "
              f"{m['precision_at_best']:>7.4f} {m['recall_at_best']:>7.4f} "
              f"{m['cohens_d']:>7.3f} {m['precision_at_90_recall']:>7.4f} "
              f"{m['fpr_at_95_tpr']:>8.4f}")
    print()

    # Practical QA metrics
    print("=" * 70)
    print("PRACTICAL QA GATE METRICS")
    print("=" * 70)
    for name, m in method_metrics.items():
        if name == "Median_threshold":
            continue
        print(f"\n--- {name} ---")
        print(f"  AUC:                    {m['auc']:.4f}")
        print(f"  Best threshold (J):     {m['best_threshold']:.6f}")
        print(f"  F1 at best threshold:   {m['f1_at_best']:.4f}")
        print(f"  Precision at best:      {m['precision_at_best']:.4f}")
        print(f"  Recall at best:         {m['recall_at_best']:.4f}")
        print(f"  Precision@90% recall:   {m['precision_at_90_recall']:.4f}")
        print(f"  FPR@95% TPR:            {m['fpr_at_95_tpr']:.4f}")
        print(f"  Cohen's d:              {m['cohens_d']:.3f}")
        print(f"  Score gap: matched {m['mean_matched']:.4f} +/- {m['std_matched']:.4f}"
              f" vs mismatched {m['mean_mismatched']:.4f} +/- {m['std_mismatched']:.4f}")
        if "elapsed_seconds" in m:
            et = max(m["elapsed_seconds"], 0.001)
            print(f"  Processing time:        {m['elapsed_seconds']:.1f}s ({len(pairs)/et:.1f} pairs/s)")

    # Bottom 10% flagging analysis
    print(f"\n{'='*70}")
    print("BOTTOM-10% FLAGGING ANALYSIS")
    print("(If you flag the lowest 10% of scores as incoherent...)")
    print(f"{'='*70}")
    for name, m in method_metrics.items():
        if name == "Median_threshold":
            continue
        print(f"  {name:<22}: TPR = {m['tpr_bottom_10pct_flag']:.2%}"
              f" ({m['n_flagged_correct_bottom10']}/{m['n_flagged_bottom10']} flagged are truly mismatched)")

    # Median threshold specific
    m_med = method_metrics["Median_threshold"]
    print(f"\n--- Median Threshold Baseline ---")
    print(f"  Median threshold:       {m_med['threshold_override']:.6f}")
    print(f"  Precision at median:    {m_med['precision_at_median']:.4f}")
    print(f"  Recall at median:       {m_med['recall_at_median']:.4f}")
    print(f"  F1 at median:           {m_med['f1_at_median']:.4f}")

    total_time = sum(t for t in all_times.values())
    print(f"\nTotal processing time:    {total_time:.1f}s for {n_samples} samples "
          f"({n_samples * 2} pairs)")

    # ──────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_data = {
        "use_case": "UC1: Offline Batch Quality Gate",
        "dataset": "AudioCaps",
        "n_samples": n_samples,
        "n_pairs": len(pairs),
        "n_matched": int(np.sum(labels == 1)),
        "n_mismatched": int(np.sum(labels == 0)),
        "seed": SEED,
        "total_elapsed_seconds": round(total_time, 2),
        "methods": method_metrics,
        "per_pair_scores": {
            method: [round(s, 6) for s in scores_list]
            for method, scores_list in all_scores.items()
            if method != "Median_threshold"
        },
        "labels": labels.tolist(),
        "derangement_indices": derangement.tolist(),
        "configuration": {
            "cmsci_calibration": str(CMSCI_CALIBRATION_PATH),
            "exmcr_weights": str(EXMCR_WEIGHTS_PATH),
            "bridge_weights": str(BRIDGE_WEIGHTS_PATH),
            "prob_clip_adapter": str(PROB_CLIP_ADAPTER_PATH),
            "prob_clap_adapter": str(PROB_CLAP_ADAPTER_PATH),
            "n_mc_samples": 50,
        },
    }

    # Replace any Infinity/NaN with null for valid JSON
    def _sanitize(obj):
        if isinstance(obj, float):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    save_data = _sanitize(save_data)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("Done!")
    sys.stdout.flush()

    # Also write output log
    log_path = OUTPUT_DIR / "usecase1_output.txt"
    with open(log_path, "w") as flog:
        import io
        # Re-generate compact summary for log
        flog.write("USE CASE 1: Offline Batch Quality Gate Results\n")
        flog.write("=" * 60 + "\n")
        flog.write(f"Dataset: AudioCaps, {n_samples} samples, {len(pairs)} pairs\n\n")
        flog.write(f"{'Method':<22} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'d':>7} {'P@90R':>7} {'FPR@95':>8}\n")
        flog.write("-" * 82 + "\n")
        for name, m in method_metrics.items():
            if name == "Median_threshold":
                continue
            flog.write(f"{name:<22} {m['auc']:>7.4f} {m['f1_at_best']:>7.4f} "
                       f"{m['precision_at_best']:>7.4f} {m['recall_at_best']:>7.4f} "
                       f"{m['cohens_d']:>7.3f} {m['precision_at_90_recall']:>7.4f} "
                       f"{m['fpr_at_95_tpr']:>8.4f}\n")
        flog.write(f"\nMedian threshold: P={m_med['precision_at_median']:.4f} R={m_med['recall_at_median']:.4f} F1={m_med['f1_at_median']:.4f}\n")
        flog.write(f"\nTotal time: {total_time:.1f}s\n")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
