#!/usr/bin/env python3
"""
Gemini Embedding Task-Type Experiment.

Tests different task_type values to find which embedding geometry produces
the best coherence signal for cMSCI v2.

Task types:
    SEMANTIC_SIMILARITY  — current default
    RETRIEVAL_DOCUMENT   — optimized for document retrieval
    RETRIEVAL_QUERY      — optimized for query encoding
    CLASSIFICATION       — optimized for classification tasks
    CLUSTERING           — optimized for clustering

For each task type, re-embeds all 30 RQ3 samples and computes:
    - Raw cosine correlation with human ratings
    - Gramian 2D (TI, TA, IA) correlations
    - Gramian 3D correlation
    - Channel standard deviations (to identify discriminative power)

Estimate: ~450 API calls total (5 task types x ~90 embeddings).

Usage:
    python scripts/experiment_task_types.py
    python scripts/experiment_task_types.py --task-types SEMANTIC_SIMILARITY CLUSTERING
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coherence.gram_volume import (
    gram_volume_2d,
    gram_volume_3d,
    normalized_gram_coherence,
)
from src.config.settings import GEMINI_CACHE_DIR

SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
SESSION_DIR = PROJECT_ROOT / "runs" / "rq3" / "sessions"

ALL_TASK_TYPES = [
    "SEMANTIC_SIMILARITY",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "CLASSIFICATION",
    "CLUSTERING",
]


def load_human_scores() -> dict:
    """Load human evaluation scores."""
    from src.evaluation.human_eval_schema import EvaluationSession
    from src.evaluation.human_eval_analyzer import aggregate_multi_rater_sessions

    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json")):
        try:
            session = EvaluationSession.load(p)
            if session.progress >= 80.0:
                sessions.append(session)
        except Exception:
            pass

    return aggregate_multi_rater_sessions(sessions)


def evaluate_task_type(
    task_type: str,
    samples: list,
    human_scores: dict,
    cache_dir: Path,
) -> dict:
    """Evaluate a single task type by re-embedding all samples."""
    from src.embeddings.gemini_embedder import GeminiEmbedder

    # Use a task-type-specific cache directory to avoid collision
    tt_cache = cache_dir / f"tt_{task_type.lower()}"
    tt_cache.mkdir(parents=True, exist_ok=True)

    # Create a fresh embedder with this task type and its own cache
    embedder = GeminiEmbedder(
        cache_dir=tt_cache,
        enable_cache=True,
        task_type=task_type,
    )

    # Collect per-sample metrics
    cos_ti_scores = []
    cos_ta_scores = []
    cos_ia_scores = []
    gram_ti_cohs = []
    gram_ta_cohs = []
    gram_ia_cohs = []
    gram_3d_cohs = []
    humans = []

    n_samples = 0
    for s in samples:
        sid = s["sample_id"]
        if sid not in human_scores:
            continue

        text = s["prompt_text"]
        image_path = s.get("image_path")
        audio_path = s.get("audio_path")

        try:
            emb_t = embedder.embed_text(text)
            emb_i = embedder.embed_image(image_path) if image_path and Path(image_path).exists() else None
            emb_a = embedder.embed_audio(audio_path) if audio_path and Path(audio_path).exists() else None
        except Exception as e:
            print(f"    WARN: Failed to embed {sid}: {e}")
            continue

        h = human_scores[sid]["weighted_score"]["mean"]

        if emb_t is not None and emb_i is not None:
            cos_ti = float(np.dot(emb_t, emb_i))
            cos_ti_scores.append(cos_ti)
            vol_ti = gram_volume_2d(emb_t, emb_i)
            gram_ti_cohs.append(normalized_gram_coherence(vol_ti))

        if emb_t is not None and emb_a is not None:
            cos_ta = float(np.dot(emb_t, emb_a))
            cos_ta_scores.append(cos_ta)
            vol_ta = gram_volume_2d(emb_t, emb_a)
            gram_ta_cohs.append(normalized_gram_coherence(vol_ta))

        if emb_i is not None and emb_a is not None:
            cos_ia = float(np.dot(emb_i, emb_a))
            cos_ia_scores.append(cos_ia)
            vol_ia = gram_volume_2d(emb_i, emb_a)
            gram_ia_cohs.append(normalized_gram_coherence(vol_ia))

        if emb_t is not None and emb_i is not None and emb_a is not None:
            vol_3d = gram_volume_3d(emb_t, emb_i, emb_a)
            gram_3d_cohs.append(normalized_gram_coherence(vol_3d, n_vectors=3))

        humans.append(h)
        n_samples += 1
        print(f"    [{n_samples}] {sid}", end="\r")

    print()

    # Compute correlations
    def safe_rho(x, y):
        if len(x) != len(y) or len(x) < 5:
            return None, None
        rho, p = sp_stats.spearmanr(x, y)
        return float(rho), float(p)

    # Average cosine (2-channel: TI+TA only, since IA is noisy)
    cos_2ch_scores = []
    for i in range(min(len(cos_ti_scores), len(cos_ta_scores))):
        cos_2ch_scores.append(0.5 * cos_ti_scores[i] + 0.5 * cos_ta_scores[i])

    # Average cosine (3-channel)
    cos_3ch_scores = []
    for i in range(min(len(cos_ti_scores), len(cos_ta_scores), len(cos_ia_scores))):
        cos_3ch_scores.append(
            (cos_ti_scores[i] + cos_ta_scores[i] + cos_ia_scores[i]) / 3.0
        )

    # Average gram coherence (2-channel)
    gram_2ch_scores = []
    for i in range(min(len(gram_ti_cohs), len(gram_ta_cohs))):
        gram_2ch_scores.append(0.5 * gram_ti_cohs[i] + 0.5 * gram_ta_cohs[i])

    result = {
        "task_type": task_type,
        "n_samples": n_samples,
        "metrics": {},
    }

    for name, scores in [
        ("cos_ti", cos_ti_scores),
        ("cos_ta", cos_ta_scores),
        ("cos_ia", cos_ia_scores),
        ("cos_2ch", cos_2ch_scores),
        ("cos_3ch", cos_3ch_scores),
        ("gram_ti", gram_ti_cohs),
        ("gram_ta", gram_ta_cohs),
        ("gram_ia", gram_ia_cohs),
        ("gram_2ch", gram_2ch_scores),
        ("gram_3d", gram_3d_cohs),
    ]:
        rho, p = safe_rho(scores, humans[:len(scores)])
        std = float(np.std(scores)) if scores else None
        result["metrics"][name] = {
            "rho": rho,
            "p": p,
            "std": std,
            "n": len(scores),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Gemini Task-Type Experiment")
    parser.add_argument(
        "--task-types", nargs="+", default=ALL_TASK_TYPES,
        help="Task types to test (default: all 5)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Gemini Embedding Task-Type Experiment")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]
    human_scores = load_human_scores()
    print(f"  {len(samples)} samples, {len(human_scores)} human ratings")

    # Run experiments
    all_results = []
    for i, task_type in enumerate(args.task_types):
        print(f"\n--- [{i+1}/{len(args.task_types)}] Task Type: {task_type} ---")
        t0 = time.time()
        result = evaluate_task_type(
            task_type, samples, human_scores,
            cache_dir=GEMINI_CACHE_DIR.parent / "embeddings_gemini_experiment",
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s ({result['n_samples']} samples)")
        all_results.append(result)

    # =====================================================================
    # RESULTS TABLE
    # =====================================================================
    print(f"\n{'='*70}")
    print("TASK-TYPE COMPARISON TABLE")
    print(f"{'='*70}")

    # Header
    key_metrics = ["cos_2ch", "cos_3ch", "gram_2ch", "gram_3d", "gram_ti", "gram_ta", "gram_ia"]
    print(f"\n  {'Task Type':<25s}", end="")
    for m in key_metrics:
        print(f"  {m:>10s}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in key_metrics:
        print(f"  {'-'*10}", end="")
    print()

    for result in all_results:
        print(f"  {result['task_type']:<25s}", end="")
        for m in key_metrics:
            info = result["metrics"].get(m, {})
            rho = info.get("rho")
            if rho is not None:
                sig = "*" if info.get("p", 1.0) < 0.05 else " "
                print(f"  {rho:9.3f}{sig}", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print()

    # Channel standard deviations (discriminative power)
    print(f"\n{'='*70}")
    print("CHANNEL STANDARD DEVIATIONS (higher = more discriminative)")
    print(f"{'='*70}")

    std_metrics = ["gram_ti", "gram_ta", "gram_ia", "gram_3d"]
    print(f"\n  {'Task Type':<25s}", end="")
    for m in std_metrics:
        print(f"  {m+'_std':>12s}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in std_metrics:
        print(f"  {'-'*12}", end="")
    print()

    for result in all_results:
        print(f"  {result['task_type']:<25s}", end="")
        for m in std_metrics:
            info = result["metrics"].get(m, {})
            std = info.get("std")
            if std is not None:
                print(f"  {std:12.6f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    # Find best task type
    print(f"\n{'='*70}")
    print("BEST TASK TYPE PER METRIC")
    print(f"{'='*70}")

    for m in key_metrics:
        best_tt = None
        best_rho = -999
        for result in all_results:
            rho = result["metrics"].get(m, {}).get("rho")
            if rho is not None and rho > best_rho:
                best_rho = rho
                best_tt = result["task_type"]
        if best_tt:
            print(f"  {m:<15s}: {best_tt:<25s} (rho={best_rho:.3f})")

    # Save results
    output_dir = PROJECT_ROOT / "runs" / "task_type_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    save_data = []
    for r in all_results:
        save_data.append({
            "task_type": r["task_type"],
            "n_samples": r["n_samples"],
            "metrics": {
                k: {"rho": v.get("rho"), "p": v.get("p"), "std": v.get("std")}
                for k, v in r["metrics"].items()
            },
        })

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
