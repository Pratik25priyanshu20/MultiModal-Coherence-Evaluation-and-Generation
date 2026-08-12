#!/usr/bin/env python3
"""
Failure Case Analysis (Task 5.1).

Identifies samples where cMSCI and human ratings disagree most.
Categorizes failure modes and creates annotated failure gallery.

Usage:
    python scripts/analyze_failures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "failure_analysis"


def load_data():
    """Load samples, human scores, and cMSCI scores."""
    from scripts.optimize_cmsci import load_human_scores

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]

    human_scores = load_human_scores()
    return samples, human_scores


def compute_cmsci_scores(samples: list) -> dict:
    """Run cMSCI on all samples and return scores."""
    from src.coherence.cmsci_engine import CalibratedCoherenceEngine
    from src.config.settings import (
        CMSCI_CALIBRATION_PATH,
        EXMCR_WEIGHTS_PATH,
        BRIDGE_WEIGHTS_PATH,
        PROB_CLIP_ADAPTER_PATH,
        PROB_CLAP_ADAPTER_PATH,
    )

    engine = CalibratedCoherenceEngine(
        calibration_path=str(CMSCI_CALIBRATION_PATH) if CMSCI_CALIBRATION_PATH.exists() else None,
        exmcr_weights_path=str(EXMCR_WEIGHTS_PATH) if EXMCR_WEIGHTS_PATH.exists() else None,
        bridge_path=str(BRIDGE_WEIGHTS_PATH) if BRIDGE_WEIGHTS_PATH.exists() else None,
        prob_clip_adapter_path=str(PROB_CLIP_ADAPTER_PATH) if PROB_CLIP_ADAPTER_PATH.exists() else None,
        prob_clap_adapter_path=str(PROB_CLAP_ADAPTER_PATH) if PROB_CLAP_ADAPTER_PATH.exists() else None,
        negative_bank_enabled=True,
    )

    results = {}
    for i, s in enumerate(samples):
        result = engine.evaluate(
            text=s["prompt_text"],
            image_path=s.get("image_path"),
            audio_path=s.get("audio_path"),
            domain=s.get("domain", ""),
        )
        results[s["sample_id"]] = result
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}", end="\r")
    print()
    return results


def categorize_failure(
    sample: dict,
    cmsci_result: dict,
    human_score: float,
    cmsci_score: float,
    residual: float,
) -> list:
    """Categorize the failure mode for a sample.

    Args:
        sample: Sample dict.
        cmsci_result: Full cMSCI result dict.
        human_score: Human rating.
        cmsci_score: cMSCI score.
        residual: human_rank - cmsci_rank (positive = cMSCI underestimates).

    Returns:
        List of failure mode labels.
    """
    modes = []

    st_i = cmsci_result["scores"]["st_i"]
    st_a = cmsci_result["scores"]["st_a"]

    # Channel imbalance
    if st_i is not None and st_a is not None:
        if abs(st_i - st_a) > 0.3:
            modes.append("channel_imbalance")
            if st_i < 0.15:
                modes.append("weak_image_match")
            if st_a < 0.15:
                modes.append("weak_audio_match")

    # Condition-specific
    condition = sample.get("condition", "")
    if condition == "wrong_image" and residual > 0:
        modes.append("wrong_image_underpenalized")
    if condition == "wrong_audio" and residual > 0:
        modes.append("wrong_audio_underpenalized")

    # Low text-image score (possible CLIP truncation)
    if st_i is not None and st_i < 0.18:
        modes.append("possible_clip_truncation")

    # Audio ambiguity (environmental sounds are inherently ambiguous)
    if st_a is not None and 0.2 < st_a < 0.5:
        modes.append("audio_ambiguity")

    # Domain-related
    domain = sample.get("domain", "")
    if domain == "mixed":
        modes.append("mixed_domain")

    # Overconfident cMSCI (high score but low human rating)
    if residual < -2:  # rank residual
        modes.append("cmsci_overconfident")

    # Underconfident cMSCI (low score but high human rating)
    if residual > 2:
        modes.append("cmsci_underconfident")

    if not modes:
        modes.append("uncategorized")

    return modes


def main():
    print("=" * 70)
    print("Failure Case Analysis")
    print("=" * 70)

    print("\n--- Loading Data ---")
    samples, human_scores = load_data()

    print("\n--- Computing cMSCI Scores ---")
    cmsci_results = compute_cmsci_scores(samples)

    # Build paired data
    paired = []
    for s in samples:
        sid = s["sample_id"]
        if sid not in human_scores or sid not in cmsci_results:
            continue

        human_val = human_scores[sid]["weighted_score"]["mean"]
        cmsci_score = cmsci_results[sid]["variant_scores"]["F_full_cmsci"]
        if cmsci_score is None:
            cmsci_score = cmsci_results[sid]["cmsci"]

        paired.append({
            "sample": s,
            "human_score": human_val,
            "cmsci_score": cmsci_score,
            "cmsci_result": cmsci_results[sid],
        })

    # Compute rank-based residuals
    human_vals = [p["human_score"] for p in paired]
    cmsci_vals = [p["cmsci_score"] for p in paired]

    human_ranks = sp_stats.rankdata(human_vals)
    cmsci_ranks = sp_stats.rankdata(cmsci_vals)

    for i, p in enumerate(paired):
        p["human_rank"] = int(human_ranks[i])
        p["cmsci_rank"] = int(cmsci_ranks[i])
        p["rank_residual"] = int(human_ranks[i] - cmsci_ranks[i])
        p["abs_rank_residual"] = abs(p["rank_residual"])

    # Sort by absolute rank residual (largest disagreements first)
    paired.sort(key=lambda x: x["abs_rank_residual"], reverse=True)

    # Categorize failures
    print("\n--- Failure Analysis ---")
    print(f"  {'ID':>5s}  {'Domain':>8s}  {'Cond':>12s}  {'Human':>6s}  {'cMSCI':>6s}  "
          f"{'H-Rank':>6s}  {'C-Rank':>6s}  {'Resid':>6s}  Failure Modes")
    print(f"  {'-----':>5s}  {'--------':>8s}  {'------------':>12s}  {'------':>6s}  {'------':>6s}  "
          f"{'------':>6s}  {'------':>6s}  {'------':>6s}  {'---'}")

    failure_modes_count = {}
    failures = []

    for p in paired:
        modes = categorize_failure(
            p["sample"], p["cmsci_result"],
            p["human_score"], p["cmsci_score"],
            p["rank_residual"],
        )

        for m in modes:
            failure_modes_count[m] = failure_modes_count.get(m, 0) + 1

        failure = {
            "sample_id": p["sample"]["sample_id"],
            "domain": p["sample"].get("domain", ""),
            "condition": p["sample"].get("condition", ""),
            "prompt_text": p["sample"]["prompt_text"],
            "human_score": p["human_score"],
            "cmsci_score": p["cmsci_score"],
            "human_rank": p["human_rank"],
            "cmsci_rank": p["cmsci_rank"],
            "rank_residual": p["rank_residual"],
            "failure_modes": modes,
            "st_i": p["cmsci_result"]["scores"]["st_i"],
            "st_a": p["cmsci_result"]["scores"]["st_a"],
            "image_path": p["sample"].get("image_path"),
            "audio_path": p["sample"].get("audio_path"),
        }
        failures.append(failure)

        sid = p["sample"]["sample_id"]
        domain = p["sample"].get("domain", "?")[:8]
        cond = p["sample"].get("condition", "?")[:12]
        modes_str = ", ".join(modes)
        print(f"  {sid:>5s}  {domain:>8s}  {cond:>12s}  {p['human_score']:6.3f}  {p['cmsci_score']:6.3f}  "
              f"{p['human_rank']:6d}  {p['cmsci_rank']:6d}  {p['rank_residual']:6d}  {modes_str}")

    # Summary
    print(f"\n{'='*70}")
    print("FAILURE MODE SUMMARY")
    print(f"{'='*70}")
    for mode, count in sorted(failure_modes_count.items(), key=lambda x: -x[1]):
        print(f"  {mode:35s}: {count:3d} samples ({100*count/len(paired):.0f}%)")

    # Top 5 worst failures
    print(f"\n  Top 5 largest disagreements:")
    for i, f in enumerate(failures[:5]):
        print(f"    {i+1}. {f['sample_id']} (residual={f['rank_residual']:+d}): "
              f"human={f['human_score']:.3f}, cMSCI={f['cmsci_score']:.3f}")
        print(f"       \"{f['prompt_text'][:60]}...\"")
        print(f"       Modes: {', '.join(f['failure_modes'])}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "failure_analysis",
        "n_samples": len(paired),
        "failure_modes_summary": failure_modes_count,
        "failures": failures,
    }
    output_path = OUTPUT_DIR / "failure_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
