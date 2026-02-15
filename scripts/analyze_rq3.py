#!/usr/bin/env python3
"""
RQ3 Analysis — Does MSCI correlate with human perception?

Aggregates multi-rater evaluation sessions and computes:
- Inter-rater reliability (Krippendorff's alpha)
- Intra-rater reliability per evaluator (Cohen's kappa)
- Spearman correlation between mean human score and MSCI
- Per-condition breakdown
- RQ3 verdict

Usage:
    python scripts/analyze_rq3.py
    python scripts/analyze_rq3.py --min-raters 3   # require >=3 raters per sample
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.human_eval_schema import EvaluationSession
from src.evaluation.human_eval_analyzer import (
    compute_intra_rater_reliability,
    compute_inter_rater_reliability,
    aggregate_multi_rater_sessions,
    compute_multi_rater_msci_correlation,
)

SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
SESSION_DIR = PROJECT_ROOT / "runs" / "rq3" / "sessions"
OUTPUT_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_analysis.json"


def load_sessions(min_progress: float = 80.0) -> list:
    """Load all sufficiently complete evaluation sessions."""
    sessions = []
    if not SESSION_DIR.exists():
        return sessions

    for p in sorted(SESSION_DIR.glob("*.json")):
        try:
            session = EvaluationSession.load(p)
            if session.progress >= min_progress:
                sessions.append(session)
            else:
                print(f"  Skipping {session.evaluator_id} ({session.progress:.0f}% complete)")
        except Exception as e:
            print(f"  Error loading {p.name}: {e}")

    return sessions


def load_sample_msci() -> dict:
    """Load MSCI scores from RQ3 sample selection."""
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    return {s["sample_id"]: s["msci"] for s in data["samples"]}


def analyze_by_condition_multi(sessions, sample_msci, sample_conditions):
    """Per-condition analysis across all raters."""
    aggregated = aggregate_multi_rater_sessions(sessions)

    by_cond = defaultdict(lambda: {"human": [], "msci": []})

    for sample_id, agg in aggregated.items():
        cond = sample_conditions.get(sample_id, "unknown")
        by_cond[cond]["human"].append(agg["weighted_score"]["mean"])
        by_cond[cond]["msci"].append(sample_msci.get(sample_id, 0))

    results = {}
    for cond, data in sorted(by_cond.items()):
        h = np.array(data["human"])
        m = np.array(data["msci"])
        results[cond] = {
            "n": len(h),
            "human_weighted_mean": round(float(np.mean(h)), 4),
            "human_weighted_std": round(float(np.std(h)), 4),
            "msci_mean": round(float(np.mean(m)), 4),
            "msci_std": round(float(np.std(m)), 4),
        }
        if len(h) >= 5:
            rho, p = stats.spearmanr(m, h)
            results[cond]["spearman_rho"] = round(float(rho), 4)
            results[cond]["spearman_p"] = float(p)

    return results


def determine_verdict(correlation_result: dict) -> str:
    """Determine RQ3 verdict based on correlation strength and significance."""
    if "error" in correlation_result:
        return "INCONCLUSIVE — Insufficient data"

    rho = correlation_result["spearman_rho"]
    p = correlation_result["spearman_p"]

    if p >= 0.05:
        return f"NOT SUPPORTED — No significant correlation (rho={rho:.3f}, p={p:.4f})"

    if abs(rho) >= 0.6:
        strength = "strong"
    elif abs(rho) >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if rho > 0 else "negative"

    if abs(rho) >= 0.3 and p < 0.05:
        return (f"SUPPORTED — Significant {strength} {direction} correlation "
                f"(rho={rho:.3f}, p={p:.4f})")
    else:
        return (f"PARTIALLY SUPPORTED — Significant but {strength} correlation "
                f"(rho={rho:.3f}, p={p:.4f})")


def main():
    parser = argparse.ArgumentParser(description="RQ3 Analysis")
    parser.add_argument("--min-raters", type=int, default=2,
                        help="Minimum raters per sample (default: 2)")
    parser.add_argument("--min-progress", type=float, default=80.0,
                        help="Minimum session progress %% to include (default: 80)")
    args = parser.parse_args()

    print("=" * 60)
    print("RQ3 Analysis: MSCI vs Human Perception")
    print("=" * 60)

    # Load sessions
    sessions = load_sessions(min_progress=args.min_progress)
    print(f"\nLoaded {len(sessions)} evaluation sessions:")
    for s in sessions:
        print(f"  {s.evaluator_id}: {len([e for e in s.evaluations if not e.is_rerating])} ratings")

    if len(sessions) < 2:
        print("\nERROR: Need at least 2 completed sessions for multi-rater analysis.")
        print(f"Found: {len(sessions)}")
        print(f"\nHave evaluators run:  python scripts/run_human_eval.py -e <name>")
        sys.exit(1)

    # Load MSCI scores
    sample_msci = load_sample_msci()

    # Load condition labels
    with open(SAMPLES_PATH) as f:
        rq3_data = json.load(f)
    sample_conditions = {s["sample_id"]: s["condition"] for s in rq3_data["samples"]}

    # --- 1. Inter-rater reliability ---
    print("\n--- Inter-Rater Reliability (Krippendorff's alpha) ---")
    irr = compute_inter_rater_reliability(sessions)
    if "error" not in irr:
        for dim in ["text_image", "text_audio", "image_audio", "overall", "weighted_score"]:
            if dim in irr:
                alpha = irr[dim]["krippendorff_alpha"]
                interp = irr[dim]["interpretation"]
                print(f"  {dim:15s}: alpha = {alpha:.4f}  ({interp})")
    else:
        print(f"  {irr['error']}")

    # --- 2. Intra-rater reliability (per evaluator) ---
    print("\n--- Intra-Rater Reliability (per evaluator) ---")
    intra_results = {}
    for session in sessions:
        rel = compute_intra_rater_reliability(session)
        if rel:
            intra_results[session.evaluator_id] = rel.to_dict()
            print(f"  {session.evaluator_id:15s}: kappa = {rel.kappa:.4f}, "
                  f"weighted_kappa = {rel.weighted_kappa:.4f}, "
                  f"n_reratings = {rel.n_reratings}")
        else:
            intra_results[session.evaluator_id] = None
            print(f"  {session.evaluator_id:15s}: no re-ratings available")

    # --- 3. MSCI vs Human correlation ---
    print("\n--- MSCI vs Human Score Correlation ---")
    correlation = compute_multi_rater_msci_correlation(sessions, sample_msci)
    if "error" not in correlation:
        print(f"  N paired:      {correlation['n_paired']}")
        print(f"  Spearman rho:  {correlation['spearman_rho']:.4f}")
        print(f"  Spearman p:    {correlation['spearman_p']:.6f}")
        print(f"  95% CI:        [{correlation['spearman_95ci'][0]:.4f}, "
              f"{correlation['spearman_95ci'][1]:.4f}]")
        print(f"  Pearson r:     {correlation['pearson_r']:.4f}")
        print(f"  Pearson p:     {correlation['pearson_p']:.6f}")
        print(f"  Interpretation: {correlation['interpretation']}")
    else:
        print(f"  {correlation['error']}")

    # --- 4. Per-condition breakdown ---
    print("\n--- Per-Condition Analysis ---")
    by_condition = analyze_by_condition_multi(sessions, sample_msci, sample_conditions)
    print(f"  {'Condition':15s} {'N':>3s} {'Human':>8s} {'MSCI':>8s}")
    print(f"  {'-'*15} {'-'*3} {'-'*8} {'-'*8}")
    for cond, data in sorted(by_condition.items()):
        print(f"  {cond:15s} {data['n']:3d} {data['human_weighted_mean']:8.4f} "
              f"{data['msci_mean']:8.4f}")

    # --- 5. Verdict ---
    verdict = determine_verdict(correlation)
    print(f"\n{'='*60}")
    print(f"RQ3 VERDICT: {verdict}")
    print(f"{'='*60}")

    # --- Save full report ---
    report = {
        "experiment": "RQ3: Human Alignment Validation",
        "n_evaluators": len(sessions),
        "evaluators": [s.evaluator_id for s in sessions],
        "n_samples": len(sample_msci),
        "inter_rater_reliability": irr,
        "intra_rater_reliability": intra_results,
        "msci_correlation": correlation,
        "by_condition": by_condition,
        "verdict": verdict,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
