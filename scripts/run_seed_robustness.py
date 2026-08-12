#!/usr/bin/env python3
"""
Seed Robustness Testing (Task 4.1).

Re-runs the full cMSCI pipeline with 10 different random seeds to test
stability of the rho correlation with human ratings.

Varies:
- Negative bank sampling order
- MC dropout randomness
- Any stochastic components

Reports: mean rho +/- std across seeds, min/max rho, per-seed breakdown.

Usage:
    python scripts/run_seed_robustness.py
    python scripts/run_seed_robustness.py --n-seeds 20
    python scripts/run_seed_robustness.py --dev-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
SESSION_DIR = PROJECT_ROOT / "runs" / "rq3" / "sessions"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "robustness"


def load_data():
    """Load samples and human scores."""
    from scripts.optimize_cmsci import load_human_scores
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]
    human_scores = load_human_scores()
    return samples, human_scores


def run_single_seed(
    samples: list,
    human_scores: dict,
    seed: int,
    sample_ids: set | None = None,
) -> dict:
    """Run cMSCI evaluation with a specific seed and return rho.

    Args:
        samples: All RQ3 samples.
        human_scores: Human score dict.
        seed: Random seed.
        sample_ids: If provided, only evaluate these sample IDs.

    Returns:
        Dict with seed, rho, p-value, per-sample scores.
    """
    np.random.seed(seed)

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

    cmsci_scores = []
    human_vals = []
    per_sample = []

    for s in samples:
        sid = s["sample_id"]
        if sample_ids is not None and sid not in sample_ids:
            continue
        if sid not in human_scores:
            continue

        result = engine.evaluate(
            text=s["prompt_text"],
            image_path=s.get("image_path"),
            audio_path=s.get("audio_path"),
            domain=s.get("domain", ""),
            n_mc_samples=100,
        )

        cmsci = result["variant_scores"]["F_full_cmsci"]
        if cmsci is None:
            cmsci = result["cmsci"]

        human_val = human_scores[sid]["weighted_score"]["mean"]
        cmsci_scores.append(cmsci)
        human_vals.append(human_val)
        per_sample.append({
            "sample_id": sid,
            "cmsci": cmsci,
            "human": human_val,
        })

    rho, p = sp_stats.spearmanr(cmsci_scores, human_vals)

    return {
        "seed": seed,
        "rho": float(rho),
        "p": float(p),
        "n_samples": len(cmsci_scores),
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Seed Robustness Testing")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--dev-only", action="store_true", help="Only use dev set samples")
    args = parser.parse_args()

    print("=" * 70)
    print("Seed Robustness Testing")
    print("=" * 70)

    samples, human_scores = load_data()
    print(f"  {len(samples)} samples, {len(human_scores)} with human ratings")

    sample_ids = None
    if args.dev_only:
        from src.experiments.data_splits import get_dev_test_split
        dev_ids, _ = get_dev_test_split()
        sample_ids = set(dev_ids)
        print(f"  Using dev set only: {len(sample_ids)} samples")

    seeds = list(range(42, 42 + args.n_seeds))
    results = []

    for i, seed in enumerate(seeds):
        t0 = time.time()
        result = run_single_seed(samples, human_scores, seed, sample_ids)
        elapsed = time.time() - t0
        results.append(result)
        sig = "*" if result["p"] < 0.05 else ""
        print(f"  Seed {seed:3d}: rho={result['rho']:.4f} (p={result['p']:.4f}){sig}  [{elapsed:.1f}s]")

    # Summary
    rhos = [r["rho"] for r in results]
    print(f"\n{'='*70}")
    print(f"SEED ROBUSTNESS SUMMARY ({args.n_seeds} seeds)")
    print(f"{'='*70}")
    print(f"  Mean rho:   {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}")
    print(f"  Median rho: {np.median(rhos):.4f}")
    print(f"  Min rho:    {np.min(rhos):.4f} (seed={seeds[np.argmin(rhos)]})")
    print(f"  Max rho:    {np.max(rhos):.4f} (seed={seeds[np.argmax(rhos)]})")
    print(f"  Range:      {np.max(rhos) - np.min(rhos):.4f}")
    n_sig = sum(1 for r in results if r["p"] < 0.05)
    print(f"  Significant: {n_sig}/{args.n_seeds} ({100*n_sig/args.n_seeds:.0f}%)")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "seed_robustness",
        "n_seeds": args.n_seeds,
        "dev_only": args.dev_only,
        "summary": {
            "mean_rho": float(np.mean(rhos)),
            "std_rho": float(np.std(rhos)),
            "median_rho": float(np.median(rhos)),
            "min_rho": float(np.min(rhos)),
            "max_rho": float(np.max(rhos)),
            "range_rho": float(np.max(rhos) - np.min(rhos)),
            "n_significant": n_sig,
        },
        "per_seed": results,
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_path = OUTPUT_DIR / "seed_robustness.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
