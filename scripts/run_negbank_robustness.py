#!/usr/bin/env python3
"""
Negative Bank Robustness Testing (Task 4.2).

Tests sensitivity of cMSCI to negative bank configuration:
- Vary negative bank size K: {1, 3, 5, 10, 20}
- Vary negative selection strategy: random vs hard (domain-based)

Usage:
    python scripts/run_negbank_robustness.py
    python scripts/run_negbank_robustness.py --dev-only
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
OUTPUT_DIR = PROJECT_ROOT / "runs" / "robustness"

K_VALUES = [1, 3, 5, 10, 20]


def load_data(dev_only: bool = False):
    """Load samples and human scores."""
    from scripts.optimize_cmsci import load_human_scores

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]

    human_scores = load_human_scores()

    if dev_only:
        from src.experiments.data_splits import get_dev_test_split
        dev_ids, _ = get_dev_test_split()
        dev_set = set(dev_ids)
        samples = [s for s in samples if s["sample_id"] in dev_set]

    return samples, human_scores


def run_with_k(
    samples: list,
    human_scores: dict,
    k: int,
    strategy: str = "hard",
) -> dict:
    """Run cMSCI evaluation with specific negative bank size K.

    Args:
        samples: RQ3 samples.
        human_scores: Human score dict.
        k: Number of hard negatives per modality.
        strategy: "hard" (domain-based) or "random".

    Returns:
        Dict with k, strategy, rho, p.
    """
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

    # Override K in the negative bank if available
    if engine._negative_bank is not None:
        engine._negative_bank._default_k = k

    cmsci_scores = []
    human_vals = []

    for s in samples:
        sid = s["sample_id"]
        if sid not in human_scores:
            continue

        # For random strategy, randomize domain to disable domain-based filtering
        domain = s.get("domain", "") if strategy == "hard" else ""

        result = engine.evaluate(
            text=s["prompt_text"],
            image_path=s.get("image_path"),
            audio_path=s.get("audio_path"),
            domain=domain,
            n_mc_samples=50,  # Reduced for speed
        )

        cmsci = result["variant_scores"]["F_full_cmsci"]
        if cmsci is None:
            cmsci = result["cmsci"]

        cmsci_scores.append(cmsci)
        human_vals.append(human_scores[sid]["weighted_score"]["mean"])

    rho, p = sp_stats.spearmanr(cmsci_scores, human_vals)

    return {
        "k": k,
        "strategy": strategy,
        "rho": float(rho),
        "p": float(p),
        "n_samples": len(cmsci_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Negative Bank Robustness Testing")
    parser.add_argument("--dev-only", action="store_true", help="Use dev set only")
    args = parser.parse_args()

    print("=" * 70)
    print("Negative Bank Robustness Testing")
    print("=" * 70)

    samples, human_scores = load_data(args.dev_only)
    print(f"  {len(samples)} samples, {len(human_scores)} with human ratings")

    results = []

    for strategy in ["hard", "random"]:
        print(f"\n--- Strategy: {strategy} ---")
        for k in K_VALUES:
            t0 = time.time()
            result = run_with_k(samples, human_scores, k, strategy)
            elapsed = time.time() - t0
            results.append(result)
            sig = "*" if result["p"] < 0.05 else ""
            print(f"  K={k:2d}: rho={result['rho']:.4f} (p={result['p']:.4f}){sig}  [{elapsed:.1f}s]")

    # Summary
    print(f"\n{'='*70}")
    print("NEGATIVE BANK ROBUSTNESS SUMMARY")
    print(f"{'='*70}")

    for strategy in ["hard", "random"]:
        strat_results = [r for r in results if r["strategy"] == strategy]
        rhos = [r["rho"] for r in strat_results]
        print(f"\n  Strategy: {strategy}")
        print(f"    Mean rho: {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}")
        print(f"    Range: [{min(rhos):.4f}, {max(rhos):.4f}]")
        best_k = strat_results[np.argmax(rhos)]["k"]
        print(f"    Best K: {best_k}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "negative_bank_robustness",
        "dev_only": args.dev_only,
        "k_values": K_VALUES,
        "strategies": ["hard", "random"],
        "results": results,
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

    output_path = OUTPUT_DIR / "negbank_robustness.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
