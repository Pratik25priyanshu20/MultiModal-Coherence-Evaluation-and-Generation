#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis (Task 4.3).

Sweeps each cMSCI hyperparameter while holding others at optimal values.
Generates sensitivity curve data for plotting.

Parameters swept (centered on current optimal from LOO-CV):
- alpha: [0, 1, 3, 5, 7, 10, 14, 20, 32]
- w_ti: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
- w_3d: [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]
- gamma: [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]

Usage:
    python scripts/run_sensitivity.py
    python scripts/run_sensitivity.py --dev-only
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
OUTPUT_DIR = PROJECT_ROOT / "runs" / "sensitivity"

# Optimal values (from LOO-CV optimization on 10,255-pair trained models)
OPTIMAL = {
    "alpha": 7,
    "w_ti": 0.30,
    "w_3d": 0.35,
    "gamma": 0.40,
    "cal_mode": "gram",
}

# Sweep ranges (centered on current optimal, spanning full parameter space)
SWEEPS = {
    "alpha": [0, 1, 3, 5, 7, 10, 14, 20, 32],
    "w_ti": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    "w_3d": [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0],
    "gamma": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
}


def load_data(dev_only: bool = False):
    """Load samples, human scores, and intermediates."""
    from scripts.optimize_cmsci import (
        load_human_scores,
        load_calibration,
        compute_gram_reference_from_rq1,
        compute_gram_reference,
        collect_intermediates_from_engine,
    )

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]

    human_scores = load_human_scores()
    calibration = load_calibration()
    gram_ref = compute_gram_reference_from_rq1()
    if gram_ref is None:
        gram_ref = compute_gram_reference(calibration)

    sample_ids = None
    if dev_only:
        from src.experiments.data_splits import get_dev_test_split
        dev_ids, _ = get_dev_test_split()
        sample_ids = set(dev_ids)

    # Collect intermediates
    intermediates = collect_intermediates_from_engine(samples)

    # Pair with human scores
    paired = []
    for inter in intermediates:
        sid = inter["sample_id"]
        if sample_ids is not None and sid not in sample_ids:
            continue
        if sid in human_scores:
            paired.append((inter, human_scores[sid]["weighted_score"]["mean"]))

    return paired, gram_ref


def sweep_parameter(
    param_name: str,
    values: list,
    paired: list,
    gram_ref: dict,
) -> list:
    """Sweep a single parameter while holding others at optimal.

    Returns list of (value, rho, p) tuples.
    """
    from scripts.optimize_cmsci import variant_f_score

    results = []
    for val in values:
        # Build config with this parameter varied
        config = dict(OPTIMAL)
        config[param_name] = val

        scores = []
        humans = []
        for inter, h in paired:
            s = variant_f_score(
                inter["z_st_i"], inter["z_st_a"], inter["margin"],
                inter.get("z_compl"), inter.get("u_ti"), inter.get("u_ta"),
                alpha=config["alpha"],
                w=config["w_ti"],
                w_3d=config["w_3d"],
                gamma=config["gamma"],
                gram_ti=inter["gram_ti"],
                gram_ta=inter["gram_ta"],
                gram_ref=gram_ref,
                cal_mode=config["cal_mode"],
            )
            scores.append(s)
            humans.append(h)

        rho, p = sp_stats.spearmanr(scores, humans)
        results.append({
            "value": val,
            "rho": float(rho),
            "p": float(p),
            "significant": p < 0.05,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Sensitivity Analysis")
    parser.add_argument("--dev-only", action="store_true", help="Use dev set only")
    args = parser.parse_args()

    print("=" * 70)
    print("Hyperparameter Sensitivity Analysis")
    print("=" * 70)
    print(f"  Optimal config: {OPTIMAL}")

    print("\n--- Loading Data ---")
    paired, gram_ref = load_data(args.dev_only)
    print(f"  {len(paired)} paired samples")

    all_sweeps = {}

    for param_name, values in SWEEPS.items():
        print(f"\n--- Sweeping {param_name} ---")
        results = sweep_parameter(param_name, values, paired, gram_ref)
        all_sweeps[param_name] = results

        # Print results
        optimal_val = OPTIMAL[param_name]
        for r in results:
            marker = " <-- optimal" if r["value"] == optimal_val else ""
            sig = "*" if r["significant"] else ""
            print(f"  {param_name}={r['value']:>6.2f}: rho={r['rho']:.4f} (p={r['p']:.4f}){sig}{marker}")

        # Sensitivity metrics
        rhos = [r["rho"] for r in results]
        print(f"  Range: [{min(rhos):.4f}, {max(rhos):.4f}] (spread={max(rhos)-min(rhos):.4f})")

    # Summary
    print(f"\n{'='*70}")
    print("SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    for param_name, results in all_sweeps.items():
        rhos = [r["rho"] for r in results]
        spread = max(rhos) - min(rhos)
        best_val = results[np.argmax(rhos)]["value"]
        sensitivity = "HIGH" if spread > 0.15 else "MODERATE" if spread > 0.05 else "LOW"
        print(f"  {param_name:>8s}: spread={spread:.4f} ({sensitivity}), "
              f"best={best_val}, optimal={OPTIMAL[param_name]}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "hyperparameter_sensitivity",
        "dev_only": args.dev_only,
        "optimal_config": OPTIMAL,
        "n_paired": len(paired),
        "sweeps": all_sweeps,
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

    output_path = OUTPUT_DIR / "sensitivity_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
