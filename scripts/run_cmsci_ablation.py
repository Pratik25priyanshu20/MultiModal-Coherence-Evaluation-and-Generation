"""
cMSCI Ablation Study: Layer-by-Layer Variant Analysis.

Tests each cMSCI layer independently to measure its marginal contribution:
    Variant A: MSCI (baseline, weighted cosine average)
    Variant B: GRAM-only (geometric, no calibration)
    Variant C: GRAM + z-norm (normalized geometric)
    Variant D: GRAM + z-norm + contrastive (calibrated geometric)
    Variant E: GRAM + z-norm + contrastive + Ex-MCR (3-way calibrated)
    Variant F: Full cMSCI (probabilistic + calibrated + 3-way)

Reports Cohen's d and human correlation for each variant.

Usage:
    python scripts/run_cmsci_ablation.py
    python scripts/run_cmsci_ablation.py --rq1-path runs/rq1/rq1_results.json
    python scripts/run_cmsci_ablation.py --human-path data/human_eval/rq3_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


VARIANT_LABELS = {
    "A_msci": "A: MSCI (weighted cosine)",
    "B_gram": "B: GRAM-only (geometric)",
    "C_gram_znorm": "C: GRAM + z-norm",
    "D_gram_znorm_contrastive": "D: GRAM + z-norm + contrastive",
    "E_gram_znorm_contrastive_exmcr": "E: GRAM + z-norm + contrastive + Ex-MCR",
    "F_full_cmsci": "F: Full cMSCI (probabilistic)",
}


def load_comparison_results(path: str) -> List[Dict[str, Any]]:
    """Load results from cmsci_comparison.json."""
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def compute_effect_size(
    results: List[Dict[str, Any]],
    variant_key: str,
    baseline_cond: str = "baseline",
    perturb_cond: str = "wrong_image",
) -> Dict[str, Any]:
    """
    Compute paired Cohen's d for a variant between baseline and perturbation.

    Aggregates by prompt (mean across seeds), then computes paired d.
    """
    prompt_scores = defaultdict(lambda: defaultdict(list))

    for r in results:
        cmsci_result = r.get("cmsci_result")
        if cmsci_result is None:
            continue
        variant_scores = cmsci_result.get("variant_scores", {})
        score = variant_scores.get(variant_key)
        if score is None:
            continue
        condition = r.get("condition")
        pid = r.get("prompt_id")
        prompt_scores[condition][pid].append(score)

    if baseline_cond not in prompt_scores or perturb_cond not in prompt_scores:
        return {"d": None, "n": 0, "mean_diff": None}

    base_avgs = {pid: np.mean(scores) for pid, scores in prompt_scores[baseline_cond].items()}
    pert_avgs = {pid: np.mean(scores) for pid, scores in prompt_scores[perturb_cond].items()}
    common = sorted(set(base_avgs) & set(pert_avgs))

    if len(common) < 3:
        return {"d": None, "n": len(common), "mean_diff": None}

    diffs = [base_avgs[pid] - pert_avgs[pid] for pid in common]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    d = mean_diff / std_diff if std_diff > 1e-10 else 0.0

    # CI for d
    se_d = np.sqrt(1 / len(common) + d**2 / (2 * len(common)))
    z = 1.96
    ci_lower = d - z * se_d
    ci_upper = d + z * se_d

    return {
        "d": float(d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": len(common),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
    }


def compute_human_correlation(
    results: List[Dict[str, Any]],
    human_results_path: str,
    variant_key: str,
) -> Optional[float]:
    """
    Compute Spearman's ρ between variant scores and human ratings.

    Returns None if human results are not available.
    """
    if not Path(human_results_path).exists():
        return None

    with open(human_results_path) as f:
        human_data = json.load(f)

    # Build prompt -> human score mapping
    human_scores = {}
    for item in human_data.get("ratings", human_data.get("results", [])):
        pid = item.get("prompt_id")
        score = item.get("mean_rating", item.get("human_score"))
        if pid and score is not None:
            human_scores[pid] = score

    if not human_scores:
        return None

    # Build prompt -> variant score mapping (baseline only, averaged across seeds)
    variant_by_prompt = defaultdict(list)
    for r in results:
        if r.get("condition") != "baseline":
            continue
        cmsci_result = r.get("cmsci_result")
        if cmsci_result is None:
            continue
        score = cmsci_result.get("variant_scores", {}).get(variant_key)
        if score is not None:
            variant_by_prompt[r["prompt_id"]].append(score)

    variant_avgs = {pid: np.mean(scores) for pid, scores in variant_by_prompt.items()}

    common = sorted(set(human_scores) & set(variant_avgs))
    if len(common) < 5:
        return None

    from scipy.stats import spearmanr
    human_arr = [human_scores[pid] for pid in common]
    variant_arr = [variant_avgs[pid] for pid in common]
    rho, _ = spearmanr(human_arr, variant_arr)
    return float(rho)


def run_ablation(
    comparison_path: str,
    human_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full ablation analysis."""
    results = load_comparison_results(comparison_path)

    print("\n" + "=" * 90)
    print("cMSCI ABLATION STUDY")
    print("=" * 90)
    print(f"  Source: {comparison_path}")
    print(f"  Samples: {len(results)}")

    ablation_results = {}

    # Header
    print(f"\n  {'Variant':<50} {'d(WI)':>8} {'d(WA)':>8} {'ρ(human)':>10}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*10}")

    for variant_key, label in VARIANT_LABELS.items():
        # Effect sizes
        es_wi = compute_effect_size(results, variant_key, "baseline", "wrong_image")
        es_wa = compute_effect_size(results, variant_key, "baseline", "wrong_audio")

        # Human correlation
        rho = None
        if human_path:
            rho = compute_human_correlation(results, human_path, variant_key)

        d_wi_str = f"{es_wi['d']:.3f}" if es_wi['d'] is not None else "N/A"
        d_wa_str = f"{es_wa['d']:.3f}" if es_wa['d'] is not None else "N/A"
        rho_str = f"{rho:.3f}" if rho is not None else "N/A"

        print(f"  {label:<50} {d_wi_str:>8} {d_wa_str:>8} {rho_str:>10}")

        ablation_results[variant_key] = {
            "label": label,
            "effect_size_wrong_image": es_wi,
            "effect_size_wrong_audio": es_wa,
            "human_correlation": rho,
        }

    # Descriptive statistics per variant per condition
    print(f"\n  DESCRIPTIVE STATISTICS BY VARIANT AND CONDITION")
    print(f"  {'-'*90}")

    for variant_key, label in VARIANT_LABELS.items():
        for condition in ["baseline", "wrong_image", "wrong_audio"]:
            scores = []
            for r in results:
                if r.get("condition") != condition:
                    continue
                cmsci_result = r.get("cmsci_result")
                if cmsci_result is None:
                    continue
                score = cmsci_result.get("variant_scores", {}).get(variant_key)
                if score is not None:
                    scores.append(score)

            if scores:
                variant_short = variant_key.split("_")[0]
                print(
                    f"  {variant_short} {condition:<14}  "
                    f"N={len(scores):>3}  mean={np.mean(scores):.4f}  "
                    f"std={np.std(scores):.4f}  median={np.median(scores):.4f}"
                )

    # Summary
    print(f"\n  IMPROVEMENT SUMMARY")
    print(f"  {'-'*60}")

    baseline_key = "A_msci"
    for variant_key, label in VARIANT_LABELS.items():
        if variant_key == baseline_key:
            continue
        es_a = compute_effect_size(results, baseline_key, "baseline", "wrong_image")
        es_v = compute_effect_size(results, variant_key, "baseline", "wrong_image")
        if es_a["d"] is not None and es_v["d"] is not None:
            delta = es_v["d"] - es_a["d"]
            pct = 100 * delta / abs(es_a["d"]) if abs(es_a["d"]) > 0.01 else 0
            print(f"  {label}: Δd = {delta:+.3f} ({pct:+.0f}% vs MSCI)")

    print("\n" + "=" * 90)

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="cMSCI Ablation Study")
    parser.add_argument("--comparison-path", default="runs/cmsci_comparison/cmsci_comparison.json")
    parser.add_argument("--rq1-path", default="runs/rq1/rq1_results.json",
                        help="Fallback: run comparison first if needed")
    parser.add_argument("--human-path", default="data/human_eval/rq3_results.json",
                        help="Human evaluation results for correlation analysis")
    parser.add_argument("--out-dir", default="runs/cmsci_ablation")
    args = parser.parse_args()

    # Run comparison first if needed
    if not Path(args.comparison_path).exists():
        print("Comparison results not found. Running comparison first...")
        if not Path(args.rq1_path).exists():
            print(f"ERROR: RQ1 results not found: {args.rq1_path}")
            return 1
        from scripts.run_cmsci_comparison import run_comparison, fit_calibration_if_needed
        fit_calibration_if_needed(args.rq1_path)
        results = run_comparison(args.rq1_path)
        out_path = Path(args.comparison_path).parent
        out_path.mkdir(parents=True, exist_ok=True)
        with open(args.comparison_path, "w") as f:
            json.dump({"results": results}, f, indent=2, default=str)

    ablation = run_ablation(
        comparison_path=args.comparison_path,
        human_path=args.human_path if Path(args.human_path).exists() else None,
    )

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_file = out_dir / "cmsci_ablation.json"
    with ablation_file.open("w") as f:
        json.dump({
            "experiment": "cMSCI Ablation Study",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ablation": ablation,
        }, f, indent=2, default=str)
    print(f"\n  Ablation saved: {ablation_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
