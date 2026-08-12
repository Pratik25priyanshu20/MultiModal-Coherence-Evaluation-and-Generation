"""
Side-by-side comparison of MSCI vs cMSCI on existing RQ1 data.

Reuses existing experiment results (same prompts, images, audio) and
recomputes scores using the CalibratedCoherenceEngine. This lets us
compare cMSCI against MSCI without running new experiments.

Outputs:
    runs/cmsci_comparison/cmsci_comparison.json — full results with both scores
    runs/cmsci_comparison/summary.txt           — human-readable summary

Usage:
    python scripts/run_cmsci_comparison.py
    python scripts/run_cmsci_comparison.py --rq1-path runs/rq1/rq1_results.json
    python scripts/run_cmsci_comparison.py --calibrate  # fit calibration first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coherence.cmsci_engine import CalibratedCoherenceEngine
from src.config.settings import (
    BRIDGE_WEIGHTS_PATH,
    CMSCI_CALIBRATION_PATH,
    EXMCR_WEIGHTS_PATH,
    PROB_CLIP_ADAPTER_PATH,
    PROB_CLAP_ADAPTER_PATH,
)


def load_rq1_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def fit_calibration_if_needed(rq1_path: str, force: bool = False) -> str:
    """Build calibration store from RQ1 baseline data."""
    cal_path = str(CMSCI_CALIBRATION_PATH)
    if Path(cal_path).exists() and not force:
        print(f"  Calibration already exists: {cal_path}")
        return cal_path

    from src.coherence.calibration import build_reference_distributions
    store = build_reference_distributions(rq1_path)
    store.save(cal_path)
    print(f"  Calibration saved: {cal_path}")
    for name, ref in store.distributions.items():
        print(f"    {name}: mean={ref.mean:.4f}, std={ref.std:.4f}, n={ref.n}")
    return cal_path


def run_comparison(
    rq1_path: str,
    calibrate: bool = True,
    out_dir: str = "runs/cmsci_comparison",
) -> List[Dict[str, Any]]:
    """
    Re-evaluate all RQ1 results with both MSCI and cMSCI.
    """
    data = load_rq1_results(rq1_path)
    results_in = data["results"]

    # Fit calibration
    cal_path = None
    if calibrate:
        cal_path = fit_calibration_if_needed(rq1_path)

    # Initialize cMSCI engine
    engine = CalibratedCoherenceEngine(
        calibration_path=cal_path if cal_path and Path(cal_path).exists() else None,
        exmcr_weights_path=str(EXMCR_WEIGHTS_PATH) if EXMCR_WEIGHTS_PATH.exists() else None,
        bridge_path=str(BRIDGE_WEIGHTS_PATH) if BRIDGE_WEIGHTS_PATH.exists() else None,
        prob_clip_adapter_path=str(PROB_CLIP_ADAPTER_PATH) if PROB_CLIP_ADAPTER_PATH.exists() else None,
        prob_clap_adapter_path=str(PROB_CLAP_ADAPTER_PATH) if PROB_CLAP_ADAPTER_PATH.exists() else None,
    )

    print(f"\n  Processing {len(results_in)} results...")
    t_start = time.time()

    results_out = []
    for i, r in enumerate(results_in):
        if "error" in r:
            results_out.append({**r, "cmsci_result": None})
            continue

        text = r.get("prompt_text", "")
        image_path = r.get("image_path")
        audio_path = r.get("audio_path")
        domain = r.get("domain", "")

        try:
            cmsci_result = engine.evaluate(
                text=text,
                image_path=image_path,
                audio_path=audio_path,
                domain=domain,
            )

            results_out.append({
                "prompt_id": r.get("prompt_id"),
                "prompt_text": text,
                "domain": domain,
                "seed": r.get("seed"),
                "condition": r.get("condition"),
                "image_path": image_path,
                "audio_path": audio_path,
                # Legacy scores
                "msci": r.get("msci"),
                "st_i": r.get("st_i"),
                "st_a": r.get("st_a"),
                # cMSCI scores
                "cmsci": cmsci_result["cmsci"],
                "cmsci_variant": cmsci_result["active_variant"],
                "cmsci_result": cmsci_result,
            })
        except Exception as e:
            results_out.append({**r, "cmsci_result": None, "cmsci_error": str(e)})

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t_start
            print(f"    [{i+1}/{len(results_in)}] ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"  Done: {len(results_out)} results in {elapsed:.1f}s")
    return results_out


def print_summary(results: List[Dict[str, Any]]) -> str:
    """Print and return comparison summary."""
    lines = []
    lines.append("=" * 90)
    lines.append("MSCI vs cMSCI COMPARISON")
    lines.append("=" * 90)

    for condition in ["baseline", "wrong_image", "wrong_audio"]:
        msci_scores = [r["msci"] for r in results if r.get("condition") == condition and r.get("msci") is not None]
        cmsci_scores = [r["cmsci"] for r in results if r.get("condition") == condition and r.get("cmsci") is not None]

        if msci_scores and cmsci_scores:
            lines.append(f"\n  {condition}:")
            lines.append(f"    MSCI:  N={len(msci_scores):>3}  mean={np.mean(msci_scores):.4f}  std={np.std(msci_scores):.4f}")
            lines.append(f"    cMSCI: N={len(cmsci_scores):>3}  mean={np.mean(cmsci_scores):.4f}  std={np.std(cmsci_scores):.4f}")

    # Effect sizes
    lines.append("\n  EFFECT SIZES (baseline vs perturbation):")

    # Aggregate by prompt
    prompt_msci = defaultdict(lambda: defaultdict(list))
    prompt_cmsci = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("msci") is not None:
            prompt_msci[r["condition"]][r["prompt_id"]].append(r["msci"])
        if r.get("cmsci") is not None:
            prompt_cmsci[r["condition"]][r["prompt_id"]].append(r["cmsci"])

    for metric_name, prompt_data in [("MSCI", prompt_msci), ("cMSCI", prompt_cmsci)]:
        for pert in ["wrong_image", "wrong_audio"]:
            if "baseline" not in prompt_data or pert not in prompt_data:
                continue
            base_avgs = {pid: np.mean(scores) for pid, scores in prompt_data["baseline"].items()}
            pert_avgs = {pid: np.mean(scores) for pid, scores in prompt_data[pert].items()}
            common = sorted(set(base_avgs) & set(pert_avgs))
            if len(common) < 3:
                continue
            diffs = [base_avgs[pid] - pert_avgs[pid] for pid in common]
            d = np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-10)
            lines.append(f"    {metric_name} baseline vs {pert}: d = {d:.3f} (N={len(common)})")

    # Correlation between MSCI and cMSCI
    paired = [(r["msci"], r["cmsci"]) for r in results if r.get("msci") is not None and r.get("cmsci") is not None]
    if paired:
        msci_arr = np.array([p[0] for p in paired])
        cmsci_arr = np.array([p[1] for p in paired])
        from scipy.stats import pearsonr, spearmanr
        r_p, _ = pearsonr(msci_arr, cmsci_arr)
        r_s, _ = spearmanr(msci_arr, cmsci_arr)
        lines.append(f"\n  MSCI-cMSCI correlation:")
        lines.append(f"    Pearson r  = {r_p:.4f}")
        lines.append(f"    Spearman ρ = {r_s:.4f}")

    # Variant distribution
    variants = [r.get("cmsci_variant", "?") for r in results if r.get("cmsci") is not None]
    if variants:
        from collections import Counter
        vc = Counter(variants)
        lines.append(f"\n  ACTIVE VARIANTS:")
        for v, count in sorted(vc.items()):
            lines.append(f"    {v}: {count} ({100*count/len(variants):.0f}%)")

    lines.append("\n" + "=" * 90)
    text = "\n".join(lines)
    print(text)
    return text


ALL_RESULT_FILES = [
    ("rq1", "runs/rq1/rq1_results.json", "runs/cmsci_comparison"),
    ("rq1_full", "runs/rq1_full/rq1_results.json", "runs/cmsci_comparison_full"),
    ("rq1_gen", "runs/rq1_gen/rq1_gen_results.json", "runs/cmsci_comparison_gen"),
    ("rq1_hybrid", "runs/rq1_hybrid/rq1_hybrid_results.json", "runs/cmsci_comparison_hybrid"),
]


def run_single(rq1_path: str, out_dir: str, calibrate: bool = True) -> int:
    """Run comparison for a single result file and save outputs."""
    if not Path(rq1_path).exists():
        print(f"  SKIP: {rq1_path} (not found)")
        return 0

    results = run_comparison(
        rq1_path=rq1_path,
        calibrate=calibrate,
        out_dir=out_dir,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_file = out_path / "cmsci_comparison.json"
    with results_file.open("w") as f:
        json.dump({
            "experiment": "MSCI vs cMSCI Comparison",
            "source": rq1_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_results": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {results_file}")

    summary = print_summary(results)
    summary_file = out_path / "summary.txt"
    summary_file.write_text(summary)
    print(f"  Summary saved: {summary_file}")

    return len(results)


def print_cross_run_summary(run_summaries: dict):
    """Print a summary table comparing effect sizes across all runs."""
    from scipy.stats import ttest_rel

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("CROSS-RUN cMSCI COMPARISON SUMMARY")
    lines.append("=" * 90)
    lines.append(f"\n  {'Run':15s} {'N':>5s}  {'Metric':>6s}  {'base→wrong_img d':>18s}  {'base→wrong_aud d':>18s}")
    lines.append(f"  {'-'*15} {'-'*5}  {'-'*6}  {'-'*18}  {'-'*18}")

    for label, rq1_path, out_dir in ALL_RESULT_FILES:
        comp_path = Path(out_dir) / "cmsci_comparison.json"
        if not comp_path.exists():
            continue
        with open(comp_path) as f:
            data = json.load(f)
        results = data["results"]
        n = len(results)

        for metric_name, metric_key in [("MSCI", "msci"), ("cMSCI", "cmsci")]:
            prompt_data = defaultdict(lambda: defaultdict(list))
            for r in results:
                val = r.get(metric_key)
                if val is not None and r.get("condition"):
                    prompt_data[r["condition"]][r.get("prompt_id", "?")].append(val)

            d_vals = {}
            for pert in ["wrong_image", "wrong_audio"]:
                if "baseline" not in prompt_data or pert not in prompt_data:
                    d_vals[pert] = "N/A"
                    continue
                base_avgs = {pid: np.mean(scores) for pid, scores in prompt_data["baseline"].items()}
                pert_avgs = {pid: np.mean(scores) for pid, scores in prompt_data[pert].items()}
                common = sorted(set(base_avgs) & set(pert_avgs))
                if len(common) < 3:
                    d_vals[pert] = "N/A"
                    continue
                diffs = [base_avgs[pid] - pert_avgs[pid] for pid in common]
                d = np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-10)
                d_vals[pert] = f"{d:.3f}"

            lines.append(
                f"  {label:15s} {n:5d}  {metric_name:>6s}  "
                f"{d_vals.get('wrong_image', 'N/A'):>18s}  "
                f"{d_vals.get('wrong_audio', 'N/A'):>18s}"
            )

    lines.append("\n" + "=" * 90)
    text = "\n".join(lines)
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="MSCI vs cMSCI Comparison")
    parser.add_argument("--rq1-path", default="runs/rq1/rq1_results.json")
    parser.add_argument("--out-dir", default="runs/cmsci_comparison")
    parser.add_argument("--calibrate", action="store_true", default=True,
                        help="Fit calibration from RQ1 baseline data (default: True)")
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--all", action="store_true",
                        help="Run comparison on all known result files")
    args = parser.parse_args()

    if args.all:
        print("=" * 90)
        print("MSCI vs cMSCI COMPARISON — ALL RESULT FILES")
        print("=" * 90)

        for label, rq1_path, out_dir in ALL_RESULT_FILES:
            print(f"\n{'─'*90}")
            print(f"  [{label}] Input: {rq1_path}")
            print(f"  [{label}] Output: {out_dir}")
            n = run_single(rq1_path, out_dir, calibrate=args.calibrate)
            if n > 0:
                print(f"  [{label}] Processed {n} results")

        print_cross_run_summary(None)
        return 0

    if not Path(args.rq1_path).exists():
        print(f"ERROR: RQ1 results not found: {args.rq1_path}")
        print("Run experiments first: python scripts/run_rq1.py --skip-text")
        return 1

    print("=" * 90)
    print("MSCI vs cMSCI COMPARISON")
    print("=" * 90)
    print(f"  Input:  {args.rq1_path}")
    print(f"  Output: {args.out_dir}")

    run_single(args.rq1_path, args.out_dir, calibrate=args.calibrate)

    return 0


if __name__ == "__main__":
    sys.exit(main())
