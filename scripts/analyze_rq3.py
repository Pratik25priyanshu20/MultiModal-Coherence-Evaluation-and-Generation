#!/usr/bin/env python3
"""
RQ3 Analysis — Does MSCI correlate with human perception?

Aggregates multi-rater evaluation sessions and computes:
- Inter-rater reliability (Krippendorff's alpha)
- Intra-rater reliability per evaluator (Cohen's kappa)
- Spearman correlation between mean human score and MSCI
- Spearman correlation between mean human score and cMSCI (Variant D)
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

from src.coherence.cmsci_engine import CalibratedCoherenceEngine
from src.config.settings import (
    BRIDGE_WEIGHTS_PATH,
    CMSCI_CALIBRATION_PATH,
    EXMCR_WEIGHTS_PATH,
    PROB_CLIP_ADAPTER_PATH,
    PROB_CLAP_ADAPTER_PATH,
    RQ3_SAMPLES_PATH,
    RQ3_SAMPLES_EXTENDED_PATH,
    RQ3_SESSIONS_DIR,
)

# Use extended samples if available, otherwise fall back to original
SAMPLES_PATH = RQ3_SAMPLES_EXTENDED_PATH if RQ3_SAMPLES_EXTENDED_PATH.exists() else RQ3_SAMPLES_PATH
SESSION_DIR = RQ3_SESSIONS_DIR
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


def compute_cmsci_scores(samples: list) -> dict:
    """Evaluate all RQ3 samples with CalibratedCoherenceEngine (all available models)."""
    cal_path = str(CMSCI_CALIBRATION_PATH) if CMSCI_CALIBRATION_PATH.exists() else None
    engine = CalibratedCoherenceEngine(
        calibration_path=cal_path,
        exmcr_weights_path=str(EXMCR_WEIGHTS_PATH) if EXMCR_WEIGHTS_PATH.exists() else None,
        bridge_path=str(BRIDGE_WEIGHTS_PATH) if BRIDGE_WEIGHTS_PATH.exists() else None,
        prob_clip_adapter_path=str(PROB_CLIP_ADAPTER_PATH) if PROB_CLIP_ADAPTER_PATH.exists() else None,
        prob_clap_adapter_path=str(PROB_CLAP_ADAPTER_PATH) if PROB_CLAP_ADAPTER_PATH.exists() else None,
    )

    scores = {}
    for s in samples:
        try:
            result = engine.evaluate(
                text=s["prompt_text"],
                image_path=s.get("image_path"),
                audio_path=s.get("audio_path"),
                domain=s.get("domain", ""),
            )
            scores[s["sample_id"]] = result
        except Exception as e:
            print(f"  cMSCI error for {s['sample_id']}: {e}")
    return scores


def compute_cmsci_human_correlation(
    sessions, cmsci_scores: dict
) -> dict:
    """Compute Spearman correlation between cMSCI and aggregated human scores."""
    aggregated = aggregate_multi_rater_sessions(sessions)

    human_vals = []
    cmsci_vals = []

    for sample_id, agg in aggregated.items():
        if sample_id in cmsci_scores and cmsci_scores[sample_id].get("cmsci") is not None:
            human_vals.append(agg["weighted_score"]["mean"])
            cmsci_vals.append(cmsci_scores[sample_id]["cmsci"])

    if len(human_vals) < 5:
        return {"error": "Too few paired samples", "n_paired": len(human_vals)}

    human_arr = np.array(human_vals)
    cmsci_arr = np.array(cmsci_vals)

    spearman = stats.spearmanr(cmsci_arr, human_arr)
    pearson = stats.pearsonr(cmsci_arr, human_arr)

    # Bootstrap 95% CI for Spearman rho
    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(10000):
        idx = rng.choice(len(human_arr), size=len(human_arr), replace=True)
        r, _ = stats.spearmanr(cmsci_arr[idx], human_arr[idx])
        boot_rhos.append(r)
    ci_lower = float(np.percentile(boot_rhos, 2.5))
    ci_upper = float(np.percentile(boot_rhos, 97.5))

    return {
        "n_paired": len(human_vals),
        "spearman_rho": round(float(spearman.correlation), 4),
        "spearman_p": float(spearman.pvalue),
        "spearman_95ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "pearson_r": round(float(pearson.statistic), 4),
        "pearson_p": float(pearson.pvalue),
    }


def compute_variant_correlations(
    sessions, cmsci_scores: dict
) -> dict:
    """Compute Spearman correlation for each cMSCI variant against human scores."""
    aggregated = aggregate_multi_rater_sessions(sessions)

    variant_keys = [
        "A_msci", "B_gram", "C_gram_znorm",
        "D_gram_znorm_contrastive", "E_gram_znorm_contrastive_exmcr",
        "F_full_cmsci",
    ]
    results = {}
    for vk in variant_keys:
        human_vals = []
        variant_vals = []
        for sample_id, agg in aggregated.items():
            if sample_id not in cmsci_scores:
                continue
            vs = cmsci_scores[sample_id].get("variant_scores", {})
            val = vs.get(vk)
            if val is not None:
                human_vals.append(agg["weighted_score"]["mean"])
                variant_vals.append(val)

        if len(human_vals) < 5:
            results[vk] = {"n_paired": len(human_vals), "error": "Too few samples"}
            continue

        rho, p = stats.spearmanr(np.array(variant_vals), np.array(human_vals))
        results[vk] = {
            "n_paired": len(human_vals),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p),
        }
    return results


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

    # --- 4. cMSCI vs Human correlation ---
    print("\n--- cMSCI vs Human Score Correlation ---")
    with open(SAMPLES_PATH) as f:
        samples_list = json.load(f)["samples"]
    print(f"  Computing cMSCI (Variant D) for {len(samples_list)} samples...")
    cmsci_scores = compute_cmsci_scores(samples_list)
    print(f"  Scored {len(cmsci_scores)}/{len(samples_list)} samples")

    # Active variant distribution
    active_variants = [v.get("active_variant", "?") for v in cmsci_scores.values()]
    from collections import Counter
    vc = Counter(active_variants)
    print(f"  Active variants: {dict(vc)}")

    cmsci_correlation = compute_cmsci_human_correlation(sessions, cmsci_scores)
    if "error" not in cmsci_correlation:
        print(f"  N paired:      {cmsci_correlation['n_paired']}")
        print(f"  Spearman rho:  {cmsci_correlation['spearman_rho']:.4f}")
        print(f"  Spearman p:    {cmsci_correlation['spearman_p']:.6f}")
        print(f"  95% CI:        [{cmsci_correlation['spearman_95ci'][0]:.4f}, "
              f"{cmsci_correlation['spearman_95ci'][1]:.4f}]")
        print(f"  Pearson r:     {cmsci_correlation['pearson_r']:.4f}")
        print(f"  Pearson p:     {cmsci_correlation['pearson_p']:.6f}")
    else:
        print(f"  {cmsci_correlation['error']}")

    # Side-by-side comparison
    print("\n--- MSCI vs cMSCI — Side-by-Side Human Correlation ---")
    msci_rho = correlation.get("spearman_rho", "N/A")
    msci_p = correlation.get("spearman_p", "N/A")
    msci_ci = correlation.get("spearman_95ci", ["N/A", "N/A"])
    cmsci_rho = cmsci_correlation.get("spearman_rho", "N/A")
    cmsci_p = cmsci_correlation.get("spearman_p", "N/A")
    cmsci_ci = cmsci_correlation.get("spearman_95ci", ["N/A", "N/A"])
    print(f"  {'Metric':10s} {'rho':>8s} {'p-value':>12s} {'95% CI':>20s}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*20}")
    if isinstance(msci_rho, float):
        print(f"  {'MSCI':10s} {msci_rho:8.4f} {msci_p:12.6f} [{msci_ci[0]:.4f}, {msci_ci[1]:.4f}]")
    if isinstance(cmsci_rho, float):
        print(f"  {'cMSCI':10s} {cmsci_rho:8.4f} {cmsci_p:12.6f} [{cmsci_ci[0]:.4f}, {cmsci_ci[1]:.4f}]")

    # Per-variant correlation
    print("\n--- Per-Variant Human Correlation ---")
    variant_corrs = compute_variant_correlations(sessions, cmsci_scores)
    print(f"  {'Variant':35s} {'N':>3s} {'rho':>8s} {'p':>10s}")
    print(f"  {'-'*35} {'-'*3} {'-'*8} {'-'*10}")
    for vk, vc_data in variant_corrs.items():
        if "error" in vc_data:
            print(f"  {vk:35s} {vc_data['n_paired']:3d}   (insufficient data)")
        else:
            sig = "*" if vc_data["spearman_p"] < 0.05 else ""
            print(f"  {vk:35s} {vc_data['n_paired']:3d} {vc_data['spearman_rho']:8.4f} "
                  f"{vc_data['spearman_p']:10.6f}{sig}")

    # --- 5. Per-condition breakdown ---
    print("\n--- Per-Condition Analysis ---")
    by_condition = analyze_by_condition_multi(sessions, sample_msci, sample_conditions)
    print(f"  {'Condition':15s} {'N':>3s} {'Human':>8s} {'MSCI':>8s}")
    print(f"  {'-'*15} {'-'*3} {'-'*8} {'-'*8}")
    for cond, data in sorted(by_condition.items()):
        print(f"  {cond:15s} {data['n']:3d} {data['human_weighted_mean']:8.4f} "
              f"{data['msci_mean']:8.4f}")

    # --- 6. Verdict ---
    verdict_msci = determine_verdict(correlation)
    verdict_cmsci = determine_verdict(cmsci_correlation) if "error" not in cmsci_correlation else "INCONCLUSIVE"
    print(f"\n{'='*60}")
    print(f"RQ3 VERDICT (MSCI):  {verdict_msci}")
    print(f"RQ3 VERDICT (cMSCI): {verdict_cmsci}")
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
        "cmsci_correlation": cmsci_correlation,
        "variant_correlations": variant_corrs,
        "by_condition": by_condition,
        "verdict_msci": verdict_msci,
        "verdict_cmsci": verdict_cmsci,
        "verdict": verdict_msci,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    print(f"\nFull report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
