#!/usr/bin/env python3
"""
Full Evaluation on Test Set + Benchmarks (Task 3.4).

Evaluates ALL methods on the held-out test set (10 human-rated samples):
- cMSCI (Variant F)
- MSCI (Variant A, legacy)
- CLIPScore
- BLIPScore (if available)
- ImageBind (if available)
- CCA / RegCCA
- VLM-as-judge (if Ollama running)
- Simple baselines (raw cosine, cosine+z, retrieval rank, concat)

Creates comprehensive comparison table.

Usage:
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --all-samples   # Evaluate on all 30 (not just test set)
    python scripts/run_full_evaluation.py --skip-vlm      # Skip VLM judge
    python scripts/run_full_evaluation.py --skip-blip     # Skip BLIP (slow)
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

from src.config.settings import (
    RQ3_SAMPLES_PATH, RQ3_SAMPLES_EXTENDED_PATH,
)

# Use extended samples if available, otherwise fall back to original
SAMPLES_PATH = RQ3_SAMPLES_EXTENDED_PATH if RQ3_SAMPLES_EXTENDED_PATH.exists() else RQ3_SAMPLES_PATH
OUTPUT_DIR = PROJECT_ROOT / "runs" / "full_evaluation"


def load_data(use_all: bool = False):
    """Load samples and human scores with dev/test split."""
    from scripts.optimize_cmsci import load_human_scores

    with open(SAMPLES_PATH) as f:
        all_samples = json.load(f)["samples"]

    human_scores = load_human_scores()

    if use_all:
        eval_samples = all_samples
        dev_samples = all_samples
        print(f"  Using ALL {len(all_samples)} samples")
    else:
        from src.experiments.data_splits import get_dev_test_split
        dev_ids, test_ids = get_dev_test_split()
        dev_set = set(dev_ids)
        test_set = set(test_ids)
        eval_samples = [s for s in all_samples if s["sample_id"] in test_set]
        dev_samples = [s for s in all_samples if s["sample_id"] in dev_set]
        print(f"  Dev: {len(dev_samples)}, Test: {len(eval_samples)}")

    return all_samples, dev_samples, eval_samples, human_scores


def run_cmsci(samples: list, human_scores: dict) -> dict:
    """Run cMSCI (Variant F) and legacy MSCI (Variant A), plus ablation."""
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
    msci_scores = []
    human_vals = []

    # Collect all variant scores for ablation
    variant_keys = [
        ("A_msci", "MSCI"),
        ("B_gram", "Gramian"),
        ("C_gram_znorm", "z-score"),
        ("D_gram_znorm_contrastive", "contrastive"),
        ("E_gram_znorm_contrastive_exmcr", "Ex-MCR"),
        ("F_full_cmsci", "full_cMSCI"),
    ]
    variant_scores = {key: [] for key, _ in variant_keys}

    for s in samples:
        sid = s["sample_id"]
        if sid not in human_scores:
            continue

        result = engine.evaluate(
            text=s["prompt_text"],
            image_path=s.get("image_path"),
            audio_path=s.get("audio_path"),
            domain=s.get("domain", ""),
        )

        cmsci = result["variant_scores"]["F_full_cmsci"] or result["cmsci"]
        msci = result["variant_scores"]["A_msci"] or result["msci"]

        cmsci_scores.append(cmsci)
        msci_scores.append(msci)
        human_vals.append(human_scores[sid]["weighted_score"]["mean"])

        for key, _ in variant_keys:
            val = result["variant_scores"].get(key)
            variant_scores[key].append(val)

    results = {}
    if len(cmsci_scores) >= 5:
        rho_c, p_c = sp_stats.spearmanr(cmsci_scores, human_vals)
        results["cMSCI"] = {"rho": float(rho_c), "p": float(p_c), "n": len(cmsci_scores)}
        rho_m, p_m = sp_stats.spearmanr(msci_scores, human_vals)
        results["MSCI"] = {"rho": float(rho_m), "p": float(p_m), "n": len(msci_scores)}

        # Compute ablation — correlation for each variant
        ablation = []
        for key, label in variant_keys:
            vals = variant_scores[key]
            valid = [(v, h) for v, h in zip(vals, human_vals) if v is not None]
            if len(valid) >= 5:
                v_arr = [x[0] for x in valid]
                h_arr = [x[1] for x in valid]
                rho, p = sp_stats.spearmanr(v_arr, h_arr)
                ablation.append({
                    "variant": key, "label": label,
                    "rho": round(float(rho), 3), "p": round(float(p), 6),
                    "significant": p < 0.05, "n": len(valid),
                })
                print(f"    Ablation {label:15s}: rho={float(rho):.3f} (p={float(p):.4f})")
        results["ablation"] = ablation

    return results


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation")
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-blip", action="store_true")
    parser.add_argument("--skip-imagebind", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("FULL EVALUATION — All Methods vs Human Ratings")
    print("=" * 70)

    all_samples, dev_samples, eval_samples, human_scores = load_data(args.all_samples)
    all_results = {}

    # 1. cMSCI and MSCI
    print("\n--- cMSCI + MSCI ---")
    t0 = time.time()
    cmsci_results = run_cmsci(eval_samples, human_scores)
    all_results.update(cmsci_results)
    print(f"  cMSCI: rho={cmsci_results.get('cMSCI', {}).get('rho', 'N/A'):.4f} [{time.time()-t0:.1f}s]")
    print(f"  MSCI:  rho={cmsci_results.get('MSCI', {}).get('rho', 'N/A'):.4f}")

    # 2. Simple baselines
    print("\n--- Simple Baselines ---")
    t0 = time.time()
    from src.baselines.simple_baselines import SimpleBaselines
    from src.config.settings import CMSCI_CALIBRATION_PATH
    simple = SimpleBaselines(
        calibration_path=str(CMSCI_CALIBRATION_PATH) if CMSCI_CALIBRATION_PATH.exists() else None,
    )
    simple_results = simple.evaluate_all(eval_samples, human_scores)
    if "correlations" in simple_results:
        for method, corr in simple_results["correlations"].items():
            all_results[method] = corr
            sig = "*" if corr.get("significant") else ""
            print(f"  {method:25s}: rho={corr['rho']:.4f} (p={corr['p']:.4f}){sig}")
    print(f"  [{time.time()-t0:.1f}s]")

    # 3. CLIPScore
    print("\n--- CLIPScore ---")
    t0 = time.time()
    try:
        from src.baselines.established_baselines import evaluate_established_baselines
        methods = ["CLIPScore"]
        if not args.skip_blip:
            methods.append("BLIPScore")
        established_results = evaluate_established_baselines(eval_samples, human_scores, methods=methods)
        if "correlations" in established_results:
            for method, corr in established_results["correlations"].items():
                all_results[method] = corr
                sig = "*" if corr.get("significant") else ""
                print(f"  {method:25s}: rho={corr['rho']:.4f} (p={corr['p']:.4f}){sig}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"  [{time.time()-t0:.1f}s]")

    # 4. CCA baselines
    print("\n--- CCA Baselines ---")
    t0 = time.time()
    try:
        from src.baselines.joint_baselines import evaluate_joint_baselines
        # When using all samples, fit CCA with leave-one-out (no exclusion)
        # When using dev/test split, exclude test set from fitting
        test_ids = None if args.all_samples else {s["sample_id"] for s in eval_samples}
        joint_results = evaluate_joint_baselines(all_samples, human_scores, test_ids=test_ids)
        if "correlations" in joint_results:
            for method, corr in joint_results["correlations"].items():
                all_results[method] = corr
                sig = "*" if corr.get("significant") else ""
                print(f"  {method:25s}: rho={corr['rho']:.4f} (p={corr['p']:.4f}){sig}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print(f"  [{time.time()-t0:.1f}s]")

    # 5. VLM judge (optional)
    if not args.skip_vlm:
        print("\n--- VLM-as-Judge ---")
        t0 = time.time()
        try:
            from src.baselines.vlm_judge import evaluate_vlm_judge
            vlm_results = evaluate_vlm_judge(eval_samples, human_scores)
            if "correlations" in vlm_results:
                for method, corr in vlm_results["correlations"].items():
                    all_results[method] = corr
                    sig = "*" if corr.get("significant") else ""
                    print(f"  {method:25s}: rho={corr['rho']:.4f} (p={corr['p']:.4f}){sig}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print(f"  [{time.time()-t0:.1f}s]")

    # 6. Gemini baselines + cMSCI v2 (optional)
    print("\n--- Gemini / cMSCI v2 ---")
    try:
        from src.config.settings import GEMINI_API_KEY
        if GEMINI_API_KEY:
            # cMSCI v2
            t0 = time.time()
            from src.coherence.cmsci_engine_v2 import CalibratedCoherenceEngineV2
            v2_engine = CalibratedCoherenceEngineV2(negative_bank_enabled=True)
            v2_scores = []
            v2_humans = []
            for s in eval_samples:
                sid = s["sample_id"]
                if sid not in human_scores:
                    continue
                r = v2_engine.evaluate(
                    text=s["prompt_text"],
                    image_path=s.get("image_path"),
                    audio_path=s.get("audio_path"),
                    domain=s.get("domain", ""),
                )
                v2_scores.append(r["cmsci_v2"])
                v2_humans.append(human_scores[sid]["weighted_score"]["mean"])
            if len(v2_scores) >= 5:
                rho, p = sp_stats.spearmanr(v2_scores, v2_humans)
                all_results["cMSCI_v2_Gemini"] = {
                    "rho": float(rho), "p": float(p), "n": len(v2_scores),
                    "significant": p < 0.05,
                }
                sig = "*" if p < 0.05 else ""
                print(f"  {'cMSCI_v2_Gemini':25s}: rho={rho:.4f} (p={p:.4f}){sig}")

            # Gemini cosine baselines
            from src.baselines.gemini_baseline import GeminiCosineBaseline
            gbl = GeminiCosineBaseline()
            gbl_results = gbl.evaluate_all(eval_samples, human_scores)
            for method, corr in gbl_results.items():
                label = f"Gemini_cosine_{method}"
                all_results[label] = corr
                if corr.get("rho") is not None:
                    sig = "*" if corr.get("sig") else ""
                    print(f"  {label:25s}: rho={corr['rho']:.4f} (p={corr['p']:.4f}){sig}")

            print(f"  [{time.time()-t0:.1f}s]")
        else:
            print("  Skipped (GOOGLE_API_KEY not set)")
    except ImportError:
        print("  Skipped (google-genai not installed)")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary Table
    print(f"\n{'='*70}")
    print("COMPREHENSIVE COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Method':25s}  {'rho':>7s}  {'p-value':>10s}  {'n':>3s}  {'Sig':>3s}")
    print(f"  {'-'*25}  {'-------':>7s}  {'----------':>10s}  {'---':>3s}  {'---':>3s}")

    # Sort by rho (exclude non-dict entries like ablation)
    method_results = {k: v for k, v in all_results.items() if isinstance(v, dict)}
    sorted_methods = sorted(method_results.items(), key=lambda x: x[1].get("rho", -999), reverse=True)
    for method, corr in sorted_methods:
        rho = corr.get("rho", float("nan"))
        p = corr.get("p", float("nan"))
        n = corr.get("n", 0)
        sig = "*" if corr.get("significant") or (p < 0.05) else ""
        print(f"  {method:25s}  {rho:7.4f}  {p:10.6f}  {n:3d}  {sig:>3s}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "full_evaluation",
        "n_eval_samples": len(eval_samples),
        "use_all": args.all_samples,
        "results": {k: v for k, v in all_results.items()},
    }
    output_path = OUTPUT_DIR / "full_evaluation.json"

    # Convert numpy types to native Python for JSON serialization
    def _to_native(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(_to_native(output), f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
