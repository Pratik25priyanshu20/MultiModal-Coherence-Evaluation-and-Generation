#!/usr/bin/env python3
"""
Run ALL baselines with 5-rater human scores for the journal paper.

Computes Spearman rho, p-value, and 95% bootstrap CI for each method.
Saves results to runs/use_case_experiments/full_baselines_5raters.json

Usage:
    python scripts/run_full_baselines_5raters.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import RQ3_SAMPLES_PATH, RQ3_SAMPLES_EXTENDED_PATH

SAMPLES_PATH = RQ3_SAMPLES_EXTENDED_PATH if RQ3_SAMPLES_EXTENDED_PATH.exists() else RQ3_SAMPLES_PATH
CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "cmsci_calibration.json"
OUTPUT_PATH = PROJECT_ROOT / "runs" / "use_case_experiments" / "full_baselines_5raters.json"

N_BOOTSTRAP = 10000
ALPHA = 0.05
RNG = np.random.RandomState(42)


def load_human_scores():
    """Load 5-rater human scores."""
    from scripts.optimize_cmsci import load_human_scores as _load
    return _load()


def load_samples():
    """Load 30 RQ3 samples."""
    with open(RQ3_SAMPLES_PATH) as f:
        data = json.load(f)
    # Handle both formats: list or dict with "samples" key
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError(f"Unexpected samples format: {type(data)}")
    # Keep only 30 original samples
    return samples[:30] if len(samples) > 30 else samples


def bootstrap_ci(scores, humans, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """Compute bootstrap 95% CI for Spearman rho."""
    n = len(scores)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.randint(0, n, size=n)
        s = [scores[j] for j in idx]
        h = [humans[j] for j in idx]
        # Handle constant arrays
        if len(set(s)) < 2 or len(set(h)) < 2:
            rhos[i] = 0.0
        else:
            rhos[i], _ = sp_stats.spearmanr(s, h)
    lo = float(np.percentile(rhos, 100 * alpha / 2))
    hi = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_correlation(method_scores, human_scores_dict, samples):
    """Compute Spearman rho + bootstrap CI for a method.

    method_scores: dict mapping sample_id -> float score
    human_scores_dict: dict mapping sample_id -> aggregated human score dict
    samples: list of sample dicts
    """
    scores = []
    humans = []
    for s in samples:
        sid = s["sample_id"]
        if sid in method_scores and sid in human_scores_dict:
            sc = method_scores[sid]
            if sc is None:
                continue
            scores.append(sc)
            humans.append(human_scores_dict[sid]["weighted_score"]["mean"])

    if len(scores) < 5:
        return {"rho": None, "p": None, "ci_lo": None, "ci_hi": None,
                "n": len(scores), "significant": False, "error": "too_few_samples"}

    rho, p = sp_stats.spearmanr(scores, humans)
    ci_lo, ci_hi = bootstrap_ci(scores, humans)
    return {
        "rho": round(float(rho), 4),
        "p": round(float(p), 6),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "n": len(scores),
        "significant": p < ALPHA,
    }


def run_simple_baselines(samples, human_scores):
    """Run raw_cosine, cosine_znorm, retrieval_rank, concatenated_cosine."""
    print("\n=== Simple Baselines ===")
    results = {}
    try:
        from src.baselines.simple_baselines import SimpleBaselines
        cal_path = str(CALIBRATION_PATH) if CALIBRATION_PATH.exists() else None
        bl = SimpleBaselines(calibration_path=cal_path)

        method_scores = {
            "raw_cosine": {},
            "cosine_znorm": {},
            "retrieval_rank": {},
            "concatenated_cosine": {},
        }

        for i, s in enumerate(samples):
            sid = s["sample_id"]
            print(f"  [{i+1}/{len(samples)}] {sid}", end="\r")
            try:
                res = bl.evaluate_sample(
                    text=s["prompt_text"],
                    image_path=s.get("image_path", ""),
                    audio_path=s.get("audio_path", ""),
                    domain=s.get("domain", ""),
                )
                for method_name, r in res.items():
                    if r.get("score") is not None:
                        method_scores[method_name][sid] = r["score"]
            except Exception as e:
                print(f"\n  ERROR on {sid}: {e}")
        print()

        for method_name, scores_dict in method_scores.items():
            corr = compute_correlation(scores_dict, human_scores, samples)
            corr["method"] = method_name
            corr["scores"] = {k: round(v, 4) for k, v in scores_dict.items()}
            results[method_name] = corr
            sig = "*" if corr.get("significant") else ""
            print(f"  {method_name}: rho={corr.get('rho')}, p={corr.get('p')}, "
                  f"CI=[{corr.get('ci_lo')}, {corr.get('ci_hi')}] {sig}")

    except Exception as e:
        print(f"  Simple baselines FAILED: {e}")
        import traceback; traceback.print_exc()

    return results


def run_joint_baselines(samples, human_scores):
    """Run CCA and Regularized CCA."""
    print("\n=== Joint (CCA) Baselines ===")
    results = {}
    try:
        from src.baselines.joint_baselines import CCABaseline, RegularizedCCABaseline

        for name, cls, kwargs in [
            ("CCA", CCABaseline, {"n_components": 10}),
            ("RegCCA", RegularizedCCABaseline, {"n_components": 10, "alpha": 1.0}),
        ]:
            print(f"  Fitting {name}...")
            bl = cls(**kwargs)
            # Fit on all 30 samples (no separate test set — we report LOO or full)
            bl.fit(samples)

            scores_dict = {}
            for i, s in enumerate(samples):
                sid = s["sample_id"]
                try:
                    r = bl.score(
                        text=s["prompt_text"],
                        image_path=s.get("image_path", ""),
                        audio_path=s.get("audio_path", ""),
                    )
                    if r.get("score") is not None:
                        scores_dict[sid] = r["score"]
                except Exception as e:
                    print(f"\n  ERROR on {sid}: {e}")

            corr = compute_correlation(scores_dict, human_scores, samples)
            corr["method"] = name
            corr["scores"] = {k: round(v, 4) for k, v in scores_dict.items()}
            results[name] = corr
            sig = "*" if corr.get("significant") else ""
            print(f"  {name}: rho={corr.get('rho')}, p={corr.get('p')}, "
                  f"CI=[{corr.get('ci_lo')}, {corr.get('ci_hi')}] {sig}")

    except Exception as e:
        print(f"  Joint baselines FAILED: {e}")
        import traceback; traceback.print_exc()

    return results


def run_blipscore(samples, human_scores):
    """Run BLIPScore + CLAPScore."""
    print("\n=== BLIPScore + CLAPScore ===")
    results = {}
    try:
        from src.baselines.established_baselines import BLIPScoreBaseline
        bl = BLIPScoreBaseline()

        scores_dict = {}
        for i, s in enumerate(samples):
            sid = s["sample_id"]
            print(f"  [{i+1}/{len(samples)}] {sid}", end="\r")
            try:
                r = bl.score(
                    text=s["prompt_text"],
                    image_path=s.get("image_path", ""),
                    audio_path=s.get("audio_path", ""),
                )
                if r.get("score") is not None:
                    scores_dict[sid] = r["score"]
            except Exception as e:
                print(f"\n  ERROR on {sid}: {e}")
        print()

        corr = compute_correlation(scores_dict, human_scores, samples)
        corr["method"] = "BLIPScore+CLAPScore"
        corr["scores"] = {k: round(v, 4) for k, v in scores_dict.items()}
        results["BLIPScore+CLAPScore"] = corr
        sig = "*" if corr.get("significant") else ""
        print(f"  BLIPScore+CLAPScore: rho={corr.get('rho')}, p={corr.get('p')}, "
              f"CI=[{corr.get('ci_lo')}, {corr.get('ci_hi')}] {sig}")

    except Exception as e:
        print(f"  BLIPScore NOT AVAILABLE: {e}")
        results["BLIPScore+CLAPScore"] = {
            "method": "BLIPScore+CLAPScore",
            "rho": None, "p": None, "ci_lo": None, "ci_hi": None,
            "n": 0, "significant": False, "error": f"not_available: {e}",
        }

    return results


def run_vlm_judge(samples, human_scores):
    """Run LLaVA-7B VLM-as-Judge via Ollama (if available)."""
    print("\n=== VLM-as-Judge (LLaVA via Ollama) ===")
    results = {}
    try:
        import subprocess
        # Check if Ollama is running
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError("Ollama not running")

        from src.baselines.vlm_judge import VLMJudge
        judge = VLMJudge(model="llava:7b")

        scores_dict = {}
        for i, s in enumerate(samples):
            sid = s["sample_id"]
            print(f"  [{i+1}/{len(samples)}] {sid}", end="\r")
            try:
                r = judge.score(
                    text=s["prompt_text"],
                    image_path=s.get("image_path", ""),
                    audio_path=s.get("audio_path", ""),
                )
                if r.get("score") is not None:
                    scores_dict[sid] = r["score"]
            except Exception as e:
                print(f"\n  ERROR on {sid}: {e}")
        print()

        corr = compute_correlation(scores_dict, human_scores, samples)
        corr["method"] = "LLaVA-7B"
        corr["scores"] = {k: round(v, 4) for k, v in scores_dict.items()}
        results["LLaVA-7B"] = corr
        sig = "*" if corr.get("significant") else ""
        print(f"  LLaVA-7B: rho={corr.get('rho')}, p={corr.get('p')}, "
              f"CI=[{corr.get('ci_lo')}, {corr.get('ci_hi')}] {sig}")

    except Exception as e:
        print(f"  VLM-as-Judge NOT AVAILABLE: {e}")
        results["LLaVA-7B"] = {
            "method": "LLaVA-7B",
            "rho": None, "p": None, "ci_lo": None, "ci_hi": None,
            "n": 0, "significant": False, "error": f"not_available: {e}",
        }

    return results


def compute_existing_method_cis(human_scores, samples):
    """Compute bootstrap 95% CIs for the existing methods we already have rho for.

    These methods already have scores computed elsewhere — we need to
    recompute or load them and compute CIs.
    """
    print("\n=== Existing Methods (CIs only) ===")
    results = {}

    # Pre-existing methods with known rho values.
    # We compute bootstrap CIs here using their actual scores.
    existing = {
        "cMSCI_v3_ensemble": 0.628,
        "cMSCI_v1": 0.581,
        "cMSCI_v2": 0.467,
        "CLIPScore+CLAPScore": 0.452,
        "Cosine+z-norm": 0.445,
        "MSCI_raw_cosine": 0.298,
    }

    # We can't recompute cMSCI v1/v2/v3 scores here without loading their engines.
    # Instead, we'll report the known rho and compute CIs via Fisher z-transform.
    # For a sample of n=30, the Fisher z CI is: z +/- 1.96/sqrt(n-3)
    for name, rho in existing.items():
        n = 30
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(n - 3)
        z_lo = z - 1.96 * se
        z_hi = z + 1.96 * se
        ci_lo = float(np.tanh(z_lo))
        ci_hi = float(np.tanh(z_hi))

        results[name] = {
            "method": name,
            "rho": rho,
            "p": None,  # Already known significant or not
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "n": n,
            "note": "CI via Fisher z-transform (scores not recomputed here)",
        }
        print(f"  {name}: rho={rho}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    return results


def main():
    start = time.time()

    # Load data
    print("Loading human scores (5 raters)...")
    human_scores = load_human_scores()
    print(f"  {len(human_scores)} samples with human scores")

    print("Loading samples...")
    samples = load_samples()
    print(f"  {len(samples)} samples loaded")

    # Keep only samples that have human scores
    sample_ids_with_scores = set(human_scores.keys())
    samples = [s for s in samples if s["sample_id"] in sample_ids_with_scores]
    print(f"  {len(samples)} samples with both method+human scores")

    all_results = {}

    # 1. Simple baselines (raw cosine, z-norm, retrieval rank, concat)
    simple = run_simple_baselines(samples, human_scores)
    all_results.update(simple)

    # 2. Joint baselines (CCA, RegCCA)
    joint = run_joint_baselines(samples, human_scores)
    all_results.update(joint)

    # 3. BLIPScore + CLAPScore
    blip = run_blipscore(samples, human_scores)
    all_results.update(blip)

    # 4. VLM-as-Judge
    vlm = run_vlm_judge(samples, human_scores)
    all_results.update(vlm)

    # 5. Existing methods CIs
    existing = compute_existing_method_cis(human_scores, samples)
    all_results.update(existing)

    # --- Final ranked table ---
    print("\n" + "=" * 80)
    print("FINAL RANKED TABLE — All Methods vs 5-Rater Human Scores")
    print("=" * 80)

    # Sort by rho (descending), None last
    ranked = sorted(
        all_results.values(),
        key=lambda x: x.get("rho") if x.get("rho") is not None else -999,
        reverse=True,
    )

    print(f"{'Rank':<5} {'Method':<25} {'rho':>7} {'p':>10} {'95% CI':>18} {'Sig':>5} {'n':>4}")
    print("-" * 80)
    for i, r in enumerate(ranked, 1):
        rho = r.get("rho")
        p = r.get("p")
        ci_lo = r.get("ci_lo")
        ci_hi = r.get("ci_hi")
        sig = r.get("significant", "")
        n = r.get("n", "")
        method = r.get("method", "unknown")

        rho_s = f"{rho:.4f}" if rho is not None else "N/A"
        p_s = f"{p:.6f}" if p is not None else "N/A"
        ci_s = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None else "N/A"
        sig_s = "*" if sig else ""
        n_s = str(n) if n else ""

        err = r.get("error", "")
        if err and rho is None:
            ci_s = f"({err[:30]})"

        print(f"{i:<5} {method:<25} {rho_s:>7} {p_s:>10} {ci_s:>18} {sig_s:>5} {n_s:>4}")

    print("=" * 80)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Strip per-sample scores for cleaner output (optional — keep them for reproducibility)
    save_data = {
        "description": "Full baselines comparison with 5-rater human scores",
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": {},
        "ranked": [],
    }

    for r in ranked:
        method = r.get("method", "unknown")
        entry = {k: v for k, v in r.items() if k != "scores"}
        save_data["methods"][method] = entry  # without per-sample scores for clean output
        save_data["ranked"].append(entry)

    # Custom serializer for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(save_data, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
