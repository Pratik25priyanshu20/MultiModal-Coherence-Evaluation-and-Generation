#!/usr/bin/env python3
"""
Head-to-Head Comparison: cMSCI v1 (CLIP+CLAP) vs cMSCI v2 (Gemini).

Produces the final paper table comparing all methods across both embedding
backends, plus v2 ablation table and bootstrap confidence intervals.

Output table:
    Method                      rho      p        Sig    Space
    ────────────────────────────────────────────────────────────
    cMSCI v2 (Gemini)          X.XXX    X.XXX     ?     Unified
    cMSCI v1 (CLIP+CLAP)       0.519    0.003     *     Dual
    Gemini cosine (3-ch)        X.XXX    X.XXX     ?     Unified
    CCA                         0.409    0.025     *     Dual
    ...

Usage:
    python scripts/run_gemini_comparison.py
    python scripts/run_gemini_comparison.py --all-samples
    python scripts/run_gemini_comparison.py --bootstrap 1000
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
    RQ3_SAMPLES_PATH, RQ3_SAMPLES_EXTENDED_PATH, RQ3_SESSIONS_DIR,
)

# Use extended samples if available, otherwise fall back to original
SAMPLES_PATH = RQ3_SAMPLES_EXTENDED_PATH if RQ3_SAMPLES_EXTENDED_PATH.exists() else RQ3_SAMPLES_PATH
SESSION_DIR = RQ3_SESSIONS_DIR
OUTPUT_DIR = PROJECT_ROOT / "runs" / "gemini_comparison"


def load_human_scores() -> dict:
    """Load human evaluation scores."""
    from scripts.optimize_cmsci import load_human_scores as _load
    return _load()


def load_samples(use_all: bool = False) -> tuple[list, set]:
    """Load samples, optionally filtered to dev/test."""
    with open(SAMPLES_PATH) as f:
        all_samples = json.load(f)["samples"]

    if use_all:
        return all_samples, set(s["sample_id"] for s in all_samples)

    split_path = PROJECT_ROOT / "artifacts" / "dev_test_split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        ids = set(split.get("dev", []) + split.get("test", []))
    else:
        ids = set(s["sample_id"] for s in all_samples)

    return all_samples, ids


def run_cmsci_v1(samples: list, human_scores: dict) -> dict:
    """Run cMSCI v1 (CLIP+CLAP) and return correlation."""
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

    scores = []
    humans = []
    variant_scores = {k: [] for k in ["A", "B", "C", "D", "E", "F"]}

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
        scores.append(result["cmsci"])
        humans.append(human_scores[sid]["weighted_score"]["mean"])

        vs = result["variant_scores"]
        variant_scores["A"].append(vs["A_msci"])
        variant_scores["B"].append(vs["B_gram"])
        variant_scores["C"].append(vs["C_gram_znorm"])
        variant_scores["D"].append(vs["D_gram_znorm_contrastive"])
        variant_scores["E"].append(vs["E_gram_znorm_contrastive_exmcr"])
        variant_scores["F"].append(vs["F_full_cmsci"])

    rho, p = sp_stats.spearmanr(scores, humans)
    return {
        "rho": float(rho), "p": float(p), "n": len(scores),
        "scores": scores, "humans": humans,
        "variant_scores": variant_scores,
    }


def run_cmsci_v2(samples: list, human_scores: dict) -> dict:
    """Run cMSCI v2 (Gemini) and return correlation."""
    from src.coherence.cmsci_engine_v2 import CalibratedCoherenceEngineV2

    engine = CalibratedCoherenceEngineV2(negative_bank_enabled=True)

    scores = []
    humans = []
    variant_scores = {k: [] for k in ["A", "B", "C", "D", "E"]}

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
        scores.append(result["cmsci_v2"])
        humans.append(human_scores[sid]["weighted_score"]["mean"])

        vs = result["variant_scores"]
        variant_scores["A"].append(vs["A_cosine_avg"])
        variant_scores["B"].append(vs["B_gram"])
        variant_scores["C"].append(vs["C_gram_znorm"])
        variant_scores["D"].append(vs["D_gram_znorm_contrastive"])
        variant_scores["E"].append(vs["E_full_cmsci_v2"])

    rho, p = sp_stats.spearmanr(scores, humans)
    return {
        "rho": float(rho), "p": float(p), "n": len(scores),
        "scores": scores, "humans": humans,
        "variant_scores": variant_scores,
    }


def run_gemini_baselines(samples: list, human_scores: dict) -> dict:
    """Run Gemini cosine baselines."""
    from src.baselines.gemini_baseline import GeminiCosineBaseline

    baseline = GeminiCosineBaseline()
    return baseline.evaluate_all(samples, human_scores)


def bootstrap_rho_ci(
    scores1: list, scores2: list, humans: list,
    n_boot: int = 1000, ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for the difference in Spearman rho between two methods."""
    rng = np.random.default_rng(seed)
    n = len(humans)
    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        h = [humans[i] for i in idx]
        s1 = [scores1[i] for i in idx]
        s2 = [scores2[i] for i in idx]
        rho1, _ = sp_stats.spearmanr(s1, h)
        rho2, _ = sp_stats.spearmanr(s2, h)
        diffs.append(rho1 - rho2)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(diffs, 100 * alpha))
    upper = float(np.percentile(diffs, 100 * (1 - alpha)))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_diff": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_boot": n_boot,
    }


def main():
    parser = argparse.ArgumentParser(description="cMSCI v1 vs v2 Head-to-Head Comparison")
    parser.add_argument("--all-samples", action="store_true", help="Use all 30 samples")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")
    args = parser.parse_args()

    print("=" * 70)
    print("cMSCI v1 (CLIP+CLAP) vs cMSCI v2 (Gemini) Comparison")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")
    all_samples, sample_ids = load_samples(args.all_samples)
    samples = [s for s in all_samples if s["sample_id"] in sample_ids]
    human_scores = load_human_scores()
    print(f"  {len(samples)} samples, {len(human_scores)} human ratings")

    # Run all methods
    results = {}

    print("\n--- Running cMSCI v1 (CLIP+CLAP) ---")
    t0 = time.time()
    results["cmsci_v1"] = run_cmsci_v1(samples, human_scores)
    print(f"  rho={results['cmsci_v1']['rho']:.4f} (p={results['cmsci_v1']['p']:.6f}) "
          f"in {time.time()-t0:.1f}s")

    print("\n--- Running cMSCI v2 (Gemini) ---")
    t0 = time.time()
    results["cmsci_v2"] = run_cmsci_v2(samples, human_scores)
    print(f"  rho={results['cmsci_v2']['rho']:.4f} (p={results['cmsci_v2']['p']:.6f}) "
          f"in {time.time()-t0:.1f}s")

    print("\n--- Running Gemini Baselines ---")
    t0 = time.time()
    gemini_bl = run_gemini_baselines(samples, human_scores)
    results["gemini_2ch"] = gemini_bl.get("2ch", {})
    results["gemini_3ch"] = gemini_bl.get("3ch", {})
    results["gemini_gram3d"] = gemini_bl.get("gram3d", {})
    print(f"  Completed in {time.time()-t0:.1f}s")

    # =====================================================================
    # COMPARISON TABLE
    # =====================================================================
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")

    rows = [
        ("cMSCI v2 (Gemini)", results["cmsci_v2"].get("rho"), results["cmsci_v2"].get("p"), "Unified"),
        ("cMSCI v1 (CLIP+CLAP)", results["cmsci_v1"].get("rho"), results["cmsci_v1"].get("p"), "Dual"),
        ("Gemini cosine (3-ch)", results["gemini_3ch"].get("rho"), results["gemini_3ch"].get("p"), "Unified"),
        ("CCA", 0.409, 0.025, "Dual"),
        ("cosine_znorm", 0.405, 0.031, "Dual"),
        ("Gemini cosine (2-ch)", results["gemini_2ch"].get("rho"), results["gemini_2ch"].get("p"), "Unified"),
        ("Gemini gram 3D", results["gemini_gram3d"].get("rho"), results["gemini_gram3d"].get("p"), "Unified"),
        ("MSCI", 0.257, 0.170, "Dual"),
        ("CLIPScore", 0.201, 0.591, "CLIP"),
    ]

    print(f"\n  {'Method':<25s}  {'rho':>7s}  {'p':>10s}  {'Sig':>3s}  {'Space':>8s}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*10}  {'-'*3}  {'-'*8}")

    for name, rho, p, space in rows:
        if rho is not None and p is not None:
            sig = "*" if p < 0.05 else ""
            print(f"  {name:<25s}  {rho:7.3f}  {p:10.6f}  {sig:>3s}  {space:>8s}")
        else:
            print(f"  {name:<25s}  {'N/A':>7s}  {'N/A':>10s}  {'':>3s}  {space:>8s}")

    # =====================================================================
    # v2 ABLATION TABLE
    # =====================================================================
    print(f"\n{'='*70}")
    print("cMSCI v2 ABLATION TABLE")
    print(f"{'='*70}")

    v2_variants = results["cmsci_v2"].get("variant_scores", {})
    v2_humans = results["cmsci_v2"].get("humans", [])

    print(f"\n  {'Variant':<25s}  {'rho':>7s}  {'p':>10s}  {'Sig':>3s}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*10}  {'-'*3}")

    for vkey, label in [
        ("A", "A: cosine average"),
        ("B", "B: Gramian"),
        ("C", "C: + z-norm"),
        ("D", "D: + contrastive"),
        ("E", "E: full cMSCI v2"),
    ]:
        vals = v2_variants.get(vkey, [])
        if vals and len(vals) == len(v2_humans):
            # Filter out None values
            paired = [(v, h) for v, h in zip(vals, v2_humans) if v is not None]
            if len(paired) >= 5:
                v_scores, h_scores = zip(*paired)
                rho, p = sp_stats.spearmanr(v_scores, h_scores)
                sig = "*" if p < 0.05 else ""
                print(f"  {label:<25s}  {rho:7.3f}  {p:10.6f}  {sig:>3s}")
            else:
                print(f"  {label:<25s}  {'N/A':>7s}  {'N/A':>10s}")
        else:
            print(f"  {label:<25s}  {'N/A':>7s}  {'N/A':>10s}")

    # =====================================================================
    # v1 ABLATION TABLE (for comparison)
    # =====================================================================
    print(f"\n{'='*70}")
    print("cMSCI v1 ABLATION TABLE (for comparison)")
    print(f"{'='*70}")

    v1_variants = results["cmsci_v1"].get("variant_scores", {})
    v1_humans = results["cmsci_v1"].get("humans", [])

    print(f"\n  {'Variant':<30s}  {'rho':>7s}  {'p':>10s}  {'Sig':>3s}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*10}  {'-'*3}")

    for vkey, label in [
        ("A", "A: MSCI"),
        ("B", "B: Gramian"),
        ("C", "C: + z-norm"),
        ("D", "D: + contrastive"),
        ("E", "E: + ExMCR"),
        ("F", "F: full cMSCI v1"),
    ]:
        vals = v1_variants.get(vkey, [])
        if vals and len(vals) == len(v1_humans):
            paired = [(v, h) for v, h in zip(vals, v1_humans) if v is not None]
            if len(paired) >= 5:
                v_scores, h_scores = zip(*paired)
                rho, p = sp_stats.spearmanr(v_scores, h_scores)
                sig = "*" if p < 0.05 else ""
                print(f"  {label:<30s}  {rho:7.3f}  {p:10.6f}  {sig:>3s}")

    # =====================================================================
    # ENSEMBLE v1 + v2
    # =====================================================================
    if (results["cmsci_v2"].get("scores") and results["cmsci_v1"].get("scores")
            and len(results["cmsci_v2"]["scores"]) == len(results["cmsci_v1"]["scores"])):

        print(f"\n{'='*70}")
        print("ENSEMBLE: v1 + v2 Fusion")
        print(f"{'='*70}")

        v1_scores = results["cmsci_v1"]["scores"]
        v2_scores = results["cmsci_v2"]["scores"]
        ens_humans = results["cmsci_v1"]["humans"]
        n = len(ens_humans)

        # Grid search over w_v1
        w_v1_grid = np.arange(0.0, 1.01, 0.1).tolist()
        ens_results = []

        for w_v1 in w_v1_grid:
            ens_scores = [
                w_v1 * v1_scores[i] + (1 - w_v1) * v2_scores[i]
                for i in range(n)
            ]
            rho, p = sp_stats.spearmanr(ens_scores, ens_humans)
            ens_results.append({"w_v1": round(w_v1, 1), "rho": float(rho), "p": float(p)})

        ens_results.sort(key=lambda x: x["rho"], reverse=True)
        best_ens = ens_results[0]

        print(f"\n  {'w_v1':>5s}  {'rho':>7s}  {'p':>10s}  Sig")
        print(f"  {'-'*5}  {'-'*7}  {'-'*10}  ---")
        for er in ens_results:
            sig = "*" if er["p"] < 0.05 else ""
            marker = " <-- best" if er["w_v1"] == best_ens["w_v1"] else ""
            print(f"  {er['w_v1']:5.1f}  {er['rho']:7.3f}  {er['p']:10.6f}  {sig:>3s}{marker}")

        # LOO-CV ensemble: inner grid search per fold
        print(f"\n  LOO-CV Ensemble (inner search per fold):")
        loo_preds = []
        loo_hs = []
        loo_ws = []
        for i in range(n):
            # Train: all except i
            train_v1 = [v1_scores[j] for j in range(n) if j != i]
            train_v2 = [v2_scores[j] for j in range(n) if j != i]
            train_h = [ens_humans[j] for j in range(n) if j != i]

            best_inner_rho = -999
            best_w = 0.5
            for w_v1 in w_v1_grid:
                inner_scores = [
                    w_v1 * train_v1[j] + (1 - w_v1) * train_v2[j]
                    for j in range(len(train_h))
                ]
                rho, _ = sp_stats.spearmanr(inner_scores, train_h)
                if rho > best_inner_rho:
                    best_inner_rho = rho
                    best_w = w_v1

            # Predict held-out
            pred = best_w * v1_scores[i] + (1 - best_w) * v2_scores[i]
            loo_preds.append(pred)
            loo_hs.append(ens_humans[i])
            loo_ws.append(best_w)

        loo_rho, loo_p = sp_stats.spearmanr(loo_preds, loo_hs)
        print(f"    LOO-CV rho: {loo_rho:.4f} (p={loo_p:.6f})")
        w_counts = {}
        for w in loo_ws:
            w_counts[w] = w_counts.get(w, 0) + 1
        for w, c in sorted(w_counts.items(), key=lambda x: -x[1])[:3]:
            print(f"      w_v1={w:.1f}: {c}/{n} folds")

        print(f"\n  BEST ENSEMBLE:")
        print(f"    w_v1 = {best_ens['w_v1']:.1f}")
        print(f"    rho  = {best_ens['rho']:.4f} (p={best_ens['p']:.6f})")
        print(f"    LOO-CV rho = {loo_rho:.4f} (p={loo_p:.6f})")
        print(f"    v1 alone:  rho={results['cmsci_v1']['rho']:.4f}")
        print(f"    v2 alone:  rho={results['cmsci_v2']['rho']:.4f}")

        improvement = best_ens["rho"] - max(results["cmsci_v1"]["rho"], results["cmsci_v2"]["rho"])
        if improvement > 0:
            print(f"    Ensemble IMPROVES over best single by +{improvement:.4f}")
        else:
            print(f"    Ensemble does NOT improve over best single ({improvement:+.4f})")

        results["ensemble"] = {
            "rho": best_ens["rho"],
            "p": best_ens["p"],
            "w_v1": best_ens["w_v1"],
            "loo_rho": float(loo_rho),
            "loo_p": float(loo_p),
            "n": n,
        }

    # =====================================================================
    # BOOTSTRAP CI
    # =====================================================================
    if (results["cmsci_v2"].get("scores") and results["cmsci_v1"].get("scores")
            and len(results["cmsci_v2"]["scores"]) == len(results["cmsci_v1"]["scores"])):

        print(f"\n{'='*70}")
        print(f"BOOTSTRAP 95% CI (n_boot={args.bootstrap})")
        print(f"{'='*70}")

        boot = bootstrap_rho_ci(
            results["cmsci_v2"]["scores"],
            results["cmsci_v1"]["scores"],
            results["cmsci_v2"]["humans"],
            n_boot=args.bootstrap,
        )
        print(f"\n  rho(v2) - rho(v1) = {boot['mean_diff']:.4f}")
        print(f"  95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
        if boot["ci_lower"] > 0:
            print(f"  v2 is SIGNIFICANTLY better than v1 (CI excludes 0)")
        elif boot["ci_upper"] < 0:
            print(f"  v1 is SIGNIFICANTLY better than v2 (CI excludes 0)")
        else:
            print(f"  No significant difference (CI includes 0)")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "comparison_results.json"

    save_data = {}
    for key, val in results.items():
        save_data[key] = {
            "rho": val.get("rho"),
            "p": val.get("p"),
            "n": val.get("n"),
            "sig": val.get("p", 1.0) < 0.05 if val.get("p") is not None else False,
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
