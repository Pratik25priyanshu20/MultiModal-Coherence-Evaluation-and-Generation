#!/usr/bin/env python3
"""
cMSCI v2 Parameter Optimization via Leave-One-Out Cross-Validation.

Same optimization structure as optimize_cmsci.py but for the Gemini-backed
cMSCI v2 pipeline. Uses cached Gemini embeddings (no API calls needed).

Grid parameters:
    C (GRAM + z-norm):
        w_ti:      text-image channel weight [0.10, 0.90]
        cal_mode:  "gram_2d" or "gram_3d"

    D (+ contrastive margins, 3-channel):
        alpha:     margin scaling [0, 20]

    E (+ complementarity + Matryoshka):
        w_compl:   complementarity weight [0.0, 0.50]
        gamma_mrl: Matryoshka mixing [0.0, 1.0]

Total grid: 21 * 17 * 2 * 11 * 11 = ~86K configs (same scale as v1)
Safety: gamma_mrl=0, w_compl=0 recovers Variant C exactly.

Usage:
    python scripts/optimize_cmsci_v2.py
    python scripts/optimize_cmsci_v2.py --dev-only
    python scripts/optimize_cmsci_v2.py --top 20
    python scripts/optimize_cmsci_v2.py --d-only
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
DEV_TEST_SPLIT_PATH = PROJECT_ROOT / "artifacts" / "dev_test_split.json"

# Grid parameters
ALPHA_GRID = np.arange(0, 21, 1).tolist()              # 21 values
W_GRID = np.arange(0.10, 0.91, 0.05).tolist()          # 17 values
W_IA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20]             # 5 values (IA channel weight)
CAL_MODES = ["gram_2d", "gram_3d"]                       # 2 modes
W_COMPL_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
GAMMA_MRL_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_human_scores() -> dict:
    """Load and aggregate human evaluation scores from RQ3 sessions."""
    from src.evaluation.human_eval_schema import EvaluationSession
    from src.evaluation.human_eval_analyzer import aggregate_multi_rater_sessions

    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json")):
        try:
            session = EvaluationSession.load(p)
            if session.progress >= 80.0:
                sessions.append(session)
        except Exception:
            pass

    if len(sessions) < 2:
        print(f"ERROR: Need >= 2 sessions, found {len(sessions)}")
        sys.exit(1)

    print(f"  Loaded {len(sessions)} evaluation sessions")
    return aggregate_multi_rater_sessions(sessions)


def collect_intermediates(samples: list, dev_only: bool = False) -> list:
    """Run cMSCI v2 engine once per sample and collect all intermediates."""
    from src.coherence.cmsci_engine_v2 import CalibratedCoherenceEngineV2

    engine = CalibratedCoherenceEngineV2(negative_bank_enabled=True)

    # Filter to dev set if requested
    if dev_only and DEV_TEST_SPLIT_PATH.exists():
        with open(DEV_TEST_SPLIT_PATH) as f:
            split = json.load(f)
        dev_ids = set(split.get("dev_ids", split.get("dev", [])))
        samples = [s for s in samples if s["sample_id"] in dev_ids]
        print(f"  Filtered to {len(samples)} dev samples")

    intermediates = []
    for i, s in enumerate(samples):
        result = engine.evaluate(
            text=s["prompt_text"],
            image_path=s.get("image_path"),
            audio_path=s.get("audio_path"),
            domain=s.get("domain", ""),
        )

        margin = 0.0
        margin_ti = 0.0
        margin_ta = 0.0
        margin_ia = 0.0
        if result["contrastive"] and result["contrastive"]["n_negatives"] > 0:
            margin = result["contrastive"].get("margin", 0.0)
            margin_ti = result["contrastive"].get("margin_ti", 0.0)
            margin_ta = result["contrastive"].get("margin_ta", 0.0)
            margin_ia = result["contrastive"].get("margin_ia", 0.0)

        # Multi-scale fused coherences (if available)
        ms = result.get("multiscale") or {}
        fused_ti = ms.get("fused_ti")
        fused_ta = ms.get("fused_ta")
        fused_ia = ms.get("fused_ia")
        fused_tia = ms.get("fused_tia")

        intermediates.append({
            "sample_id": s["sample_id"],
            # Gram volumes
            "gram_ti": result["gram"]["ti"],
            "gram_ta": result["gram"]["ta"],
            "gram_ia": result["gram"]["ia"],
            "gram_tia": result["gram"]["tia"],
            "gram_coherence_2d_avg": result["gram"]["coherence_2d_avg"],
            "gram_coherence_3d": result["gram"]["coherence_3d"],
            # Calibration z-scores
            "z_gram_ti": result["calibration"]["z_gram_ti"],
            "z_gram_ta": result["calibration"]["z_gram_ta"],
            "z_gram_ia": result["calibration"]["z_gram_ia"],
            "z_gram_tia": result["calibration"]["z_gram_tia"],
            # Margins
            "margin": margin,
            "margin_ti": margin_ti,
            "margin_ta": margin_ta,
            "margin_ia": margin_ia,
            # Complementarity
            "z_compl": result["calibration"].get("z_compl"),
            # Matryoshka
            "matryoshka_consistency": (
                result["matryoshka"]["consistency"] if result.get("matryoshka") else None
            ),
            "mrl_raw_w_ti": result["calibration"].get("mrl_raw_w_ti"),
            # Multi-scale fused coherences
            "fused_ti": fused_ti,
            "fused_ta": fused_ta,
            "fused_ia": fused_ia,
            "fused_tia": fused_tia,
            # Reference scores from engine
            "variant_a_ref": result["variant_scores"]["A_cosine_avg"],
            "variant_b_ref": result["variant_scores"]["B_gram"],
            "variant_c_ref": result["variant_scores"]["C_gram_znorm"],
            "variant_d_ref": result["variant_scores"]["D_gram_znorm_contrastive"],
            "variant_e_ref": result["variant_scores"]["E_full_cmsci_v2"],
        })
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}: margin={margin:.4f}", end="\r")

    print()
    return intermediates


# ─── Scoring Functions ──────────────────────────────────────────

def _compute_z_2d(inter: dict, w: float, cal_mode: str, w_ia: float = 0.0) -> float | None:
    """Compute 2-way z-score from intermediates."""
    if cal_mode == "gram_3d" and inter.get("z_gram_tia") is not None:
        return inter["z_gram_tia"]

    # Weighted average of 2D z-scores
    z_parts = []
    z_weights = []
    if inter.get("z_gram_ti") is not None:
        z_parts.append(inter["z_gram_ti"])
        z_weights.append(w)
    if inter.get("z_gram_ta") is not None:
        z_parts.append(inter["z_gram_ta"])
        z_weights.append(1.0 - w)
    if inter.get("z_gram_ia") is not None and w_ia > 0:
        z_parts.append(inter["z_gram_ia"])
        z_weights.append(w_ia)

    if not z_parts:
        return None

    total_w = sum(z_weights)
    return sum(z * wt for z, wt in zip(z_parts, z_weights)) / total_w


def variant_d_score(inter: dict, alpha: float, w: float, cal_mode: str, w_ia: float = 0.0) -> float:
    """Compute Variant D: sigmoid(z_2d + alpha * margin)."""
    z_2d = _compute_z_2d(inter, w, cal_mode, w_ia=w_ia)
    if z_2d is None:
        return 0.5
    logit = z_2d + alpha * inter.get("margin", 0.0)
    return float(1.0 / (1.0 + np.exp(-logit)))


def variant_e_score(
    inter: dict,
    alpha: float, w: float, cal_mode: str,
    w_compl: float, gamma_mrl: float,
    w_ia: float = 0.0,
) -> float:
    """Compute Variant E: full cMSCI v2 with complementarity + Matryoshka."""
    # Adaptive weight from Matryoshka scale consistency
    # mrl_raw_w_ti is the raw consistency-based weight BEFORE gamma mixing
    w_final = w
    if gamma_mrl > 0 and inter.get("mrl_raw_w_ti") is not None:
        w_final = (1.0 - gamma_mrl) * w + gamma_mrl * inter["mrl_raw_w_ti"]

    z_2d = _compute_z_2d(inter, w_final, cal_mode, w_ia=w_ia)
    if z_2d is None:
        return 0.5

    logit = z_2d + alpha * inter.get("margin", 0.0)

    if w_compl > 0 and inter.get("z_compl") is not None:
        logit += w_compl * inter["z_compl"]

    return float(1.0 / (1.0 + np.exp(-logit)))


# ─── Evaluation Functions ───────────────────────────────────────

def evaluate_d_config(paired: list, alpha: float, w: float, cal_mode: str, w_ia: float = 0.0) -> tuple[float, float]:
    """Compute Spearman rho for a D-only config."""
    scores = [variant_d_score(inter, alpha, w, cal_mode, w_ia=w_ia) for inter, _ in paired]
    humans = [h for _, h in paired]
    rho, p = sp_stats.spearmanr(scores, humans)
    return float(rho), float(p)


def evaluate_full_config(
    paired: list,
    alpha: float, w: float, cal_mode: str,
    w_compl: float, gamma_mrl: float,
    w_ia: float = 0.0,
) -> tuple[float, float]:
    """Compute Spearman rho for a full pipeline config."""
    scores = [
        variant_e_score(inter, alpha, w, cal_mode, w_compl, gamma_mrl, w_ia=w_ia)
        for inter, _ in paired
    ]
    humans = [h for _, h in paired]
    rho, p = sp_stats.spearmanr(scores, humans)
    return float(rho), float(p)


# ─── Grid Search Functions ──────────────────────────────────────

def grid_search_d(paired: list) -> list:
    """Run D-only grid search."""
    results = []
    for cal_mode in CAL_MODES:
        for alpha in ALPHA_GRID:
            for w in W_GRID:
                for w_ia in W_IA_GRID:
                    rho, p = evaluate_d_config(paired, alpha, w, cal_mode, w_ia=w_ia)
                    results.append({
                        "alpha": alpha, "w": round(w, 2), "cal_mode": cal_mode,
                        "w_ia": round(w_ia, 2),
                        "w_compl": 0.0, "gamma_mrl": 0.0,
                        "rho": rho, "p": p,
                    })
    results.sort(key=lambda x: x["rho"], reverse=True)
    return results


def grid_search_full(
    paired: list,
    has_compl: bool = False,
    has_matryoshka: bool = False,
) -> list:
    """Run full pipeline grid search."""
    w_compl_grid = W_COMPL_GRID if has_compl else [0.0]
    gamma_grid = GAMMA_MRL_GRID if has_matryoshka else [0.0]
    total = (len(CAL_MODES) * len(ALPHA_GRID) * len(W_GRID)
             * len(W_IA_GRID) * len(w_compl_grid) * len(gamma_grid))

    results = []
    count = 0
    t0 = time.time()

    for cal_mode in CAL_MODES:
        for alpha in ALPHA_GRID:
            for w in W_GRID:
                for w_ia in W_IA_GRID:
                    for w_compl in w_compl_grid:
                        for gamma_mrl in gamma_grid:
                            rho, p = evaluate_full_config(
                                paired, alpha, w, cal_mode, w_compl, gamma_mrl,
                                w_ia=w_ia,
                            )
                            results.append({
                                "alpha": alpha, "w": round(w, 2),
                                "w_ia": round(w_ia, 2),
                                "cal_mode": cal_mode,
                                "w_compl": round(w_compl, 2),
                                "gamma_mrl": round(gamma_mrl, 1),
                                "rho": rho, "p": p,
                            })
                            count += 1
                            if count % 5000 == 0:
                                elapsed = time.time() - t0
                                print(f"    {count}/{total} ({100*count/total:.0f}%) "
                                      f"elapsed={elapsed:.1f}s", end="\r")

    results.sort(key=lambda x: x["rho"], reverse=True)
    return results


# ─── LOO-CV Functions ───────────────────────────────────────────

def loo_stability(
    paired: list,
    alpha: float, w: float, cal_mode: str,
    w_compl: float = 0.0, gamma_mrl: float = 0.0,
    w_ia: float = 0.0,
) -> tuple[float, float]:
    """Leave-one-out stability check."""
    loo_rhos = []
    for i in range(len(paired)):
        train = paired[:i] + paired[i+1:]
        if w_compl > 0 or gamma_mrl > 0:
            rho, _ = evaluate_full_config(train, alpha, w, cal_mode, w_compl, gamma_mrl, w_ia=w_ia)
        else:
            rho, _ = evaluate_d_config(train, alpha, w, cal_mode, w_ia=w_ia)
        loo_rhos.append(rho)
    return float(np.mean(loo_rhos)), float(np.std(loo_rhos))


def full_loo_cv(
    paired: list,
    has_compl: bool = False,
    has_matryoshka: bool = False,
) -> tuple[float, float, dict]:
    """Proper LOO-CV: inner search per fold, predict held-out."""
    w_compl_grid = W_COMPL_GRID if has_compl else [0.0]
    gamma_grid = GAMMA_MRL_GRID if has_matryoshka else [0.0]

    loo_predictions = []
    loo_humans = []
    config_counts: dict[str, int] = {}

    for i in range(len(paired)):
        held_out_inter, held_out_human = paired[i]
        train = paired[:i] + paired[i+1:]

        best_rho = -999.0
        best_cfg = {"alpha": 1, "w": 0.5, "cal_mode": "gram_2d",
                     "w_ia": 0.0, "w_compl": 0.0, "gamma_mrl": 0.0}

        for cal_mode in CAL_MODES:
            for alpha in ALPHA_GRID:
                for w in W_GRID:
                    for w_ia in W_IA_GRID:
                        for w_compl in w_compl_grid:
                            for gamma_mrl in gamma_grid:
                                rho, _ = evaluate_full_config(
                                    train, alpha, w, cal_mode, w_compl, gamma_mrl,
                                    w_ia=w_ia,
                                )
                                if rho > best_rho:
                                    best_rho = rho
                                    best_cfg = {
                                        "alpha": alpha, "w": round(w, 2),
                                        "w_ia": round(w_ia, 2),
                                        "cal_mode": cal_mode,
                                        "w_compl": round(w_compl, 2),
                                        "gamma_mrl": round(gamma_mrl, 1),
                                    }

        key = (f"a={best_cfg['alpha']},w={best_cfg['w']},wia={best_cfg['w_ia']},"
               f"m={best_cfg['cal_mode']},wc={best_cfg['w_compl']},g={best_cfg['gamma_mrl']}")
        config_counts[key] = config_counts.get(key, 0) + 1

        pred = variant_e_score(
            held_out_inter,
            best_cfg["alpha"], best_cfg["w"], best_cfg["cal_mode"],
            best_cfg["w_compl"], best_cfg["gamma_mrl"],
            w_ia=best_cfg["w_ia"],
        )
        loo_predictions.append(pred)
        loo_humans.append(held_out_human)
        print(f"    LOO fold {i+1}/{len(paired)}: inner best rho={best_rho:.4f}", end="\r")

    print()
    rho, p = sp_stats.spearmanr(loo_predictions, loo_humans)
    return float(rho), float(p), config_counts


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="cMSCI v2 Parameter Optimization (LOO-CV)")
    parser.add_argument("--top", type=int, default=10, help="Show top N configs")
    parser.add_argument("--dev-only", action="store_true", help="Use dev set only")
    parser.add_argument("--d-only", action="store_true", help="Only optimize Variant D")
    args = parser.parse_args()

    print("=" * 70)
    print("cMSCI v2 (Gemini) Parameter Optimization (LOO-CV)")
    print("=" * 70)

    # --- Load data ---
    print("\n--- Loading Data ---")
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]
    print(f"  {len(samples)} RQ3 samples loaded")

    human_scores = load_human_scores()

    # --- Collect intermediates ---
    print("\n--- Collecting Intermediates (Gemini cMSCI v2) ---")
    t0 = time.time()
    intermediates = collect_intermediates(samples, dev_only=args.dev_only)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Check data availability
    has_compl = any(i.get("z_compl") is not None for i in intermediates)
    has_matryoshka = any(i.get("matryoshka_consistency") is not None for i in intermediates)
    print(f"  z_compl available: {'YES' if has_compl else 'NO'}")
    print(f"  Matryoshka available: {'YES' if has_matryoshka else 'NO'}")

    # Pair with human scores
    paired = []
    for inter in intermediates:
        sid = inter["sample_id"]
        if sid in human_scores:
            paired.append((inter, human_scores[sid]["weighted_score"]["mean"]))

    print(f"  {len(paired)} samples paired with human ratings")
    if len(paired) < 10:
        print("ERROR: Too few paired samples")
        sys.exit(1)

    # --- Reference scores ---
    print("\n--- Engine Reference Scores ---")
    for vkey, label in [
        ("variant_a_ref", "A (cosine avg)"),
        ("variant_b_ref", "B (gram)"),
        ("variant_c_ref", "C (gram+znorm)"),
        ("variant_d_ref", "D (contrastive)"),
        ("variant_e_ref", "E (full cMSCI v2)"),
    ]:
        vals = [i[vkey] for i in intermediates if i.get(vkey) is not None]
        humans = [
            human_scores[i["sample_id"]]["weighted_score"]["mean"]
            for i in intermediates
            if i["sample_id"] in human_scores and i.get(vkey) is not None
        ]
        if len(vals) == len(humans) and len(vals) >= 5:
            rho, p = sp_stats.spearmanr(vals, humans)
            sig = "*" if p < 0.05 else ""
            print(f"  {label:25s}: rho={rho:.4f} (p={p:.6f}) {sig}")

    # =====================================================================
    # PHASE 1: Variant D Grid Search
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Variant D Grid Search")
    print(f"{'='*70}")
    total_d = len(ALPHA_GRID) * len(W_GRID) * len(W_IA_GRID) * len(CAL_MODES)
    print(f"  Grid: {len(ALPHA_GRID)} x {len(W_GRID)} x {len(W_IA_GRID)} x {len(CAL_MODES)} = {total_d} configs")

    t0 = time.time()
    d_results = grid_search_d(paired)
    print(f"  Completed in {time.time() - t0:.2f}s")

    print(f"\n  Top 5 D-only configs:")
    print(f"  {'Rank':>4s}  {'alpha':>5s}  {'w':>5s}  {'w_ia':>5s}  {'Mode':>7s}  {'rho':>7s}  {'p-value':>10s}")
    for i, cfg in enumerate(d_results[:5]):
        print(f"  {i+1:4d}  {cfg['alpha']:5.0f}  {cfg['w']:5.2f}  {cfg['w_ia']:5.2f}  {cfg['cal_mode']:>7s}  "
              f"{cfg['rho']:7.4f}  {cfg['p']:10.6f}")

    best_d = d_results[0]

    if args.d_only or (not has_compl and not has_matryoshka):
        if not args.d_only:
            print("\n  No complementarity or Matryoshka data — skipping E optimization")

        loo_mean, loo_std = loo_stability(
            paired, best_d["alpha"], best_d["w"], best_d["cal_mode"], w_ia=best_d["w_ia"],
        )
        print(f"\n  RECOMMENDED (Variant D):")
        print(f"    CMSCI_V2_ALPHA    = {best_d['alpha']}")
        print(f"    CMSCI_V2_W_TI     = {best_d['w']:.2f}")
        print(f"    CMSCI_V2_W_IA     = {best_d['w_ia']:.2f}")
        print(f"    CMSCI_V2_CAL_MODE = '{best_d['cal_mode']}'")
        print(f"    rho = {best_d['rho']:.4f}, LOO = {loo_mean:.4f} +/- {loo_std:.4f}")
        return

    # =====================================================================
    # PHASE 2: Full Pipeline Grid Search (Variant E)
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Full Pipeline Grid Search (Variant E)")
    print(f"{'='*70}")

    wc_count = len(W_COMPL_GRID) if has_compl else 1
    gm_count = len(GAMMA_MRL_GRID) if has_matryoshka else 1
    total = len(CAL_MODES) * len(ALPHA_GRID) * len(W_GRID) * len(W_IA_GRID) * wc_count * gm_count
    print(f"  Grid: {len(ALPHA_GRID)} x {len(W_GRID)} x {len(W_IA_GRID)} x {len(CAL_MODES)}"
          f" x {wc_count} x {gm_count} = {total} configs")

    t0 = time.time()
    full_results = grid_search_full(paired, has_compl, has_matryoshka)
    print(f"\n  Completed in {time.time() - t0:.1f}s")

    print(f"\n  Top {args.top} full pipeline configs:")
    print(f"  {'Rank':>4s}  {'alpha':>5s}  {'w_ti':>5s}  {'w_ia':>5s}  {'Mode':>7s}  {'w_compl':>7s}  "
          f"{'gamma':>5s}  {'rho':>7s}  {'p-value':>10s}  Sig")
    for i, cfg in enumerate(full_results[:args.top]):
        sig = "*" if cfg["p"] < 0.05 else ""
        print(f"  {i+1:4d}  {cfg['alpha']:5.0f}  {cfg['w']:5.2f}  {cfg['w_ia']:5.2f}  {cfg['cal_mode']:>7s}  "
              f"{cfg['w_compl']:7.2f}  {cfg['gamma_mrl']:5.1f}  {cfg['rho']:7.4f}  "
              f"{cfg['p']:10.6f}  {sig}")

    # LOO stability for top 5
    print(f"\n  LOO Stability (top 5):")
    stable_configs = []
    for i, cfg in enumerate(full_results[:5]):
        loo_mean, loo_std = loo_stability(
            paired, cfg["alpha"], cfg["w"], cfg["cal_mode"],
            cfg["w_compl"], cfg["gamma_mrl"], w_ia=cfg["w_ia"],
        )
        gap = cfg["rho"] - loo_mean
        safe = gap < 0.10
        print(f"  {i+1:4d}  rho={cfg['rho']:.4f}  LOO={loo_mean:.4f}  gap={gap:.3f}  "
              f"{'SAFE' if safe else 'WARN'}")
        stable_configs.append({**cfg, "loo_mean": loo_mean, "loo_std": loo_std, "gap": gap, "safe": safe})

    # Full LOO-CV
    print(f"\n  Full LOO-CV (inner search per fold):")
    t0 = time.time()
    full_loo_rho, full_loo_p, config_counts = full_loo_cv(paired, has_compl, has_matryoshka)
    print(f"    LOO-CV rho: {full_loo_rho:.4f} (p={full_loo_p:.6f})")
    print(f"    Completed in {time.time() - t0:.1f}s")
    for key, count in sorted(config_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"      {key}: {count}/{len(paired)} folds")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY (cMSCI v2 / Gemini)")
    print(f"{'='*70}")

    best_full = full_results[0]
    print(f"\n  D-only best:   rho={best_d['rho']:.4f}")
    print(f"  Full best:     rho={best_full['rho']:.4f}")
    print(f"  Full LOO-CV:   rho={full_loo_rho:.4f} (p={full_loo_p:.6f})")

    # Find best safe config
    recommended = None
    for sc in stable_configs:
        if sc["safe"] and sc["p"] < 0.05:
            recommended = sc
            break

    if recommended is None:
        print("\n  No safe improvement found. Using D-only best.")
        recommended = best_d

    print(f"\n  RECOMMENDED CONFIG:")
    print(f"    CMSCI_V2_ALPHA     = {recommended['alpha']}")
    print(f"    CMSCI_V2_W_TI      = {recommended['w']:.2f}")
    print(f"    CMSCI_V2_W_IA      = {recommended.get('w_ia', 0.0):.2f}")
    print(f"    CMSCI_V2_CAL_MODE  = '{recommended['cal_mode']}'")
    print(f"    CMSCI_V2_W_COMPL   = {recommended.get('w_compl', 0.0):.2f}")
    print(f"    CMSCI_V2_GAMMA_MRL = {recommended.get('gamma_mrl', 0.0):.1f}")
    print(f"    rho = {recommended['rho']:.4f} (p={recommended['p']:.6f})")
    if "loo_mean" in recommended:
        print(f"    LOO = {recommended['loo_mean']:.4f} +/- {recommended['loo_std']:.4f}")
        print(f"    overfit gap = {recommended['gap']:.3f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
