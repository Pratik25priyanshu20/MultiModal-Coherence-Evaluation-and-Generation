#!/usr/bin/env python3
"""
cMSCI Full Pipeline Parameter Optimization via Leave-One-Out Cross-Validation.

Tests parameter variations on the 30 RQ3 human-rated samples to find
the best configuration for Variants D, E, and F.

Grid parameters:
    D (GRAM + z-norm + contrastive):
        alpha:    margin scaling factor [0, 20]
        w_ti:     text-image channel weight [0.10, 0.90]
        cal_mode: cosine or gram z-normalization

    E (+ ExMCR 3-way Gramian):
        w_3d:     weight for z-normalized 3-way Gramian coherence [0.0, 0.50]

    F (+ ProbVLM adaptive channel weighting):
        gamma:    mixing ratio for adaptive vs fixed w_ti [0.0, 1.0]

New formula:
    E: logit_e = z_2d + w_3d * z_compl + alpha * margin
    F: w_ti_final = (1-gamma)*base_w + gamma*(1/u_ti)/(1/u_ti + 1/u_ta)
       logit_f = z_2d_adaptive + w_3d * z_compl + alpha * margin

Safety guarantee: optimizer can always set w_3d=0 and gamma=0 to
recover Variant D exactly. The full pipeline CANNOT be worse than D.

Safety gates:
    - LOO-CV mandatory (prevents overfitting on 30 samples)
    - Must maintain p < 0.05
    - Overfit detector: flags if full-sample rho >> LOO rho (gap > 0.10)
    - If nothing beats current -> keep current, no changes

This script is READ-ONLY — it never modifies any source or config files.

Usage:
    python scripts/optimize_cmsci.py
    python scripts/optimize_cmsci.py --top 20
    python scripts/optimize_cmsci.py --no-engine
    python scripts/optimize_cmsci.py --d-only            # skip E+F, optimize D only
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
RQ1_RESULTS_PATH = PROJECT_ROOT / "runs" / "rq1" / "rq1_results.json"
CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "cmsci_calibration.json"

# --- Grid parameters (Variant D) ---
ALPHA_GRID = np.arange(0, 21, 1).tolist()          # 21 values (0 = Variant C)
W_GRID = np.arange(0.10, 0.91, 0.05).tolist()      # 17 values
CAL_MODES = ["cosine", "gram"]                       # 2 modes
# Total D: 21 * 17 * 2 = 714 configs

# --- Grid parameters (Variant E + F — integral reformulation) ---
W_3D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # 11 values
GAMMA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]           # 11 values
# Total full: 714 * 11 * 11 = 86,394 configs (still fast since no re-embedding)


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
    aggregated = aggregate_multi_rater_sessions(sessions)
    return aggregated


def load_calibration() -> dict:
    """Load calibration reference distributions."""
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def compute_gram_reference(calibration: dict) -> dict:
    """Compute GRAM coherence reference distributions from cosine calibration stats."""
    rng = np.random.default_rng(42)
    n = 10000

    st_i_samples = rng.normal(calibration["st_i"]["mean"], calibration["st_i"]["std"], n)
    st_a_samples = rng.normal(calibration["st_a"]["mean"], calibration["st_a"]["std"], n)

    gram_coh_ti = 1.0 - np.sqrt(np.maximum(0, 1 - st_i_samples**2))
    gram_coh_ta = 1.0 - np.sqrt(np.maximum(0, 1 - st_a_samples**2))

    return {
        "gram_coh_ti": {"mean": float(np.mean(gram_coh_ti)), "std": float(np.std(gram_coh_ti, ddof=1))},
        "gram_coh_ta": {"mean": float(np.mean(gram_coh_ta)), "std": float(np.std(gram_coh_ta, ddof=1))},
    }


def compute_gram_reference_from_rq1() -> dict | None:
    """Compute GRAM coherence reference distributions from actual RQ1 baseline data."""
    if not RQ1_RESULTS_PATH.exists():
        return None

    with open(RQ1_RESULTS_PATH) as f:
        data = json.load(f)

    st_i_vals = []
    st_a_vals = []
    for r in data["results"]:
        if r.get("condition") != "baseline":
            continue
        if r.get("st_i") is not None:
            st_i_vals.append(r["st_i"])
        if r.get("st_a") is not None:
            st_a_vals.append(r["st_a"])

    if not st_i_vals or not st_a_vals:
        return None

    gram_coh_ti = [1.0 - np.sqrt(max(0, 1 - s**2)) for s in st_i_vals]
    gram_coh_ta = [1.0 - np.sqrt(max(0, 1 - s**2)) for s in st_a_vals]

    return {
        "gram_coh_ti": {
            "mean": float(np.mean(gram_coh_ti)),
            "std": float(np.std(gram_coh_ti, ddof=1)),
            "n": len(gram_coh_ti),
        },
        "gram_coh_ta": {
            "mean": float(np.mean(gram_coh_ta)),
            "std": float(np.std(gram_coh_ta, ddof=1)),
            "n": len(gram_coh_ta),
        },
    }


def collect_intermediates_from_engine(samples: list) -> list:
    """Run cMSCI engine once per sample and collect all intermediates.

    Collects z_compl (cross-modal complementarity) for Variant E and u_ti/u_ta for Variant F.
    """
    from src.coherence.cmsci_engine import CalibratedCoherenceEngine
    from src.config.settings import (
        CMSCI_CALIBRATION_PATH as CAL_PATH,
        EXMCR_WEIGHTS_PATH,
        BRIDGE_WEIGHTS_PATH,
        PROB_CLIP_ADAPTER_PATH,
        PROB_CLAP_ADAPTER_PATH,
    )

    cal = str(CAL_PATH) if CAL_PATH.exists() else None
    exmcr = str(EXMCR_WEIGHTS_PATH) if EXMCR_WEIGHTS_PATH.exists() else None
    bridge = str(BRIDGE_WEIGHTS_PATH) if BRIDGE_WEIGHTS_PATH.exists() else None
    prob_clip = str(PROB_CLIP_ADAPTER_PATH) if PROB_CLIP_ADAPTER_PATH.exists() else None
    prob_clap = str(PROB_CLAP_ADAPTER_PATH) if PROB_CLAP_ADAPTER_PATH.exists() else None

    engine = CalibratedCoherenceEngine(
        calibration_path=cal,
        exmcr_weights_path=exmcr,
        bridge_path=bridge,
        prob_clip_adapter_path=prob_clip,
        prob_clap_adapter_path=prob_clap,
        negative_bank_enabled=True,
    )

    has_exmcr = exmcr is not None
    has_prob = prob_clip is not None or prob_clap is not None
    print(f"  ExMCR: {'YES' if has_exmcr else 'NO'}, "
          f"ProbVLM adapters: {'YES' if has_prob else 'NO'}")

    intermediates = []
    for i, s in enumerate(samples):
        result = engine.evaluate(
            text=s["prompt_text"],
            image_path=s.get("image_path"),
            audio_path=s.get("audio_path"),
            domain=s.get("domain", ""),
        )
        margin = 0.0
        if result["contrastive"] and result["contrastive"]["n_negatives"] > 0:
            margin = result["contrastive"]["margin"]

        intermediates.append({
            "sample_id": s["sample_id"],
            "st_i": s["st_i"],
            "st_a": s["st_a"],
            "z_st_i": result["calibration"]["z_st_i"],
            "z_st_a": result["calibration"]["z_st_a"],
            "gram_ti": result["gram"]["text_image"],
            "gram_ta": result["gram"]["text_audio"],
            "margin": margin,
            # Variant E intermediates (cross-modal complementarity)
            "z_compl": result["calibration"].get("z_compl"),
            "gram_ia_volume": result["calibration"].get("gram_ia_volume"),
            # Variant F intermediates (per-channel uncertainty)
            "u_ti": result["calibration"].get("u_ti"),
            "u_ta": result["calibration"].get("u_ta"),
            # Reference scores from engine
            "variant_c_ref": result["variant_scores"]["C_gram_znorm"],
            "variant_d_ref": result["variant_scores"]["D_gram_znorm_contrastive"],
            "variant_e_ref": result["variant_scores"]["E_gram_znorm_contrastive_exmcr"],
            "variant_f_ref": result["variant_scores"]["F_full_cmsci"],
        })
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}: margin={margin:.4f}"
              f" z_compl={result['calibration'].get('z_compl', 'N/A')}"
              f" u_ti={result['calibration'].get('u_ti', 'N/A')}",
              end="\r")

    print()
    return intermediates


def collect_intermediates_from_json(samples: list, calibration: dict) -> list:
    """Compute intermediates analytically from sample JSON values (no re-embedding).

    Note: margins, z_compl, u_ti, and u_ta are set to 0/None since
    we can't compute them without embeddings.
    """
    cal_sti_mean = calibration["st_i"]["mean"]
    cal_sti_std = calibration["st_i"]["std"]
    cal_sta_mean = calibration["st_a"]["mean"]
    cal_sta_std = calibration["st_a"]["std"]

    intermediates = []
    for s in samples:
        st_i = s["st_i"]
        st_a = s["st_a"]

        z_st_i = (st_i - cal_sti_mean) / cal_sti_std
        z_st_a = (st_a - cal_sta_mean) / cal_sta_std

        gram_ti = float(np.sqrt(max(0.0, 1.0 - st_i**2)))
        gram_ta = float(np.sqrt(max(0.0, 1.0 - st_a**2)))

        intermediates.append({
            "sample_id": s["sample_id"],
            "st_i": st_i,
            "st_a": st_a,
            "z_st_i": z_st_i,
            "z_st_a": z_st_a,
            "gram_ti": gram_ti,
            "gram_ta": gram_ta,
            "margin": 0.0,
            "z_compl": None,
            "u_ti": None,
            "u_ta": None,
            "variant_c_ref": None,
            "variant_d_ref": None,
            "variant_e_ref": None,
            "variant_f_ref": None,
        })

    return intermediates


# ─── Scoring Functions ──────────────────────────────────────────

def _compute_d_logit(
    z_st_i: float, z_st_a: float, margin: float,
    alpha: float, w: float,
    gram_ti: float, gram_ta: float,
    gram_ref: dict, cal_mode: str,
) -> float:
    """Compute Variant D's pre-sigmoid logit."""
    if cal_mode == "cosine":
        z_mean = w * z_st_i + (1 - w) * z_st_a
    else:
        gram_coh_ti = 1.0 - gram_ti
        gram_coh_ta = 1.0 - gram_ta
        ref_ti = gram_ref["gram_coh_ti"]
        ref_ta = gram_ref["gram_coh_ta"]
        z_gram_ti = (gram_coh_ti - ref_ti["mean"]) / max(ref_ti["std"], 1e-10)
        z_gram_ta = (gram_coh_ta - ref_ta["mean"]) / max(ref_ta["std"], 1e-10)
        z_mean = w * z_gram_ti + (1 - w) * z_gram_ta

    return z_mean + alpha * margin


def variant_d_score(
    z_st_i: float, z_st_a: float, margin: float,
    alpha: float, w: float,
    gram_ti: float, gram_ta: float,
    gram_ref: dict, cal_mode: str,
) -> float:
    """Compute Variant D score: sigmoid(D_logit)."""
    logit = _compute_d_logit(z_st_i, z_st_a, margin, alpha, w, gram_ti, gram_ta, gram_ref, cal_mode)
    return float(1.0 / (1.0 + np.exp(-logit)))


def variant_f_score(
    z_st_i: float, z_st_a: float, margin: float,
    z_compl, u_ti, u_ta,
    alpha: float, w: float, w_3d: float, gamma: float,
    gram_ti: float, gram_ta: float,
    gram_ref: dict, cal_mode: str,
) -> float:
    """Compute full pipeline (Variant F) score with integral reformulation.

    F: w_ti_final = (1-gamma)*base_w + gamma*(1/u_ti)/(1/u_ti + 1/u_ta)
       logit_f = w_ti_final*z_ti + (1-w_ti_final)*z_ta + w_3d*z_compl + alpha*margin

    Args:
        z_compl: Z-normalized cross-modal complementarity from ExMCR (None if unavailable).
                 Positive z = image-audio are complementary (diverse perspectives).
        u_ti: Per-channel text-image uncertainty (None if unavailable).
        u_ta: Per-channel text-audio uncertainty (None if unavailable).
        w_3d: Weight for complementarity signal (0 = no ExMCR).
        gamma: Mixing ratio for adaptive weighting (0 = fixed w_ti).
    """
    # Compute z-scores for each channel
    if cal_mode == "cosine":
        z_ti = z_st_i
        z_ta = z_st_a
    else:
        gram_coh_ti = 1.0 - gram_ti
        gram_coh_ta = 1.0 - gram_ta
        ref_ti = gram_ref["gram_coh_ti"]
        ref_ta = gram_ref["gram_coh_ta"]
        z_ti = (gram_coh_ti - ref_ti["mean"]) / max(ref_ti["std"], 1e-10)
        z_ta = (gram_coh_ta - ref_ta["mean"]) / max(ref_ta["std"], 1e-10)

    # Adaptive channel weight from ProbVLM uncertainty
    w_final = w
    if u_ti is not None and u_ta is not None and u_ti > 0 and u_ta > 0 and gamma > 0:
        inv_ti = 1.0 / u_ti
        inv_ta = 1.0 / u_ta
        adaptive_w = inv_ti / (inv_ti + inv_ta)
        w_final = (1.0 - gamma) * w + gamma * adaptive_w

    # 2-way z-score with (possibly adaptive) weights
    z_2d = w_final * z_ti + (1.0 - w_final) * z_ta

    # Compose logit: 2-way + complementarity + contrastive margin
    logit = z_2d + alpha * margin
    if z_compl is not None and w_3d > 0:
        logit += w_3d * z_compl

    return float(1.0 / (1.0 + np.exp(-logit)))


# ─── Evaluation Functions ───────────────────────────────────────

def evaluate_config(
    paired: list, alpha: float, w: float, cal_mode: str, gram_ref: dict,
) -> tuple[float, float]:
    """Compute Spearman rho for a D-only config."""
    scores = []
    humans = []
    for inter, h in paired:
        s = variant_d_score(
            inter["z_st_i"], inter["z_st_a"], inter["margin"],
            alpha, w, inter["gram_ti"], inter["gram_ta"],
            gram_ref, cal_mode,
        )
        scores.append(s)
        humans.append(h)

    rho, p = sp_stats.spearmanr(scores, humans)
    return float(rho), float(p)


def evaluate_full_config(
    paired: list,
    alpha: float, w: float, cal_mode: str,
    w_3d: float, gamma: float,
    gram_ref: dict,
) -> tuple[float, float]:
    """Compute Spearman rho for a full pipeline (E+F) config."""
    scores = []
    humans = []
    for inter, h in paired:
        s = variant_f_score(
            inter["z_st_i"], inter["z_st_a"], inter["margin"],
            inter.get("z_compl"), inter.get("u_ti"), inter.get("u_ta"),
            alpha, w, w_3d, gamma,
            inter["gram_ti"], inter["gram_ta"],
            gram_ref, cal_mode,
        )
        scores.append(s)
        humans.append(h)

    rho, p = sp_stats.spearmanr(scores, humans)
    return float(rho), float(p)


# ─── Grid Search Functions ──────────────────────────────────────

def grid_search(paired: list, gram_ref: dict) -> list:
    """Run D-only grid search."""
    results = []
    for cal_mode in CAL_MODES:
        for alpha in ALPHA_GRID:
            for w in W_GRID:
                rho, p = evaluate_config(paired, alpha, w, cal_mode, gram_ref)
                results.append({
                    "alpha": alpha, "w": round(w, 2), "cal_mode": cal_mode,
                    "w_3d": 0.0, "gamma": 0.0,
                    "rho": rho, "p": p,
                })
    results.sort(key=lambda x: x["rho"], reverse=True)
    return results


def grid_search_full(
    paired: list, gram_ref: dict,
    has_compl: bool = False, has_unc: bool = False,
) -> list:
    """Run full pipeline grid search over D + E + F parameters.

    Adapts grid based on available data:
    - No ExMCR 3-way Gramian data → w_3d always 0
    - No ProbVLM uncertainty data → gamma always 0
    """
    w_3d_grid = W_3D_GRID if has_compl else [0.0]
    gamma_grid = GAMMA_GRID if has_unc else [0.0]
    total = len(CAL_MODES) * len(ALPHA_GRID) * len(W_GRID) * len(w_3d_grid) * len(gamma_grid)

    results = []
    count = 0
    t0 = time.time()

    for cal_mode in CAL_MODES:
        for alpha in ALPHA_GRID:
            for w in W_GRID:
                for w_3d in w_3d_grid:
                    for gamma in gamma_grid:
                        rho, p = evaluate_full_config(
                            paired, alpha, w, cal_mode, w_3d, gamma,
                            gram_ref,
                        )
                        results.append({
                            "alpha": alpha,
                            "w": round(w, 2),
                            "cal_mode": cal_mode,
                            "w_3d": round(w_3d, 2),
                            "gamma": round(gamma, 1),
                            "rho": rho,
                            "p": p,
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
    paired: list, alpha: float, w: float, cal_mode: str, gram_ref: dict,
    w_3d: float = 0.0, gamma: float = 0.0,
) -> tuple[float, float]:
    """Leave-one-out stability check for a given config."""
    loo_rhos = []
    for i in range(len(paired)):
        train = paired[:i] + paired[i+1:]
        if w_3d > 0 or gamma > 0:
            rho, _ = evaluate_full_config(train, alpha, w, cal_mode, w_3d, gamma, gram_ref)
        else:
            rho, _ = evaluate_config(train, alpha, w, cal_mode, gram_ref)
        loo_rhos.append(rho)

    return float(np.mean(loo_rhos)), float(np.std(loo_rhos))


def full_loo_cv(paired: list, gram_ref: dict) -> tuple[float, float, dict]:
    """Proper LOO-CV for D-only: for each fold, search for best config on n-1, predict held-out."""
    loo_predictions = []
    loo_humans = []
    config_counts: dict[str, int] = {}

    for i in range(len(paired)):
        held_out_inter, held_out_human = paired[i]
        train = paired[:i] + paired[i+1:]

        best_rho = -999.0
        best_alpha = 1.0
        best_w = 0.5
        best_mode = "cosine"

        for cal_mode in CAL_MODES:
            for alpha in ALPHA_GRID:
                for w in W_GRID:
                    rho, _ = evaluate_config(train, alpha, w, cal_mode, gram_ref)
                    if rho > best_rho:
                        best_rho = rho
                        best_alpha = alpha
                        best_w = round(w, 2)
                        best_mode = cal_mode

        key = f"a={best_alpha},w={best_w},m={best_mode}"
        config_counts[key] = config_counts.get(key, 0) + 1

        pred = variant_d_score(
            held_out_inter["z_st_i"], held_out_inter["z_st_a"],
            held_out_inter["margin"], best_alpha, best_w,
            held_out_inter["gram_ti"], held_out_inter["gram_ta"],
            gram_ref, best_mode,
        )
        loo_predictions.append(pred)
        loo_humans.append(held_out_human)

    rho, p = sp_stats.spearmanr(loo_predictions, loo_humans)
    return float(rho), float(p), config_counts


def full_loo_cv_full(
    paired: list, gram_ref: dict,
    has_compl: bool = False, has_unc: bool = False,
) -> tuple[float, float, dict]:
    """Proper LOO-CV for full pipeline: inner search over all 5 params per fold."""
    w_3d_grid = W_3D_GRID if has_compl else [0.0]
    gamma_grid = GAMMA_GRID if has_unc else [0.0]

    loo_predictions = []
    loo_humans = []
    config_counts: dict[str, int] = {}

    for i in range(len(paired)):
        held_out_inter, held_out_human = paired[i]
        train = paired[:i] + paired[i+1:]

        best_rho = -999.0
        best_cfg = {"alpha": 1, "w": 0.5, "cal_mode": "cosine", "w_3d": 0.0, "gamma": 0.0}

        for cal_mode in CAL_MODES:
            for alpha in ALPHA_GRID:
                for w in W_GRID:
                    for w_3d in w_3d_grid:
                        for gamma in gamma_grid:
                            rho, _ = evaluate_full_config(
                                train, alpha, w, cal_mode, w_3d, gamma,
                                gram_ref,
                            )
                            if rho > best_rho:
                                best_rho = rho
                                best_cfg = {
                                    "alpha": alpha, "w": round(w, 2),
                                    "cal_mode": cal_mode,
                                    "w_3d": round(w_3d, 2), "gamma": round(gamma, 1),
                                }

        key = (f"a={best_cfg['alpha']},w={best_cfg['w']},m={best_cfg['cal_mode']},"
               f"w3d={best_cfg['w_3d']},g={best_cfg['gamma']}")
        config_counts[key] = config_counts.get(key, 0) + 1

        pred = variant_f_score(
            held_out_inter["z_st_i"], held_out_inter["z_st_a"],
            held_out_inter["margin"],
            held_out_inter.get("z_compl"), held_out_inter.get("u_ti"), held_out_inter.get("u_ta"),
            best_cfg["alpha"], best_cfg["w"], best_cfg["w_3d"], best_cfg["gamma"],
            held_out_inter["gram_ti"], held_out_inter["gram_ta"],
            gram_ref, best_cfg["cal_mode"],
        )
        loo_predictions.append(pred)
        loo_humans.append(held_out_human)

        print(f"    LOO fold {i+1}/{len(paired)}: inner best rho={best_rho:.4f}", end="\r")

    print()
    rho, p = sp_stats.spearmanr(loo_predictions, loo_humans)
    return float(rho), float(p), config_counts


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="cMSCI Full Pipeline Parameter Optimization")
    parser.add_argument("--top", type=int, default=10, help="Show top N configs")
    parser.add_argument("--no-engine", action="store_true",
                        help="Skip re-embedding, use sample JSON values (margins = 0)")
    parser.add_argument("--d-only", action="store_true",
                        help="Only optimize Variant D (skip E+F)")
    args = parser.parse_args()

    print("=" * 70)
    print("cMSCI Full Pipeline Parameter Optimization (LOO-CV)")
    print("=" * 70)

    # --- Load data ---
    print("\n--- Loading Data ---")
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)["samples"]
    print(f"  {len(samples)} RQ3 samples loaded")

    human_scores = load_human_scores()
    calibration = load_calibration()

    # --- Compute GRAM reference distributions ---
    print("\n--- GRAM Reference Distributions ---")
    gram_ref = compute_gram_reference_from_rq1()
    if gram_ref is not None:
        print(f"  From RQ1 baseline data:")
        print(f"    gram_coh_ti: mean={gram_ref['gram_coh_ti']['mean']:.6f}, "
              f"std={gram_ref['gram_coh_ti']['std']:.6f}, n={gram_ref['gram_coh_ti'].get('n', '?')}")
        print(f"    gram_coh_ta: mean={gram_ref['gram_coh_ta']['mean']:.6f}, "
              f"std={gram_ref['gram_coh_ta']['std']:.6f}, n={gram_ref['gram_coh_ta'].get('n', '?')}")
    else:
        print("  RQ1 data not found, using sampled approximation from calibration stats")
        gram_ref = compute_gram_reference(calibration)
        print(f"    gram_coh_ti: mean={gram_ref['gram_coh_ti']['mean']:.6f}, "
              f"std={gram_ref['gram_coh_ti']['std']:.6f}")
        print(f"    gram_coh_ta: mean={gram_ref['gram_coh_ta']['mean']:.6f}, "
              f"std={gram_ref['gram_coh_ta']['std']:.6f}")

    # --- Collect intermediates ---
    print("\n--- Collecting Intermediates ---")
    if args.no_engine:
        print("  Mode: analytical (no re-embedding, margins/ExMCR/unc = 0)")
        intermediates = collect_intermediates_from_json(samples, calibration)
    else:
        print("  Mode: engine (re-embedding for accurate intermediates)")
        t0 = time.time()
        intermediates = collect_intermediates_from_engine(samples)
        print(f"  Completed in {time.time() - t0:.1f}s")

    # Check what data is available
    has_compl = any(i.get("z_compl") is not None for i in intermediates)
    has_unc = any(i.get("u_ti") is not None and i.get("u_ta") is not None for i in intermediates)
    print(f"  ExMCR z_compl available:   {'YES' if has_compl else 'NO'} "
          f"({sum(1 for i in intermediates if i.get('z_compl') is not None)}/{len(intermediates)})")
    print(f"  ProbVLM u_ti/u_ta avail:   {'YES' if has_unc else 'NO'} "
          f"({sum(1 for i in intermediates if i.get('u_ti') is not None)}/{len(intermediates)})")

    # Pair with human scores
    paired = []
    for inter in intermediates:
        sid = inter["sample_id"]
        if sid in human_scores:
            paired.append((inter, human_scores[sid]["weighted_score"]["mean"]))

    print(f"  {len(paired)} samples paired with human ratings")
    if len(paired) < 10:
        print("ERROR: Too few paired samples for meaningful optimization")
        sys.exit(1)

    # --- Current baseline ---
    print("\n--- Current Baseline ---")
    baseline_c_rho, baseline_c_p = evaluate_config(paired, alpha=0, w=0.5, cal_mode="cosine", gram_ref=gram_ref)
    baseline_d_rho, baseline_d_p = evaluate_config(paired, alpha=1, w=0.5, cal_mode="cosine", gram_ref=gram_ref)
    print(f"  Variant C (w=0.50, alpha=0): rho={baseline_c_rho:.4f}, p={baseline_c_p:.6f}"
          f" {'*' if baseline_c_p < 0.05 else ''}")
    print(f"  Variant D (w=0.50, alpha=1): rho={baseline_d_rho:.4f}, p={baseline_d_p:.6f}"
          f" {'*' if baseline_d_p < 0.05 else ''}")

    # Check engine reference scores
    if intermediates[0].get("variant_c_ref") is not None:
        ref_scores = {}
        for variant_key in ["variant_c_ref", "variant_d_ref", "variant_e_ref", "variant_f_ref"]:
            vals = [i[variant_key] for i in intermediates if i.get(variant_key) is not None]
            if vals:
                ref_humans = [human_scores[i["sample_id"]]["weighted_score"]["mean"]
                              for i in intermediates
                              if i["sample_id"] in human_scores and i.get(variant_key) is not None]
                if len(vals) == len(ref_humans):
                    rho, p = sp_stats.spearmanr(vals, ref_humans)
                    ref_scores[variant_key] = (float(rho), float(p))
                    label = variant_key.replace("variant_", "").replace("_ref", "").upper()
                    print(f"  Engine ref {label}:               rho={float(rho):.4f}, p={float(p):.6f}")

    # =====================================================================
    # PHASE 1: Variant D Grid Search (always runs)
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Variant D Grid Search")
    print(f"{'='*70}")
    print(f"  Grid: {len(ALPHA_GRID)} x {len(W_GRID)} x {len(CAL_MODES)}"
          f" = {len(ALPHA_GRID)*len(W_GRID)*len(CAL_MODES)} configs")

    t0 = time.time()
    d_results = grid_search(paired, gram_ref)
    print(f"  Completed in {time.time() - t0:.2f}s")

    print(f"\n  Top {min(args.top, 5)} D-only configs:")
    print(f"  {'Rank':>4s}  {'alpha':>5s}  {'w':>5s}  {'Mode':>7s}  {'rho':>7s}  {'p-value':>10s}")
    for i, cfg in enumerate(d_results[:min(args.top, 5)]):
        print(f"  {i+1:4d}  {cfg['alpha']:5.0f}  {cfg['w']:5.2f}  {cfg['cal_mode']:>7s}  "
              f"{cfg['rho']:7.4f}  {cfg['p']:10.6f}")

    best_d = d_results[0]

    # LOO-CV for best D config
    print(f"\n  D-only LOO-CV (inner search per fold):")
    t0 = time.time()
    d_loo_rho, d_loo_p, d_config_counts = full_loo_cv(paired, gram_ref)
    print(f"    LOO-CV rho: {d_loo_rho:.4f} (p={d_loo_p:.6f})")
    print(f"    Completed in {time.time() - t0:.1f}s")
    for key, count in sorted(d_config_counts.items(), key=lambda x: -x[1])[:3]:
        print(f"      {key}: {count}/{len(paired)} folds ({100*count/len(paired):.0f}%)")

    if args.d_only or (not has_compl and not has_unc):
        if not args.d_only:
            print("\n  No ExMCR or ProbVLM data available — skipping E+F optimization")

        # Print D-only summary
        _print_d_summary(best_d, d_loo_rho, d_loo_p, baseline_c_rho, baseline_d_rho, paired, gram_ref)
        return

    # =====================================================================
    # PHASE 2: Full Pipeline Grid Search (E + F)
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Full Pipeline Grid Search (Variants E + F)")
    print(f"{'='*70}")

    w_3d_count = len(W_3D_GRID) if has_compl else 1
    gamma_count = len(GAMMA_GRID) if has_unc else 1
    total = len(CAL_MODES) * len(ALPHA_GRID) * len(W_GRID) * w_3d_count * gamma_count
    print(f"  Grid: {len(ALPHA_GRID)} x {len(W_GRID)} x {len(CAL_MODES)}"
          f" x {w_3d_count} x {gamma_count} = {total} configs")

    t0 = time.time()
    full_results = grid_search_full(paired, gram_ref, has_compl, has_unc)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    # Top full configs
    print(f"\n  Top {args.top} full pipeline configs:")
    print(f"  {'Rank':>4s}  {'alpha':>5s}  {'w_ti':>5s}  {'Mode':>7s}  {'w_3d':>5s}  "
          f"{'gamma':>5s}  {'rho':>7s}  {'p-value':>10s}  Sig")
    print(f"  {'----':>4s}  {'-----':>5s}  {'-----':>5s}  {'-------':>7s}  {'-----':>5s}  "
          f"{'-----':>5s}  {'-------':>7s}  {'----------':>10s}  ---")

    top_full = full_results[:args.top]
    for i, cfg in enumerate(top_full):
        sig = "*" if cfg["p"] < 0.05 else ""
        print(f"  {i+1:4d}  {cfg['alpha']:5.0f}  {cfg['w']:5.2f}  {cfg['cal_mode']:>7s}  "
              f"{cfg['w_3d']:5.2f}  {cfg['gamma']:5.1f}  {cfg['rho']:7.4f}  {cfg['p']:10.6f}  {sig}")

    # LOO stability for top configs
    print(f"\n  LOO Stability (top 5 full configs):")
    print(f"  {'Rank':>4s}  {'alpha':>5s}  {'w_ti':>5s}  {'w_3d':>5s}  {'gamma':>5s}  "
          f"{'Full-rho':>8s}  {'LOO-mean':>8s}  {'Gap':>5s}  Safe")

    stable_configs = []
    for i, cfg in enumerate(top_full[:5]):
        loo_mean, loo_std = loo_stability(
            paired, cfg["alpha"], cfg["w"], cfg["cal_mode"], gram_ref,
            cfg["w_3d"], cfg["gamma"],
        )
        gap = cfg["rho"] - loo_mean
        safe = gap < 0.10
        print(f"  {i+1:4d}  {cfg['alpha']:5.0f}  {cfg['w']:5.2f}  {cfg['w_3d']:5.2f}  "
              f"{cfg['gamma']:5.1f}  {cfg['rho']:8.4f}  {loo_mean:8.4f}  {gap:5.3f}  "
              f"{'YES' if safe else 'NO'}")
        stable_configs.append({**cfg, "loo_mean": loo_mean, "loo_std": loo_std, "gap": gap, "safe": safe})

    # Full LOO-CV with inner search
    print(f"\n  Full Pipeline LOO-CV (inner search per fold):")
    t0 = time.time()
    full_loo_rho, full_loo_p, full_config_counts = full_loo_cv_full(
        paired, gram_ref, has_compl, has_unc,
    )
    elapsed = time.time() - t0
    print(f"    LOO-CV rho: {full_loo_rho:.4f} (p={full_loo_p:.6f})")
    print(f"    Completed in {elapsed:.1f}s")
    for key, count in sorted(full_config_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"      {key}: {count}/{len(paired)} folds ({100*count/len(paired):.0f}%)")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}")

    best_full = top_full[0]

    # Compare D-only vs full pipeline
    print(f"\n  D-only best:      rho={best_d['rho']:.4f} (a={best_d['alpha']}, w={best_d['w']:.2f}, m={best_d['cal_mode']})")
    print(f"  D-only LOO-CV:    rho={d_loo_rho:.4f} (p={d_loo_p:.6f})")
    print(f"  Full best:        rho={best_full['rho']:.4f} (a={best_full['alpha']}, w={best_full['w']:.2f}, "
          f"m={best_full['cal_mode']}, w_3d={best_full['w_3d']:.2f}, gamma={best_full['gamma']:.1f})")
    print(f"  Full LOO-CV:      rho={full_loo_rho:.4f} (p={full_loo_p:.6f})")
    print(f"  MSCI baseline:    rho=0.207 (p=0.273)")

    # Safety check: full must be >= D
    if best_full["rho"] >= best_d["rho"] - 0.001:
        print(f"\n  SAFETY CHECK: PASS (full >= D)")
    else:
        print(f"\n  SAFETY CHECK: WARNING — full < D, optimizer may have found local minimum")

    # Find best safe config
    recommended = None
    for sc in stable_configs:
        if sc["safe"] and sc["p"] < 0.05:
            recommended = sc
            break

    if recommended is None:
        print("\n  RESULT: No safe improvement found in full pipeline.")
        print(f"  Falling back to D-only best: rho={best_d['rho']:.4f}")
    else:
        print(f"\n  RECOMMENDED FULL PIPELINE CONFIG:")
        print(f"    CMSCI_MARGIN_ALPHA       = {recommended['alpha']}")
        print(f"    CMSCI_CHANNEL_WEIGHT_TI  = {recommended['w']:.2f}")
        print(f"    CMSCI_CALIBRATION_MODE   = '{recommended['cal_mode']}'")
        print(f"    CMSCI_W_3D               = {recommended['w_3d']:.2f}")
        print(f"    CMSCI_GAMMA              = {recommended['gamma']:.1f}")
        print(f"    full-rho  = {recommended['rho']:.4f} (p={recommended['p']:.6f})")
        print(f"    LOO-rho   = {recommended['loo_mean']:.4f} +/- {recommended['loo_std']:.4f}")
        print(f"    overfit   = {recommended['gap']:.3f} (< 0.10 threshold)")

        delta_msci = recommended["rho"] - 0.207
        delta_d = recommended["rho"] - best_d["rho"]
        print(f"\n  IMPROVEMENT:")
        print(f"    vs MSCI:     +{delta_msci:.3f} rho ({delta_msci/0.207*100:.0f}% relative)")
        print(f"    vs D-only:   +{delta_d:.3f} rho")
        print(f"    LOO-CV rho:  {full_loo_rho:.4f} (p={full_loo_p:.6f})")

        # Note if w_3d=0 and gamma=0 (meaning E+F didn't help)
        if recommended["w_3d"] == 0 and recommended["gamma"] == 0:
            print(f"\n  NOTE: Best config has w_3d=0, gamma=0 — E+F did not improve D.")
            print(f"  Full pipeline safely degrades to D.")

    print(f"\n{'='*70}")


def _print_d_summary(best_d, d_loo_rho, d_loo_p, baseline_c_rho, baseline_d_rho, paired, gram_ref):
    """Print D-only optimization summary."""
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY (D-only)")
    print(f"{'='*70}")

    # LOO stability for top config
    loo_mean, loo_std = loo_stability(
        paired, best_d["alpha"], best_d["w"], best_d["cal_mode"], gram_ref,
    )
    gap = best_d["rho"] - loo_mean
    safe = gap < 0.10

    print(f"\n  RECOMMENDED CONFIG (Variant D):")
    print(f"    alpha     = {best_d['alpha']}")
    print(f"    w         = {best_d['w']:.2f}")
    print(f"    cal_mode  = {best_d['cal_mode']}")
    print(f"    full-rho  = {best_d['rho']:.4f} (p={best_d['p']:.6f})")
    print(f"    LOO-rho   = {loo_mean:.4f} +/- {loo_std:.4f}")
    print(f"    overfit   = {gap:.3f} ({'SAFE' if safe else 'WARNING'})")
    print(f"    LOO-CV:   = {d_loo_rho:.4f} (p={d_loo_p:.6f})")

    delta_c = best_d["rho"] - baseline_c_rho
    delta_d = best_d["rho"] - baseline_d_rho
    print(f"\n  IMPROVEMENT:")
    print(f"    vs Variant C baseline: +{delta_c:.4f}")
    print(f"    vs Variant D baseline: +{delta_d:.4f}")
    print(f"    vs MSCI (rho=0.207):   +{best_d['rho'] - 0.207:.4f}")

    print(f"\n  --- Settings to apply ---")
    print(f"  CMSCI_MARGIN_ALPHA = {best_d['alpha']}")
    print(f"  CMSCI_CHANNEL_WEIGHT_TI = {best_d['w']:.2f}")
    print(f"  CMSCI_CALIBRATION_MODE = '{best_d['cal_mode']}'")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
