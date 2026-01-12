from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_coherence_stats(path: str = "artifacts/coherence_stats.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_metric(stats: Dict[str, Any], name: str, value: float) -> float:
    """
    Normalize metric to [0,1] using robust percentiles if available.
    Expected stats formats we support:
      stats[name]["p05"], stats[name]["p95"]
      OR stats["metrics"][name]["p05"], ...
    Falls back to (value) clipped to [0,1] if no stats found.
    """
    p05 = _safe_get(stats, name, "p05")
    p95 = _safe_get(stats, name, "p95")
    if p05 is None or p95 is None:
        p05 = _safe_get(stats, "metrics", name, "p05")
        p95 = _safe_get(stats, "metrics", name, "p95")

    if p05 is None or p95 is None or p95 == p05:
        return max(0.0, min(1.0, float(value)))

    v = (float(value) - float(p05)) / (float(p95) - float(p05))
    return max(0.0, min(1.0, v))


@dataclass
class CoherenceScoringConfig:
    w_msci: float = 0.35
    w_st_i: float = 0.20
    w_st_a: float = 0.20
    w_si_a: float = 0.25

    global_drift_penalty: float = 0.18
    visual_drift_penalty: float = 0.10
    audio_drift_penalty: float = 0.10

    weakness_floor: float = 0.35
    weakness_max_extra: float = 0.12


def compute_base_score(
    scores: Dict[str, float],
    stats: Dict[str, Any],
    cfg: CoherenceScoringConfig,
) -> Dict[str, Any]:
    msci = normalize_metric(stats, "msci", scores.get("msci", 0.0))
    st_i = normalize_metric(stats, "st_i", scores.get("st_i", 0.0))
    st_a = normalize_metric(stats, "st_a", scores.get("st_a", 0.0))
    si_a = normalize_metric(stats, "si_a", scores.get("si_a", 0.0))

    weights = [cfg.w_msci, cfg.w_st_i, cfg.w_st_a, cfg.w_si_a]
    wsum = sum(weights) if sum(weights) > 0 else 1.0
    w_msci, w_st_i, w_st_a, w_si_a = [w / wsum for w in weights]

    base = w_msci * msci + w_st_i * st_i + w_st_a * st_a + w_si_a * si_a

    return {
        "base_score": float(max(0.0, min(1.0, base))),
        "normalized": {"msci": msci, "st_i": st_i, "st_a": st_a, "si_a": si_a},
        "weights": {"msci": w_msci, "st_i": w_st_i, "st_a": w_st_a, "si_a": w_si_a},
    }


def compute_drift_penalties(
    normalized: Dict[str, float],
    drift: Dict[str, bool],
    cfg: CoherenceScoringConfig,
) -> Dict[str, Any]:
    penalties: Dict[str, float] = {}

    if drift.get("global_drift", False):
        penalties["global_drift"] = cfg.global_drift_penalty
    if drift.get("visual_drift", False):
        penalties["visual_drift"] = cfg.visual_drift_penalty
    if drift.get("audio_drift", False):
        penalties["audio_drift"] = cfg.audio_drift_penalty

    weakest = min(normalized.values()) if normalized else 1.0
    if weakest < cfg.weakness_floor:
        ratio = (cfg.weakness_floor - weakest) / max(1e-6, cfg.weakness_floor)
        penalties["weakest_modality"] = float(
            min(cfg.weakness_max_extra, cfg.weakness_max_extra * ratio)
        )

    total = float(sum(penalties.values()))
    return {"penalties": penalties, "total_penalty": total, "weakest": float(weakest)}


def compute_final_coherence(
    scores: Dict[str, float],
    drift: Dict[str, bool],
    stats_path: str = "artifacts/coherence_stats.json",
    cfg: Optional[CoherenceScoringConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or CoherenceScoringConfig()
    stats = load_coherence_stats(stats_path)

    base_pack = compute_base_score(scores, stats, cfg)
    drift_pack = compute_drift_penalties(base_pack["normalized"], drift, cfg)

    final = base_pack["base_score"] - drift_pack["total_penalty"]
    final = float(max(0.0, min(1.0, final)))

    return {
        "base_score": base_pack["base_score"],
        "final_score": final,
        "normalized": base_pack["normalized"],
        "weights": base_pack["weights"],
        "penalties": drift_pack["penalties"],
        "total_penalty": drift_pack["total_penalty"],
        "weakest_modality": drift_pack["weakest"],
        "used_stats_file": stats_path,
        "stats_loaded": bool(stats),
    }
