from __future__ import annotations

from typing import Any, Dict, Optional

from src.coherence.thresholds import AdaptiveThresholds


def detect_drift(
    msci: Optional[float],
    st_i: Optional[float],
    st_a: Optional[float],
    si_a: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Drift logic (Phase-3C):
    - global_drift is True if msci is FAIL OR >=2 metrics are FAIL.
    - includes drift_score (fail_count / metrics_count) for debugging.
    """
    drift: Dict[str, Any] = {
        "visual_drift": False,
        "audio_drift": False,
        "global_drift": False,
    }

    if st_i is not None and st_a is not None and st_i + 0.15 < st_a:
        drift["visual_drift"] = True
    if st_i is not None and st_a is not None and st_a + 0.15 < st_i:
        drift["audio_drift"] = True

    metrics: Dict[str, float] = {}
    if msci is not None:
        metrics["msci"] = msci
    if st_i is not None:
        metrics["st_i"] = st_i
    if st_a is not None:
        metrics["st_a"] = st_a
    if si_a is not None:
        metrics["si_a"] = si_a

    try:
        if metrics:
            thresholds = AdaptiveThresholds()
            statuses = {k: thresholds.classify_value(k, v) for k, v in metrics.items()}
            fail_count = sum(1 for s in statuses.values() if s == "FAIL")
            msci_fail = statuses.get("msci") == "FAIL"
            drift["global_drift"] = bool(msci_fail or fail_count >= 2)
            drift["global_drift_score"] = float(fail_count / max(len(statuses), 1))
            drift["fail_count"] = fail_count
            drift["msci_fail"] = msci_fail
            drift["status"] = statuses
        else:
            drift["global_drift"] = False
            drift["global_drift_score"] = 0.0
            drift["fail_count"] = 0
            drift["msci_fail"] = False
            drift["status"] = {}
    except Exception:
        if msci is not None:
            drift["global_drift"] = msci < 0.35
            drift["global_drift_score"] = 1.0 if drift["global_drift"] else 0.0
        else:
            drift["global_drift"] = False
            drift["global_drift_score"] = 0.0

    return drift
