from __future__ import annotations

from typing import Any, Dict


def extract_coherence_features(
    scores: Dict[str, float],
    drift: Dict[str, bool],
    coherence: Dict[str, Any],
) -> Dict[str, float]:
    """
    Convert a run into flat numeric features for classification.
    """
    return {
        "msci": scores.get("msci", 0.0),
        "st_i": scores.get("st_i", 0.0),
        "st_a": scores.get("st_a", 0.0),
        "si_a": scores.get("si_a", 0.0),
        "base_score": coherence.get("base_score", 0.0),
        "final_score": coherence.get("final_score", 0.0),
        "weakest_modality": coherence.get("weakest_modality", 1.0),
        "global_drift": float(drift.get("global_drift", False)),
        "visual_drift": float(drift.get("visual_drift", False)),
        "audio_drift": float(drift.get("audio_drift", False)),
    }
