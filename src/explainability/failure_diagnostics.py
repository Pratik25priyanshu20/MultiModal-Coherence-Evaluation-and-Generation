from __future__ import annotations

from typing import Any, Dict, Optional


# Map weakest metric -> human meaning + fix suggestions
METRIC_HINTS = {
    "st_i": {
        "dominant_failure_mode": "text_image_misalignment",
        "suggested_fix": [
            "Make the visual plan more specific: add concrete objects, setting, lighting, and camera cues.",
            "Ensure primary_entities appear in visual_attributes (e.g., 'bus', 'station', 'crowd').",
            "Avoid abstract captions; rewrite into a visualizable scene.",
        ],
    },
    "st_a": {
        "dominant_failure_mode": "text_audio_misalignment",
        "suggested_fix": [
            "Strengthen audio_intent + audio_elements: include distinct sound sources (rain, wind, traffic, birds).",
            "Add timing/texture words: 'distant', 'foreground', 'soft', 'rhythmic', 'echo'.",
            "Avoid silent/ambiguous scenes unless the prompt implies quiet.",
        ],
    },
    "si_a": {
        "dominant_failure_mode": "image_audio_misalignment",
        "suggested_fix": [
            "Align audio sources with visible scene elements (city -> traffic/hum, beach -> waves/seagulls).",
            "Remove conflicting audio elements (e.g., birds in neon city street).",
            "Add must_include constraints tying audio cues to visual objects.",
        ],
    },
    "msci": {
        "dominant_failure_mode": "global_cross_modal_incoherence",
        "suggested_fix": [
            "Regenerate the unified plan with stronger must_include/must_avoid constraints.",
            "Use prompt decomposition: scene -> visual -> audio subplans, then merge.",
            "If repeated failure: retry generation with tighter constraints (regeneration policy).",
        ],
    },
}


def diagnose_run(
    *,
    prompt: str,
    plan: Optional[Dict[str, Any]],
    narrative: Optional[Dict[str, Any]],
    scores: Dict[str, float],
    classification: Dict[str, Any],
    drift: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produces a compact, human-readable diagnostic block for bundle.json.
    """
    weakest = None
    if isinstance(classification, dict):
        weakest = classification.get("weakest_metric")

    hint = METRIC_HINTS.get(weakest, None)

    score_flags = []
    for key in ["msci", "st_i", "st_a", "si_a"]:
        value = scores.get(key)
        if value is not None and value < 0:
            score_flags.append(f"{key}<0")

    drift_flags = []
    if isinstance(drift, dict):
        for key in ["visual_drift", "audio_drift", "global_drift"]:
            if drift.get(key) is True:
                drift_flags.append(key)

    diagnostics = {
        "weakest_metric": weakest,
        "dominant_failure_mode": (hint["dominant_failure_mode"] if hint else "unknown"),
        "suggested_fix": (
            hint["suggested_fix"]
            if hint
            else ["Inspect plan + outputs; no heuristic available."]
        ),
        "evidence": {
            "score_flags": score_flags,
            "drift_flags": drift_flags,
        },
        "notes": {
            "prompt_summary": (prompt[:220] + "...")
            if len(prompt) > 220
            else prompt,
            "plan_domain": (plan.get("domain") if isinstance(plan, dict) else None),
            "plan_scene_summary": (
                plan.get("scene_summary") if isinstance(plan, dict) else None
            ),
        },
    }

    if isinstance(classification, dict) and classification.get("label") == "HIGH_COHERENCE":
        diagnostics["dominant_failure_mode"] = "none_high_coherence"
        diagnostics["suggested_fix"] = [
            "Optional: improve the weakest metric slightly by tightening constraints for that modality.",
            "Run multi-seed stability to ensure coherence is consistent across random seeds.",
        ]

    return diagnostics
