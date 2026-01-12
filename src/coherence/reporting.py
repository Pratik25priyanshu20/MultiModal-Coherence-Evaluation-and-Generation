from __future__ import annotations

from typing import Any, Dict

LABEL_EXPLANATION = {
    "HIGH_COHERENCE": "All modalities align strongly in meaning and tone.",
    "LOCAL_MODALITY_WEAKNESS": "One modality shows mild abstraction but does not harm overall coherence.",
    "MODALITY_FAILURE": "Modalities diverge semantically and require regeneration.",
    "GLOBAL_FAILURE": "Cross-modal semantic alignment failed.",
}

LABEL_TO_SUMMARY = {
    "HIGH_COHERENCE": "Strong cross-modal semantic agreement.",
    "LOCAL_MODALITY_WEAKNESS": "Minor abstraction in one modality; overall coherence preserved.",
    "MODALITY_FAILURE": "Significant mismatch detected; regeneration required.",
    "GLOBAL_FAILURE": "Global mismatch detected; full regeneration required.",
}

LABEL_TO_DECISION = {
    "HIGH_COHERENCE": ("ACCEPTED", "high", "pipeline_completed"),
    "LOCAL_MODALITY_WEAKNESS": ("ACCEPTED_WITH_NOTE", "medium", "pipeline_completed"),
    "MODALITY_FAILURE": ("REGENERATE", "low", "targeted_regeneration"),
    "GLOBAL_FAILURE": ("REGENERATE", "low", "full_regeneration"),
}


def build_final_assessment(
    coherence: Dict[str, Any],
    retry_outcomes: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    classification = coherence.get("classification", {})
    label = classification.get("label", "UNKNOWN")
    reason = classification.get("reason", "No classification available.")

    decision, confidence, system_action = LABEL_TO_DECISION.get(
        label,
        ("UNKNOWN", "unknown", "none"),
    )
    summary = LABEL_TO_SUMMARY.get(label, "No summary available.")

    if retry_outcomes:
        system_action = "retry_attempted"

    return {
        "decision": decision,
        "confidence": confidence,
        "summary": f"{summary} Reason: {reason}",
        "system_action": system_action,
    }
