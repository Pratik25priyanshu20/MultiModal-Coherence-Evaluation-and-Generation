from typing import Any, Dict

LIST_FIELDS = {
    "primary_entities",
    "secondary_entities",
    "visual_attributes",
    "style",
    "mood_emotion",
    "narrative_tone",
    "audio_intent",
    "audio_elements",
    "must_include",
    "must_avoid",
}


def normalize_plan_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all list-based semantic fields are lists,
    even if LLM returns a single string.
    """
    for field in LIST_FIELDS:
        if field in data:
            value = data[field]
            if isinstance(value, str):
                data[field] = [value]
            elif value is None:
                data[field] = []
    return data
