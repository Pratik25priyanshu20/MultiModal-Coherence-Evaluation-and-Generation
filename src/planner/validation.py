from typing import Any, Dict

from src.planner.schema import SemanticPlan

_LIST_FIELDS = {
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

_LIST_PATHS = [
    ("core_semantics", "main_subjects"),
    ("core_semantics", "actions"),
    ("style_controls", "visual_style"),
    ("style_controls", "color_palette"),
    ("style_controls", "lighting"),
    ("style_controls", "camera"),
    ("style_controls", "mood_emotion"),
    ("style_controls", "narrative_tone"),
    ("image_constraints", "must_include"),
    ("image_constraints", "must_avoid"),
    ("image_constraints", "objects"),
    ("image_constraints", "environment_details"),
    ("image_constraints", "composition"),
    ("audio_constraints", "audio_intent"),
    ("audio_constraints", "sound_sources"),
    ("audio_constraints", "ambience"),
    ("audio_constraints", "must_include"),
    ("audio_constraints", "must_avoid"),
    ("text_constraints", "must_include"),
    ("text_constraints", "must_avoid"),
    ("text_constraints", "keywords"),
]

_STRING_PATHS = [
    ("scene_summary",),
    ("domain",),
    ("core_semantics", "setting"),
    ("core_semantics", "time_of_day"),
    ("core_semantics", "weather"),
    ("audio_constraints", "tempo"),
    ("text_constraints", "length"),
]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return ""
    return str(value)


def _get_parent(data: Dict[str, Any], path: tuple[str, ...]) -> Dict[str, Any] | None:
    cur: Any = data
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if not isinstance(cur, dict):
        return None
    return cur


def _normalize_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in _LIST_FIELDS:
        if key in data:
            data[key] = _as_list(data[key])

    for path in _LIST_PATHS:
        parent = _get_parent(data, path)
        if parent is not None and path[-1] in parent:
            parent[path[-1]] = _as_list(parent[path[-1]])

    for path in _STRING_PATHS:
        parent = _get_parent(data, path)
        if parent is not None and path[-1] in parent:
            parent[path[-1]] = _as_str(parent[path[-1]])

    return data


def validate_semantic_plan_dict(data: Dict[str, Any]) -> None:
    data = _normalize_fields(data)
    SemanticPlan(**data)
