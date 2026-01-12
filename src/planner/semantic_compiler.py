from typing import Any, Dict, List


def _get_nested(plan: Dict[str, Any], *keys: str) -> Any:
    cur: Any = plan
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _join(parts: List[str]) -> str:
    cleaned = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return ", ".join(cleaned)


def compile_semantic_prompts(plan: Dict[str, Any]) -> Dict[str, str]:
    """
    Turn a semantic plan into modality-locked prompts.
    No creativity allowed here.
    """
    core = plan.get("scene_summary") or ""

    primary = _as_list(plan.get("primary_entities"))
    if not primary:
        primary = _as_list(_get_nested(plan, "core_semantics", "main_subjects"))

    visuals = _as_list(plan.get("visual_attributes"))
    if not visuals:
        visuals = (
            _as_list(_get_nested(plan, "style_controls", "visual_style"))
            + _as_list(_get_nested(plan, "style_controls", "color_palette"))
            + _as_list(_get_nested(plan, "style_controls", "lighting"))
            + _as_list(_get_nested(plan, "style_controls", "camera"))
            + _as_list(_get_nested(plan, "image_constraints", "environment_details"))
            + _as_list(_get_nested(plan, "image_constraints", "composition"))
        )

    mood = _as_list(plan.get("mood_emotion"))
    if not mood:
        mood = _as_list(_get_nested(plan, "style_controls", "mood_emotion"))

    audio = _as_list(plan.get("audio_elements"))
    if not audio:
        audio = (
            _as_list(_get_nested(plan, "audio_constraints", "sound_sources"))
            + _as_list(_get_nested(plan, "audio_constraints", "ambience"))
        )

    style = _as_list(plan.get("style"))
    if not style:
        style = _as_list(_get_nested(plan, "style_controls", "visual_style"))

    text_prompt = (
        "Describe the following scene clearly and literally:\n"
        f"Scene: {core}\n"
        f"Entities: {_join(primary)}\n"
        f"Mood: {_join(mood)}\n"
        f"Style: {_join(style)}\n"
        "Do not add new elements."
    )

    image_prompt = (
        f"{core}. "
        f"Visual elements: {_join(visuals)}. "
        f"Entities present: {_join(primary)}. "
        f"Style: {_join(style)}. "
        "No extra objects, no text, no symbols."
    )

    audio_prompt = (
        f"Audio scene matching: {core}. "
        f"Sound elements: {_join(audio)}. "
        f"Mood: {_join(mood)}. "
        "No music unless explicitly stated."
    )

    return {
        "text": text_prompt,
        "image": image_prompt,
        "audio": audio_prompt,
    }
