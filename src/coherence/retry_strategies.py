from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _plan_list(plan: Any, name: str) -> List[str]:
    if isinstance(plan, dict):
        value = plan.get(name, [])
    else:
        value = getattr(plan, name, [])
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def check_constraints(text_output: str, plan: Any) -> List[tuple[str, str]]:
    violations: List[tuple[str, str]] = []
    text_lower = (text_output or "").lower()

    for term in _plan_list(plan, "must_include"):
        if term.lower() not in text_lower:
            violations.append(("missing", term))

    for term in _plan_list(plan, "must_avoid"):
        if term.lower() in text_lower:
            violations.append(("forbidden", term))

    return violations


def retry_si_a(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix image-audio semantic mismatch by regenerating audio grounded in visuals.
    """
    plan = context.get("semantic_plan", {})
    narrative = context.get("narrative_structured", {})

    image_caption = narrative.get("visual_description") or plan.get("scene_summary", "")
    visual_entities = plan.get("primary_entities", [])

    audio_prompt = f"""
Generate environmental audio that matches this visual scene:

Scene description:
{image_caption}

Key visual elements:
{', '.join(visual_entities)}

Audio should reflect environment, materials, motion, and atmosphere.
Avoid unrelated sounds.
"""

    return {
        "regenerate": "audio",
        "failed_metric": "si_a",
        "strategy": "ALIGN_AUDIO_TO_IMAGE",
        "audio_prompt": audio_prompt.strip(),
    }


def retry_st_a(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix text-audio semantic mismatch by regenerating audio grounded in text.
    """
    plan = context.get("semantic_plan", {})
    narrative = context.get("narrative_structured", {})

    text_scene = narrative.get("combined_scene") or plan.get("scene_summary", "")
    audio_elements = plan.get("audio_elements", [])
    mood = plan.get("mood_emotion", [])

    audio_prompt = f"""
Generate environmental audio aligned with this scene description:

Scene description:
{text_scene}

Audio elements:
{', '.join(audio_elements)}

Mood:
{', '.join(mood)}

Avoid unrelated sounds or musical shifts.
"""

    return {
        "regenerate": "audio",
        "failed_metric": "st_a",
        "strategy": "ALIGN_AUDIO_TO_TEXT",
        "audio_prompt": audio_prompt.strip(),
    }


def retry_st_i(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix text-image semantic mismatch by regenerating image selection.
    """
    plan = context.get("semantic_plan", {})

    scene = plan.get("scene_summary", "")
    primary = plan.get("primary_entities", [])
    visual = plan.get("visual_attributes", [])
    style = plan.get("style", [])
    mood = plan.get("mood_emotion", [])
    must_include = plan.get("must_include", [])
    must_avoid = plan.get("must_avoid", [])

    image_prompt_parts = [
        scene,
        f"Primary entities: {', '.join(primary)}" if primary else "",
        f"Visual attributes: {', '.join(visual)}" if visual else "",
        f"Style: {', '.join(style)}" if style else "",
        f"Mood: {', '.join(mood)}" if mood else "",
        f"Must include: {', '.join(must_include)}" if must_include else "",
        f"Must avoid: {', '.join(must_avoid)}" if must_avoid else "",
    ]
    image_prompt = "\n".join([p for p in image_prompt_parts if p]).strip()

    return {
        "regenerate": "image",
        "failed_metric": "st_i",
        "strategy": "ALIGN_IMAGE_TO_TEXT",
        "image_prompt": image_prompt,
    }


def retry_msci(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Global coherence failure: replan + regenerate narrative, image, and audio.
    """
    return {
        "regenerate": "full",
        "failed_metric": "msci",
        "strategy": "REPLAN_AND_REGEN",
    }
