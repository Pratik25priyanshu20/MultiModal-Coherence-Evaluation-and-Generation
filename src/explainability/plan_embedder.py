from typing import Any, Dict, List

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.planner.canonical_text import plan_to_text
from src.planner.semantic_plan import SemanticPlan


class PlanEmbedder:
    def __init__(self, embedder: AlignedEmbedder | None = None):
        self.embedder = embedder or AlignedEmbedder(target_dim=512)

    def embed(self, plan: Any):
        text = _plan_to_text(plan)
        return self.embedder.embed_text(text)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _join(items: List[str]) -> str:
    cleaned = [i.strip() for i in items if i and i.strip()]
    return ", ".join(cleaned)


def _plan_to_text(plan: Any) -> str:
    if isinstance(plan, SemanticPlan) or hasattr(plan, "scene"):
        return plan_to_text(plan)

    if hasattr(plan, "model_dump"):
        data = plan.model_dump()
    elif isinstance(plan, dict):
        data = plan
    else:
        data = {}

    scene_summary = str(data.get("scene_summary", "")).strip()
    core = data.get("core_semantics", {}) or {}
    style = data.get("style_controls", {}) or {}
    image_c = data.get("image_constraints", {}) or {}
    audio_c = data.get("audio_constraints", {}) or {}
    text_c = data.get("text_constraints", {}) or {}

    parts: List[str] = []
    if scene_summary:
        parts.append(scene_summary)

    setting = str(core.get("setting", "")).strip()
    time_of_day = str(core.get("time_of_day", "")).strip()
    weather = str(core.get("weather", "")).strip()
    if setting or time_of_day or weather:
        parts.append(
            f"Setting: {', '.join([p for p in [setting, time_of_day, weather] if p])}."
        )

    main_subjects = _as_list(core.get("main_subjects"))
    actions = _as_list(core.get("actions"))
    if main_subjects:
        parts.append(f"Subjects: {_join(main_subjects)}.")
    if actions:
        parts.append(f"Actions: {_join(actions)}.")

    visual_style = _as_list(style.get("visual_style"))
    color_palette = _as_list(style.get("color_palette"))
    lighting = _as_list(style.get("lighting"))
    camera = _as_list(style.get("camera"))
    mood = _as_list(style.get("mood_emotion"))
    tone = _as_list(style.get("narrative_tone"))
    if visual_style or color_palette or lighting or camera:
        parts.append(
            f"Visual style: {_join(visual_style + color_palette + lighting + camera)}."
        )
    if mood:
        parts.append(f"Mood: {_join(mood)}.")
    if tone:
        parts.append(f"Tone: {_join(tone)}.")

    objects = _as_list(image_c.get("objects"))
    environment = _as_list(image_c.get("environment_details"))
    composition = _as_list(image_c.get("composition"))
    img_include = _as_list(image_c.get("must_include"))
    img_avoid = _as_list(image_c.get("must_avoid"))
    if objects or environment or composition:
        parts.append(
            f"Image constraints: {_join(objects + environment + composition)}."
        )
    if img_include:
        parts.append(f"Image must include: {_join(img_include)}.")
    if img_avoid:
        parts.append(f"Image must avoid: {_join(img_avoid)}.")

    audio_intent = _as_list(audio_c.get("audio_intent"))
    sound_sources = _as_list(audio_c.get("sound_sources"))
    ambience = _as_list(audio_c.get("ambience"))
    tempo = str(audio_c.get("tempo", "")).strip()
    aud_include = _as_list(audio_c.get("must_include"))
    aud_avoid = _as_list(audio_c.get("must_avoid"))
    if audio_intent or sound_sources or ambience:
        parts.append(
            f"Audio intent: {_join(audio_intent + sound_sources + ambience)}."
        )
    if tempo:
        parts.append(f"Audio tempo: {tempo}.")
    if aud_include:
        parts.append(f"Audio must include: {_join(aud_include)}.")
    if aud_avoid:
        parts.append(f"Audio must avoid: {_join(aud_avoid)}.")

    keywords = _as_list(text_c.get("keywords"))
    text_include = _as_list(text_c.get("must_include"))
    text_avoid = _as_list(text_c.get("must_avoid"))
    length = str(text_c.get("length", "")).strip()
    if keywords:
        parts.append(f"Text keywords: {_join(keywords)}.")
    if text_include:
        parts.append(f"Text must include: {_join(text_include)}.")
    if text_avoid:
        parts.append(f"Text must avoid: {_join(text_avoid)}.")
    if length:
        parts.append(f"Text length: {length}.")

    return " ".join([p.strip() for p in parts if p.strip()])
