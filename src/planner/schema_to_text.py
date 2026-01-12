from __future__ import annotations

from typing import Any, Dict, List


def _norm_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    return [str(x).strip()]


def _join(items: List[str], sep: str = ", ") -> str:
    items = [i.strip() for i in items if i and i.strip()]
    return sep.join(items)


def _sent(items: List[str]) -> str:
    """Sentence-ish join. Keeps it readable."""
    items = [i.strip() for i in items if i and i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "; ".join(items)


def plan_to_prompts(plan: Any) -> Dict[str, str]:
    """
    Convert the UnifiedPlanner JSON schema output into STRICT, modality-specific prompts.
    This is the key fix: generators must obey the same semantic contract.

    Returns:
      {
        "text_prompt":  "...",
        "image_prompt": "...",
        "audio_prompt": "...",
        "shared_brief": "..."
      }
    """

    # Accept either pydantic model or dict-like
    if hasattr(plan, "model_dump"):
        p = plan.model_dump()
    elif isinstance(plan, dict):
        p = plan
    else:
        # last resort
        p = dict(plan)

    scene_summary = str(p.get("scene_summary", "")).strip()
    domain = str(p.get("domain", "")).strip()

    # Extract from nested structure (UnifiedPlan schema)
    core_sem = p.get("core_semantics", {})
    style_ctrl = p.get("style_controls", {})
    img_const = p.get("image_constraints", {})
    aud_const = p.get("audio_constraints", {})
    text_const = p.get("text_constraints", {})

    # Primary entities from core_semantics.main_subjects
    primary = _norm_list(core_sem.get("main_subjects") if isinstance(core_sem, dict) else [])
    # Secondary entities (not in schema, but check for compatibility)
    secondary = _norm_list(p.get("secondary_entities", []))
    
    # Visual attributes from style_controls and image_constraints
    visual_style = _norm_list(style_ctrl.get("visual_style", []) if isinstance(style_ctrl, dict) else [])
    color_palette = _norm_list(style_ctrl.get("color_palette", []) if isinstance(style_ctrl, dict) else [])
    lighting = _norm_list(style_ctrl.get("lighting", []) if isinstance(style_ctrl, dict) else [])
    img_objects = _norm_list(img_const.get("objects", []) if isinstance(img_const, dict) else [])
    env_details = _norm_list(img_const.get("environment_details", []) if isinstance(img_const, dict) else [])
    visual_attrs = visual_style + color_palette + lighting + img_objects + env_details
    
    # Style from style_controls
    style = visual_style  # Use visual_style as style
    
    # Mood from style_controls
    mood = _norm_list(style_ctrl.get("mood_emotion", []) if isinstance(style_ctrl, dict) else [])
    
    # Tone from style_controls
    tone = _norm_list(style_ctrl.get("narrative_tone", []) if isinstance(style_ctrl, dict) else [])
    
    # Audio from audio_constraints
    audio_intent = _norm_list(aud_const.get("audio_intent", []) if isinstance(aud_const, dict) else [])
    sound_sources = _norm_list(aud_const.get("sound_sources", []) if isinstance(aud_const, dict) else [])
    ambience = _norm_list(aud_const.get("ambience", []) if isinstance(aud_const, dict) else [])
    audio_elems = audio_intent + sound_sources + ambience
    
    # Must include/avoid from constraints
    img_must_include = _norm_list(img_const.get("must_include", []) if isinstance(img_const, dict) else [])
    img_must_avoid = _norm_list(img_const.get("must_avoid", []) if isinstance(img_const, dict) else [])
    must_include = img_must_include  # Use image constraints for now
    must_avoid = img_must_avoid

    # -------------------------
    # SHARED BRIEF (NO INSTRUCTIONS)
    # -------------------------
    # Important: This is NOT "do X". It's "X is present".
    brief_parts: List[str] = []

    if scene_summary:
        brief_parts.append(scene_summary)

    if domain:
        brief_parts.append(f"Domain: {domain}.")

    if primary:
        brief_parts.append(f"Primary entities: {_join(primary)}.")
    if secondary:
        brief_parts.append(f"Secondary entities: {_join(secondary)}.")

    if visual_attrs:
        brief_parts.append(f"Visual attributes: {_join(visual_attrs)}.")
    if style:
        brief_parts.append(f"Style: {_join(style)}.")
    if mood:
        brief_parts.append(f"Mood/emotion: {_join(mood)}.")
    if tone:
        brief_parts.append(f"Narrative tone: {_join(tone)}.")

    if must_include:
        brief_parts.append(f"Must include: {_join(must_include)}.")
    if must_avoid:
        brief_parts.append(f"Must avoid: {_join(must_avoid)}.")

    shared_brief = " ".join([b.strip() for b in brief_parts if b.strip()])

    # -------------------------
    # TEXT PROMPT (STRICT)
    # -------------------------
    # Goal: stop instruction-echo. We never say “describe” or “generate”.
    # We demand a literal depiction, short, grounded.
    text_lines: List[str] = []
    text_lines.append("Write a vivid, literal description of the exact scene below.")
    text_lines.append("Do not include instructions, bullets, headings, or meta commentary.")
    text_lines.append("Do not mention 'prompt' or 'plan'.")
    text_lines.append("")
    text_lines.append(shared_brief)
    text_lines.append("")
    text_lines.append("Constraints:")
    if must_include:
        text_lines.append(f"- Include: {_join(must_include)}")
    if must_avoid:
        text_lines.append(f"- Avoid: {_join(must_avoid)}")
    text_lines.append("- Length: 3 to 6 sentences.")

    text_prompt = "\n".join(text_lines).strip()

    # -------------------------
    # IMAGE PROMPT (STRICT VISUAL CONTRACT)
    # -------------------------
    # Build a rich, specific prompt for better image retrieval
    img_parts: List[str] = []
    
    # Core scene
    if scene_summary:
        img_parts.append(scene_summary)
    
    # Main subjects (most important for matching)
    if primary:
        img_parts.append(_join(primary))
    
    # Visual details
    if visual_attrs:
        # Use first few most important visual attributes
        key_visuals = visual_attrs[:5]  # Limit to avoid too long prompts
        img_parts.append(_join(key_visuals))
    
    # Style and mood
    if style:
        img_parts.append(_join(style[:2]))  # Limit style tags
    if mood:
        img_parts.append(_join(mood[:2]))  # Limit mood tags
    
    # Core semantics for context
    if isinstance(core_sem, dict):
        setting = core_sem.get("setting", "")
        time_of_day = core_sem.get("time_of_day", "")
        weather = core_sem.get("weather", "")
        if setting:
            img_parts.append(setting)
        if time_of_day:
            img_parts.append(time_of_day)
        if weather:
            img_parts.append(weather)
    
    # Build final prompt - more specific for retrieval
    image_prompt = ", ".join([p for p in img_parts if p]).strip()
    
    # Fallback if empty
    if not image_prompt:
        image_prompt = scene_summary or "scene"

    # -------------------------
    # AUDIO PROMPT (STRICT AUDIO CONTRACT)
    # -------------------------
    # Build a specific, detailed audio prompt for AudioLDM
    aud_parts: List[str] = []
    
    # Core scene context
    if scene_summary:
        aud_parts.append(scene_summary)
    
    # Audio elements (most important)
    if sound_sources:
        aud_parts.append("sounds of " + _join(sound_sources[:4]))  # Limit to avoid too long
    if ambience:
        aud_parts.append("ambient " + _join(ambience[:3]))
    if audio_intent:
        aud_parts.append(_join(audio_intent))
    
    # Context from core semantics
    if isinstance(core_sem, dict):
        setting = core_sem.get("setting", "")
        weather = core_sem.get("weather", "")
        if weather and weather.lower() not in ["clear", "sunny"]:
            aud_parts.append(weather.lower() + " weather sounds")
        if setting:
            aud_parts.append(setting.lower() + " environment")
    
    # Tempo/mood from audio constraints
    if isinstance(aud_const, dict):
        tempo = aud_const.get("tempo", "")
        if tempo:
            aud_parts.append(tempo + " tempo")
    
    # Build final prompt - specific and concise for AudioLDM
    audio_prompt = ", ".join([p for p in aud_parts if p]).strip()
    
    # Fallback if empty
    if not audio_prompt:
        audio_prompt = scene_summary or "ambient soundscape"
    
    # Add quality hints for AudioLDM
    if not audio_prompt.endswith("sound") and not audio_prompt.endswith("audio"):
        audio_prompt += " soundscape"

    return {
        "text_prompt": text_prompt,
        "image_prompt": image_prompt,
        "audio_prompt": audio_prompt,
        "shared_brief": shared_brief,
    }


# Backward compatible function name (if older code imports it)
def plan_to_canonical_text(plan: Any) -> str:
    """
    Legacy: returns the shared brief. Keep this to avoid breaking other imports.
    """
    return plan_to_prompts(plan)["shared_brief"]