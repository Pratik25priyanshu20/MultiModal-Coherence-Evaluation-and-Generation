from __future__ import annotations

from typing import Any, Dict


def build_audio_prompt_from_plan(plan: Dict[str, Any]) -> str:
    scene = plan.get("scene_summary", "")
    prim = ", ".join(plan.get("primary_entities", []))
    sec = ", ".join(plan.get("secondary_entities", []))

    return (
        f"Ambient audio for scene: {scene}. "
        f"Primary elements: {prim}. Secondary: {sec}. "
        "STRICT: match the visual environment. "
        "If urban -> city ambience, traffic, rain. "
        "If forest -> wind, birds, leaves. "
        "NO mismatch."
    )


def retry_si_a(context: Dict[str, Any], max_tries: int = 2) -> Dict[str, Any]:
    plan = context["semantic_plan"]
    audio_gen = context["audio_generator"]
    out_path = context.get("audio_path")

    if not out_path:
        raise RuntimeError("Missing audio_path for si_a retry.")

    prompt = build_audio_prompt_from_plan(plan)

    last_audio = context.get("audio") or out_path
    for attempt in range(1, max_tries + 1):
        result = audio_gen.generate(prompt, out_path)
        last_audio = result.audio_path

        context["audio"] = result.audio_path
        context["audio_backend"] = result.backend
        context["retry"] = {
            "type": "si_a",
            "strategy": "ALIGN_AUDIO_TO_IMAGE",
            "attempt": attempt,
            "prompt_used": prompt,
        }
        return context

    context["audio"] = last_audio
    return context
