from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

FORBIDDEN_HINTS = [
    "beach",
    "waves",
    "ocean",
    "sea",
    "forest",
    "mountain",
    "fog",
    "snow",
]


def looks_offtopic(image_path: str) -> bool:
    name = Path(image_path).name.lower()
    return any(hint in name for hint in FORBIDDEN_HINTS)


def build_strict_image_prompt(plan: Dict[str, Any]) -> str:
    scene = plan.get("scene_summary", "")
    prim = ", ".join(plan.get("primary_entities", []))
    sec = ", ".join(plan.get("secondary_entities", []))
    vis = ", ".join(plan.get("visual_attributes", []))
    style = ", ".join(plan.get("style", []))

    return (
        f"{scene}. Primary: {prim}. Secondary: {sec}. Visual: {vis}. Style: {style}. "
        "STRICT: urban city street at NIGHT with NEON LIGHTS and RAIN. "
        "DO NOT generate beach, ocean, forest, mountains, snow, village, desert. "
        "Must include: neon signs, wet pavement reflections, rain."
    )


def _generate_candidate(image_gen: Any, prompt: str, attempt: int, max_tries: int) -> str:
    if hasattr(image_gen, "generate"):
        return image_gen.generate(prompt)
    if hasattr(image_gen, "retrieve_top_k"):
        candidates = image_gen.retrieve_top_k(prompt, k=max(max_tries, 1))
        if not candidates:
            raise RuntimeError("No image candidates returned for st_i retry.")
        idx = min(attempt - 1, len(candidates) - 1)
        return candidates[idx][0]
    raise RuntimeError("Image generator does not support generate or retrieve_top_k.")


def retry_st_i(context: Dict[str, Any], max_tries: int = 3) -> Dict[str, Any]:
    plan = context["semantic_plan"]
    image_gen = context["image_generator"]

    prompt = build_strict_image_prompt(plan)

    last_path = context.get("image") or context.get("image_path")
    for attempt in range(1, max_tries + 1):
        new_path = _generate_candidate(image_gen, prompt, attempt, max_tries)
        last_path = new_path
        if not looks_offtopic(new_path):
            context["image"] = new_path
            context["retry"] = {
                "type": "st_i",
                "attempt": attempt,
                "prompt_used": prompt,
                "result": "accepted",
            }
            return context

    context["image"] = last_path
    context["retry"] = {
        "type": "st_i",
        "attempt": max_tries,
        "prompt_used": prompt,
        "result": "all_rejected_offtopic",
    }
    return context
