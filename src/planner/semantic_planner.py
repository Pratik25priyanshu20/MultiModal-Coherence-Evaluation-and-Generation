from typing import Any, Dict

from src.planner.semantic_plan import SemanticPlan, Scene
from src.llm.llm_router import get_llm


SYSTEM_PROMPT = """
You are a semantic planner.

Convert the user prompt into a STRICT JSON object matching this schema:

{
  "scene": {
    "setting": "urban | natural | indoor",
    "time": "day | night | sunset",
    "weather": "clear | rain | fog | wind"
  },
  "visual_elements": [string, ...],
  "audio_elements": [string, ...],
  "mood": "calm | tense | futuristic | melancholic",
  "motion": "static | slow | dynamic"
}

Rules:
- Output JSON ONLY
- No explanations
- No extra keys
- Use best semantic judgment
"""


def generate_semantic_plan(prompt: str) -> SemanticPlan:
    llm = get_llm()
    data: Dict[str, Any] = llm.generate_json(
        f"{SYSTEM_PROMPT}\n\nUser prompt: {prompt}"
    )

    try:
        scene = Scene(**data["scene"])
        plan = SemanticPlan(
            scene=scene,
            visual_elements=data["visual_elements"],
            audio_elements=data["audio_elements"],
            mood=data["mood"],
            motion=data["motion"],
        )
    except Exception as e:
        raise ValueError(f"Planner output does not match schema:\n{data}") from e

    return plan
