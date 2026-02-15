from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

from src.llm.llm_router import get_llm
from src.narrative.schema import NarrativeOutput


class NarrativeGenerator:
    def __init__(self, prompt_path: str = "src/narrative/prompts/narrative.txt"):
        self.llm = get_llm()
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def generate(self, plan_json: Dict[str, Any]) -> NarrativeOutput:
        prompt = self.prompt_template.replace(
            "{{PLAN_JSON}}",
            json.dumps(plan_json, ensure_ascii=False, indent=2),
        )
        raw = self.llm.generate_json(prompt)
        return self._validate_raw(raw)

    def repair_visual_description(
        self,
        plan_json: Dict[str, Any],
        image_path: str | None = None,
    ) -> NarrativeOutput:
        """
        Regenerate narrative with a strict, concrete visual_description.
        """
        image_hint = f"Image path: {image_path}\n" if image_path else ""
        prompt = (
            "You are a STRICT JSON generator.\n"
            "Rules:\n"
            "- Output ONLY valid JSON\n"
            "- No explanations or markdown\n"
            "- Rewrite visual_description using ONLY concrete, visible objects, "
            "attributes, and spatial layout (e.g., wet pavement, neon signs, "
            "tall buildings, reflections).\n"
            "- Avoid mood-only language unless tied to a visible cause.\n\n"
            f"{image_hint}"
            f"Semantic Plan:\n{json.dumps(plan_json, ensure_ascii=False, indent=2)}\n\n"
            "Return EXACTLY this JSON schema:\n"
            "{\n"
            '  "visual_description": "string",\n'
            '  "audio_description": "string",\n'
            '  "combined_scene": "string",\n'
            '  "style_summary": "string",\n'
            '  "mood_summary": "string"\n'
            "}\n"
        )
        raw = self.llm.generate_json(prompt)
        return self._validate_raw(raw)

    def _validate_raw(self, raw: Any) -> NarrativeOutput:
        required_keys = {
            "visual_description",
            "audio_description",
            "combined_scene",
            "style_summary",
            "mood_summary",
        }
        if not isinstance(raw, dict) or not required_keys.issubset(raw.keys()):
            raise RuntimeError(
                "[NarrativeGenerator] Model did not return required schema.\n"
                f"Keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}\n"
                f"Raw: {str(raw)[:400]}"
            )

        try:
            return NarrativeOutput(**raw)
        except ValidationError as e:
            raise RuntimeError(
                "[NarrativeGenerator] Narrative schema validation failed"
            ) from e
