"""
Extended Prompt Planner Module

A planner that uses 3× the token budget of a single planner in a single call.
This controls for the compute/token difference in Council-Lite (which uses 3 calls).

The ablation question: "Is Council-Lite's benefit from structure or just more tokens?"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import requests
import time

from src.planner.schema import SemanticPlan
from src.planner.validation import validate_semantic_plan_dict
from src.planner.single_planner import PlannerMetrics
from src.utils.json_repair import try_repair_json


EXTENDED_PROMPT_TEMPLATE = """You are an expert multimodal content planner. Your task is to create a detailed,
comprehensive semantic plan for generating coherent multimodal content (text, image, audio).

IMPORTANT: You have an extended budget for this task. Take your time to:
1. Deeply analyze the user's request
2. Consider multiple perspectives and interpretations
3. Ensure semantic consistency across all modalities
4. Provide rich, detailed specifications

Think step by step about:
- What is the core scene/concept?
- What visual elements would best represent this?
- What sounds would naturally occur in this scene?
- How can text describe this scene evocatively?

After your analysis, produce a SINGLE valid JSON object matching this schema:

{
  "scene_summary": string (detailed 2-3 sentence summary),
  "domain": string,

  "core_semantics": {
    "setting": string (specific, detailed setting),
    "time_of_day": string,
    "weather": string,
    "main_subjects": [string] (at least 3 items),
    "actions": [string] (at least 3 items)
  },

  "style_controls": {
    "visual_style": [string] (at least 3 descriptors),
    "color_palette": [string] (at least 4 colors),
    "lighting": [string] (at least 2 aspects),
    "camera": [string],
    "mood_emotion": [string] (at least 3 moods),
    "narrative_tone": [string]
  },

  "image_constraints": {
    "must_include": [string] (at least 3 elements),
    "must_avoid": [string],
    "objects": [string] (detailed object list),
    "environment_details": [string] (at least 3 details),
    "composition": [string]
  },

  "audio_constraints": {
    "audio_intent": [string],
    "sound_sources": [string] (at least 4 distinct sounds),
    "ambience": [string] (detailed ambience description),
    "tempo": string,
    "must_include": [string] (at least 2 sounds),
    "must_avoid": [string]
  },

  "text_constraints": {
    "must_include": [string],
    "must_avoid": [string],
    "keywords": [string] (at least 5 keywords),
    "length": string
  }
}

RULES:
- Be comprehensive and detailed
- Ensure cross-modal consistency (visual elements should have corresponding sounds)
- Every field MUST exist and be properly typed
- Do NOT include markdown formatting
- Output ONLY the JSON object

User request:
{{USER_PROMPT}}

Take your time to create a richly detailed plan:"""


class ExtendedPromptPlannerLLM:
    """
    Extended token budget planner for ablation study.

    Uses 3× the typical token budget in a single LLM call to control for
    the multi-call nature of Council-Lite.

    This helps answer: "Does Council-Lite improve coherence due to its
    multi-agent structure, or simply because it uses more tokens?"
    """

    name = "ExtendedPromptPlanner"

    def __init__(
        self,
        model: str = "qwen2:7b",
        base_url: str = "http://localhost:11434",
        token_multiplier: float = 3.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url
        self.url = f"{base_url}/api/generate"
        self.token_multiplier = token_multiplier
        self.max_retries = max_retries
        self.metrics: Optional[PlannerMetrics] = None
        print(f"ExtendedPromptPlanner using {token_multiplier}x token budget")

    def plan(self, user_prompt: str) -> SemanticPlan:
        """
        Generate a semantic plan with extended token budget.

        Args:
            user_prompt: The user's scene description

        Returns:
            SemanticPlan object
        """
        # Build the extended prompt
        prompt = EXTENDED_PROMPT_TEMPLATE.replace("{{USER_PROMPT}}", user_prompt)

        start_time = time.time()

        # Extended token budget: 3× the standard (2000 → 6000)
        base_tokens = 2000
        extended_tokens = int(base_tokens * self.token_multiplier)

        last_error = None
        last_raw = None

        for attempt in range(1, self.max_retries + 1):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": extended_tokens,
                    "temperature": 0.3,  # Slightly higher for more creative detail
                },
            }

            try:
                resp = requests.post(self.url, json=payload, timeout=300)  # Longer timeout
                resp.raise_for_status()
            except requests.RequestException as e:
                last_error = f"Request failed (attempt {attempt}): {e}"
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")

            raw_text = resp.json().get("response", "").strip()
            last_raw = raw_text

            # Try to extract JSON from the response
            # The model might include thinking before the JSON
            json_start = raw_text.find("{")
            if json_start >= 0:
                raw_text = raw_text[json_start:]

            data = try_repair_json(raw_text)

            if data is not None:
                end_time = time.time()

                # Track metrics
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(str(data).split()) * 1.3

                self.metrics = PlannerMetrics(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    total_tokens=int(input_tokens + output_tokens),
                    llm_calls=1,
                    generation_time_ms=(end_time - start_time) * 1000,
                )

                validate_semantic_plan_dict(data)
                return SemanticPlan(**data)

            last_error = f"JSON parsing failed (attempt {attempt})"
            if attempt < self.max_retries:
                time.sleep(0.5)

        raise ValueError(
            f"[ExtendedPromptPlanner] Could not parse JSON after {self.max_retries} attempts.\n"
            f"Last error: {last_error}\n"
            f"Last raw output (first 1000 chars):\n{last_raw[:1000] if last_raw else 'None'}"
        )

    def get_metrics(self) -> Optional[PlannerMetrics]:
        """Get metrics from the last plan() call."""
        return self.metrics


# Alias
ExtendedPromptPlanner = ExtendedPromptPlannerLLM
