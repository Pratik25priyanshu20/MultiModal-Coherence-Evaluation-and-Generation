from __future__ import annotations

from pathlib import Path

from src.planner.ollama_llm import OllamaPlannerLLM
from src.planner.schema import SemanticPlan
from src.planner.validation import validate_semantic_plan_dict


class VisualPlannerLLM:
    name = "VisualPlannerOllama"

    def __init__(self, prompt_path: str = "src/planner/prompts/visual.txt"):
        self.llm = OllamaPlannerLLM()
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    def plan(self, user_prompt: str) -> SemanticPlan:
        prompt = self.prompt_template.replace("{{USER_PROMPT}}", user_prompt)
        data = self.llm.generate_json(prompt)
        validate_semantic_plan_dict(data)
        return SemanticPlan(**data)
