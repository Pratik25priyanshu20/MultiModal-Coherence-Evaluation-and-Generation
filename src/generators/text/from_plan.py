from __future__ import annotations

from src.planner.schema_to_text import plan_to_canonical_text
from .generator import TextGenerator


class PlanToText:
    def __init__(self):
        self.generator = TextGenerator()

    def run(self, semantic_plan) -> str:
        prompt = (
            "Write a vivid but concise description based on the following plan:\n\n"
            f"{plan_to_canonical_text(semantic_plan)}\n\n"
            "Description:"
        )
        return self.generator.generate(prompt).text
