from __future__ import annotations

from src.planner.schema_to_text import plan_to_canonical_text
from .generator import ImageRetrievalGenerator


class PlanToImage:
    def __init__(self):
        self.generator = ImageRetrievalGenerator()

    def run(self, semantic_plan, out_path: str) -> str:
        prompt = plan_to_canonical_text(semantic_plan)
        results = self.generator.retrieve_top_k(prompt, k=1)
        if not results:
            raise RuntimeError("No images available for retrieval.")
        return results[0][0]
