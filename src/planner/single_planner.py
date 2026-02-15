"""
Single Planner Module

A single-LLM-call planner for ablation comparison with Council-Lite.
Uses the same prompt template and schema as UnifiedPlanner but
provides explicit token tracking for fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.planner.ollama_llm import OllamaPlannerLLM
from src.planner.schema import SemanticPlan
from src.planner.validation import validate_semantic_plan_dict


@dataclass
class PlannerMetrics:
    """Metrics for tracking planner resource usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    generation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "generation_time_ms": self.generation_time_ms,
        }


class SinglePlannerLLM:
    """
    Single-call semantic planner.

    This is the baseline planner condition for ablation:
    - 1 LLM call
    - Standard token budget
    - Full semantic plan output

    Equivalent to UnifiedPlanner but with explicit metrics.
    """

    name = "SinglePlanner"

    def __init__(
        self,
        prompt_path: str = "src/planner/prompts/unified.txt",
        model: str = "qwen2:7b",
        base_url: str = "http://localhost:11434",
    ):
        self.llm = OllamaPlannerLLM(model=model, base_url=base_url)
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.metrics: Optional[PlannerMetrics] = None

    def plan(self, user_prompt: str) -> SemanticPlan:
        """
        Generate a semantic plan from user prompt.

        Args:
            user_prompt: The user's scene description

        Returns:
            SemanticPlan object
        """
        import time

        # Build prompt
        prompt = self.prompt_template.replace("{{USER_PROMPT}}", user_prompt)

        # Track metrics
        start_time = time.time()

        # Generate plan (single call)
        data = self.llm.generate_json(prompt)

        end_time = time.time()

        # Estimate token counts (rough approximation)
        input_tokens = len(prompt.split()) * 1.3  # ~1.3 tokens per word
        output_tokens = len(str(data).split()) * 1.3

        self.metrics = PlannerMetrics(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(input_tokens + output_tokens),
            llm_calls=1,
            generation_time_ms=(end_time - start_time) * 1000,
        )

        # Validate and return
        validate_semantic_plan_dict(data)
        return SemanticPlan(**data)

    def get_metrics(self) -> Optional[PlannerMetrics]:
        """Get metrics from the last plan() call."""
        return self.metrics


# Alias for consistency
SinglePlanner = SinglePlannerLLM
