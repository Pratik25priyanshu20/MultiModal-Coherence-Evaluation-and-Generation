from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schema import SemanticPlan
from .merge_logic import merge_council_plans, MergeReport
from .unified_planner import UnifiedPlannerLLM


class Planner(Protocol):
    name: str

    def plan(self, user_prompt: str) -> SemanticPlan:
        ...


@dataclass(frozen=True)
class CouncilResult:
    plan_a: SemanticPlan
    plan_b: SemanticPlan
    plan_c: SemanticPlan
    merged_plan: SemanticPlan
    merge_report: MergeReport


class SemanticPlanningCouncil:
    """
    Council-lite (3 planners):
      - A: Core semantics & constraints
      - B: Mood/style emphasis
      - C: Soundscape/audio emphasis
    """

    def __init__(self, planner_a: Planner, planner_b: Planner, planner_c: Planner):
        self.planner_a = planner_a
        self.planner_b = planner_b
        self.planner_c = planner_c

    def run(self, user_prompt: str) -> CouncilResult:
        plan_a = self.planner_a.plan(user_prompt)
        plan_b = self.planner_b.plan(user_prompt)
        plan_c = self.planner_c.plan(user_prompt)

        merged, report = merge_council_plans(plan_a, plan_b, plan_c)
        return CouncilResult(
            plan_a=plan_a,
            plan_b=plan_b,
            plan_c=plan_c,
            merged_plan=merged,
            merge_report=report,
        )


class PlannerCouncil:
    def __init__(self):
        self.planner = UnifiedPlannerLLM()

    def run(self, user_prompt: str) -> SemanticPlan:
        return self.planner.plan(user_prompt)
