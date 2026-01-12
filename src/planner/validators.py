from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .schema import SemanticPlan, RiskFlag, ComplexityLevel


@dataclass(frozen=True)
class PlanValidationResult:
    ok: bool
    warnings: List[str]
    risk_flags: List[RiskFlag]
    complexity_level: ComplexityLevel


def validate_plan(plan: SemanticPlan) -> PlanValidationResult:
    warnings: List[str] = []
    risk_flags: List[RiskFlag] = list(plan.risk_flags)

    if len(plan.primary_entities) > 5:
        warnings.append("Too many primary_entities; may reduce coherence.")
        if RiskFlag.ambiguity not in risk_flags:
            risk_flags.append(RiskFlag.ambiguity)

    if len(plan.must_include) > 7:
        warnings.append("Too many must_include constraints; generation may become brittle.")
        if plan.complexity_level == ComplexityLevel.low:
            complexity = ComplexityLevel.medium
        else:
            complexity = plan.complexity_level
    else:
        complexity = plan.complexity_level

    lower_moods = {m.lower() for m in plan.mood_emotion}
    if "calm" in lower_moods and "tense" in lower_moods:
        warnings.append("Conflicting mood_emotion signals (calm + tense).")
        if RiskFlag.emotional_conflict not in risk_flags:
            risk_flags.append(RiskFlag.emotional_conflict)

    overlap = {x.lower() for x in plan.must_include}.intersection(
        {y.lower() for y in plan.must_avoid}
    )
    if overlap:
        warnings.append(
            "Conflicting constraints: "
            f"{sorted(overlap)} appear in both must_include and must_avoid."
        )
        if RiskFlag.conflicting_constraints not in risk_flags:
            risk_flags.append(RiskFlag.conflicting_constraints)

    return PlanValidationResult(
        ok=True,
        warnings=warnings,
        risk_flags=risk_flags,
        complexity_level=complexity,
    )
