from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple

from .schema import SemanticPlan, RiskFlag


@dataclass(frozen=True)
class MergeReport:
    agreement_score: float
    per_section_agreement: Dict[str, float]
    conflicts: Dict[str, List[str]]
    notes: str


def _list_agreement(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    aset = {x.lower().strip() for x in a if x.strip()}
    bset = {x.lower().strip() for x in b if x.strip()}
    if not aset and not bset:
        return 1.0
    inter = len(aset.intersection(bset))
    union = len(aset.union(bset))
    return inter / union if union else 1.0


def _text_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, (a or "").strip().lower(), (b or "").strip().lower()).ratio()


def merge_council_plans(
    plan_a: SemanticPlan,
    plan_b: SemanticPlan,
    plan_c: SemanticPlan,
) -> Tuple[SemanticPlan, MergeReport]:
    """
    Council merge rules (V1):
    - Factual: entities/constraints -> Planner A wins on conflicts
    - Style/mood: Planner B wins on conflicts
    - Audio: Planner C wins on conflicts
    - Overlaps are merged; conflicts are logged; uncertainty -> risk_flags + notes
    """

    conflicts: Dict[str, List[str]] = {}
    per_section: Dict[str, float] = {}

    sim_ab = _text_sim(plan_a.scene_summary, plan_b.scene_summary)
    sim_ac = _text_sim(plan_a.scene_summary, plan_c.scene_summary)
    sim_bc = _text_sim(plan_b.scene_summary, plan_c.scene_summary)
    per_section["scene_summary"] = (sim_ab + sim_ac + sim_bc) / 3.0

    scene_summary = plan_a.scene_summary if sim_ab >= 0.6 else plan_b.scene_summary
    if per_section["scene_summary"] < 0.55:
        conflicts["scene_summary"] = [
            plan_a.scene_summary,
            plan_b.scene_summary,
            plan_c.scene_summary,
        ]

    domain = plan_a.domain or plan_b.domain or plan_c.domain

    per_section["primary_entities"] = (
        _list_agreement(plan_a.primary_entities, plan_b.primary_entities)
        + _list_agreement(plan_a.primary_entities, plan_c.primary_entities)
        + _list_agreement(plan_b.primary_entities, plan_c.primary_entities)
    ) / 3.0
    if per_section["primary_entities"] < 0.5:
        conflicts["primary_entities"] = [
            f"A={plan_a.primary_entities}",
            f"B={plan_b.primary_entities}",
            f"C={plan_c.primary_entities}",
        ]
    primary_entities = plan_a.primary_entities

    secondary_entities = _dedupe_preserve(
        plan_a.secondary_entities
        + plan_b.secondary_entities
        + plan_c.secondary_entities
    )

    per_section["visual_attributes"] = (
        _list_agreement(plan_a.visual_attributes, plan_b.visual_attributes)
        + _list_agreement(plan_a.visual_attributes, plan_c.visual_attributes)
        + _list_agreement(plan_b.visual_attributes, plan_c.visual_attributes)
    ) / 3.0
    visual_attributes = _dedupe_preserve(
        plan_a.visual_attributes + plan_b.visual_attributes
    )

    per_section["style"] = (
        _list_agreement(plan_a.style, plan_b.style)
        + _list_agreement(plan_a.style, plan_c.style)
        + _list_agreement(plan_b.style, plan_c.style)
    ) / 3.0
    style = plan_b.style or plan_a.style or plan_c.style
    if per_section["style"] < 0.5:
        conflicts["style"] = [plan_a.style, plan_b.style, plan_c.style]

    per_section["mood_emotion"] = (
        _list_agreement(plan_a.mood_emotion, plan_b.mood_emotion)
        + _list_agreement(plan_a.mood_emotion, plan_c.mood_emotion)
        + _list_agreement(plan_b.mood_emotion, plan_c.mood_emotion)
    ) / 3.0
    mood_emotion = _dedupe_preserve(plan_b.mood_emotion + plan_a.mood_emotion)
    narrative_tone = plan_b.narrative_tone or plan_a.narrative_tone or plan_c.narrative_tone

    per_section["audio_intent"] = (
        _list_agreement(plan_a.audio_intent, plan_b.audio_intent)
        + _list_agreement(plan_a.audio_intent, plan_c.audio_intent)
        + _list_agreement(plan_b.audio_intent, plan_c.audio_intent)
    ) / 3.0
    audio_intent = _dedupe_preserve(plan_c.audio_intent + plan_b.audio_intent)
    audio_elements = _dedupe_preserve(
        plan_c.audio_elements + plan_b.audio_elements + plan_a.audio_elements
    )

    per_section["must_include"] = (
        _list_agreement(plan_a.must_include, plan_b.must_include)
        + _list_agreement(plan_a.must_include, plan_c.must_include)
        + _list_agreement(plan_b.must_include, plan_c.must_include)
    ) / 3.0
    per_section["must_avoid"] = (
        _list_agreement(plan_a.must_avoid, plan_b.must_avoid)
        + _list_agreement(plan_a.must_avoid, plan_c.must_avoid)
        + _list_agreement(plan_b.must_avoid, plan_c.must_avoid)
    ) / 3.0

    must_include = _dedupe_preserve(plan_a.must_include + plan_b.must_include)
    must_avoid = _dedupe_preserve(
        plan_a.must_avoid + plan_b.must_avoid + plan_c.must_avoid
    )

    per_section_avg = sum(per_section.values()) / max(len(per_section), 1)
    risk_flags = _dedupe_preserve(
        [*plan_a.risk_flags, *plan_b.risk_flags, *plan_c.risk_flags]
    )

    notes: List[str] = []
    if per_section_avg < 0.6:
        if RiskFlag.ambiguity not in risk_flags:
            risk_flags.append(RiskFlag.ambiguity)
        notes.append("Low cross-planner agreement; prompt likely ambiguous or underspecified.")

    if conflicts:
        notes.append(f"Conflicts detected in fields: {sorted(list(conflicts.keys()))}")

    weights = {
        "scene_summary": 0.10,
        "primary_entities": 0.20,
        "visual_attributes": 0.10,
        "style": 0.05,
        "mood_emotion": 0.15,
        "audio_intent": 0.15,
        "must_include": 0.15,
        "must_avoid": 0.10,
    }
    agreement_score = 0.0
    for key, weight in weights.items():
        agreement_score += per_section.get(key, 1.0) * weight

    merged = SemanticPlan(
        scene_summary=scene_summary,
        domain=domain,
        primary_entities=primary_entities,
        secondary_entities=secondary_entities,
        visual_attributes=visual_attributes,
        style=style,
        mood_emotion=mood_emotion,
        narrative_tone=narrative_tone,
        audio_intent=audio_intent,
        audio_elements=audio_elements,
        must_include=must_include,
        must_avoid=must_avoid,
        complexity_level=plan_a.complexity_level,
        risk_flags=risk_flags,
        notes=" ".join(notes) if notes else None,
    )

    report = MergeReport(
        agreement_score=float(round(agreement_score, 4)),
        per_section_agreement={k: float(round(v, 4)) for k, v in per_section.items()},
        conflicts=conflicts,
        notes=merged.notes or "",
    )
    return merged, report


def _dedupe_preserve(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in items:
        key = str(item).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
