from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple

from .schema import (
    SemanticPlan,
    UnifiedPlan,
    CoreSemantics,
    StyleControls,
    ImageConstraints,
    AudioConstraints,
    TextConstraints,
    RiskFlag,
)


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


def _avg3_list(a: List[str], b: List[str], c: List[str]) -> float:
    return (
        _list_agreement(a, b)
        + _list_agreement(a, c)
        + _list_agreement(b, c)
    ) / 3.0


def merge_council_plans(
    plan_a: SemanticPlan,
    plan_b: SemanticPlan,
    plan_c: SemanticPlan,
) -> Tuple[SemanticPlan, MergeReport]:
    """
    Council merge for UnifiedPlan (nested schema).

    Merge strategy:
    - Planner A wins on factual conflicts (entities, setting)
    - Planner B wins on style/mood conflicts
    - Planner C wins on audio conflicts
    - Overlaps are merged via dedup; conflicts are logged
    """
    conflicts: Dict[str, List[str]] = {}
    per_section: Dict[str, float] = {}

    # --- Scene summary ---
    sim_ab = _text_sim(plan_a.scene_summary, plan_b.scene_summary)
    sim_ac = _text_sim(plan_a.scene_summary, plan_c.scene_summary)
    sim_bc = _text_sim(plan_b.scene_summary, plan_c.scene_summary)
    per_section["scene_summary"] = (sim_ab + sim_ac + sim_bc) / 3.0
    scene_summary = plan_a.scene_summary if sim_ab >= 0.6 else plan_b.scene_summary
    if per_section["scene_summary"] < 0.55:
        conflicts["scene_summary"] = [
            plan_a.scene_summary, plan_b.scene_summary, plan_c.scene_summary,
        ]

    domain = plan_a.domain or plan_b.domain or plan_c.domain

    # --- Core semantics (Planner A wins) ---
    cs_a, cs_b, cs_c = plan_a.core_semantics, plan_b.core_semantics, plan_c.core_semantics

    per_section["main_subjects"] = _avg3_list(
        cs_a.main_subjects, cs_b.main_subjects, cs_c.main_subjects
    )
    if per_section["main_subjects"] < 0.5:
        conflicts["main_subjects"] = [
            f"A={cs_a.main_subjects}", f"B={cs_b.main_subjects}", f"C={cs_c.main_subjects}",
        ]

    core_semantics = CoreSemantics(
        setting=cs_a.setting or cs_b.setting or cs_c.setting,
        time_of_day=cs_a.time_of_day or cs_b.time_of_day or cs_c.time_of_day,
        weather=cs_a.weather or cs_b.weather or cs_c.weather,
        main_subjects=cs_a.main_subjects,
        actions=_dedupe_preserve(cs_a.actions + cs_b.actions + cs_c.actions),
    )

    # --- Style controls (Planner B wins) ---
    sc_a, sc_b, sc_c = plan_a.style_controls, plan_b.style_controls, plan_c.style_controls

    per_section["visual_style"] = _avg3_list(
        sc_a.visual_style, sc_b.visual_style, sc_c.visual_style
    )
    per_section["mood_emotion"] = _avg3_list(
        sc_a.mood_emotion, sc_b.mood_emotion, sc_c.mood_emotion
    )

    style_controls = StyleControls(
        visual_style=sc_b.visual_style or sc_a.visual_style or sc_c.visual_style,
        color_palette=_dedupe_preserve(sc_b.color_palette + sc_a.color_palette),
        lighting=_dedupe_preserve(sc_b.lighting + sc_a.lighting),
        camera=sc_b.camera or sc_a.camera or sc_c.camera,
        mood_emotion=_dedupe_preserve(sc_b.mood_emotion + sc_a.mood_emotion),
        narrative_tone=sc_b.narrative_tone or sc_a.narrative_tone or sc_c.narrative_tone,
    )

    # --- Image constraints (Planner A wins) ---
    ic_a, ic_b, ic_c = plan_a.image_constraints, plan_b.image_constraints, plan_c.image_constraints

    per_section["image_must_include"] = _avg3_list(
        ic_a.must_include, ic_b.must_include, ic_c.must_include
    )

    image_constraints = ImageConstraints(
        must_include=_dedupe_preserve(ic_a.must_include + ic_b.must_include),
        must_avoid=_dedupe_preserve(ic_a.must_avoid + ic_b.must_avoid + ic_c.must_avoid),
        objects=_dedupe_preserve(ic_a.objects + ic_b.objects + ic_c.objects),
        environment_details=_dedupe_preserve(
            ic_a.environment_details + ic_b.environment_details
        ),
        composition=ic_b.composition or ic_a.composition or ic_c.composition,
    )

    # --- Audio constraints (Planner C wins) ---
    ac_a, ac_b, ac_c = plan_a.audio_constraints, plan_b.audio_constraints, plan_c.audio_constraints

    per_section["audio_intent"] = _avg3_list(
        ac_a.audio_intent, ac_b.audio_intent, ac_c.audio_intent
    )

    audio_constraints = AudioConstraints(
        audio_intent=_dedupe_preserve(ac_c.audio_intent + ac_b.audio_intent),
        sound_sources=_dedupe_preserve(
            ac_c.sound_sources + ac_b.sound_sources + ac_a.sound_sources
        ),
        ambience=_dedupe_preserve(ac_c.ambience + ac_a.ambience),
        tempo=ac_c.tempo or ac_a.tempo or ac_b.tempo,
        must_include=_dedupe_preserve(ac_c.must_include + ac_a.must_include),
        must_avoid=_dedupe_preserve(ac_c.must_avoid + ac_a.must_avoid + ac_b.must_avoid),
    )

    # --- Text constraints (Planner A wins) ---
    tc_a, tc_b, tc_c = plan_a.text_constraints, plan_b.text_constraints, plan_c.text_constraints

    text_constraints = TextConstraints(
        must_include=_dedupe_preserve(tc_a.must_include + tc_b.must_include),
        must_avoid=_dedupe_preserve(tc_a.must_avoid + tc_b.must_avoid + tc_c.must_avoid),
        keywords=_dedupe_preserve(tc_a.keywords + tc_b.keywords + tc_c.keywords),
        length=tc_a.length or tc_b.length or tc_c.length,
    )

    # --- Agreement score ---
    per_section_avg = sum(per_section.values()) / max(len(per_section), 1)

    risk_flags = _dedupe_preserve(
        [*plan_a.risk_flags, *plan_b.risk_flags, *plan_c.risk_flags]
    )

    notes: List[str] = []
    if per_section_avg < 0.6:
        if RiskFlag.ambiguity not in risk_flags:
            risk_flags.append(RiskFlag.ambiguity)
        notes.append("Low cross-planner agreement; prompt likely ambiguous.")

    if conflicts:
        notes.append(f"Conflicts in: {sorted(list(conflicts.keys()))}")

    weights = {
        "scene_summary": 0.15,
        "main_subjects": 0.20,
        "visual_style": 0.10,
        "mood_emotion": 0.15,
        "audio_intent": 0.15,
        "image_must_include": 0.15,
    }
    agreement_score = 0.0
    for key, weight in weights.items():
        agreement_score += per_section.get(key, 1.0) * weight
    # Normalize for missing keys
    total_weight = sum(weights.values())
    agreement_score /= total_weight

    merged = UnifiedPlan(
        scene_summary=scene_summary,
        domain=domain,
        core_semantics=core_semantics,
        style_controls=style_controls,
        image_constraints=image_constraints,
        audio_constraints=audio_constraints,
        text_constraints=text_constraints,
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
