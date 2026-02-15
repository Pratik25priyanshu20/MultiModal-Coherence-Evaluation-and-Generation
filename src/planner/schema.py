from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class RiskFlag(str, Enum):
    ambiguity = "ambiguity"
    emotional_conflict = "emotional_conflict"
    conflicting_constraints = "conflicting_constraints"


class CoreSemantics(BaseModel):
    setting: str
    time_of_day: str
    weather: str
    main_subjects: List[str]
    actions: List[str]


class StyleControls(BaseModel):
    visual_style: List[str]
    color_palette: List[str]
    lighting: List[str]
    camera: List[str]
    mood_emotion: List[str]
    narrative_tone: List[str]


class ImageConstraints(BaseModel):
    must_include: List[str]
    must_avoid: List[str]
    objects: List[str]
    environment_details: List[str]
    composition: List[str]


class AudioConstraints(BaseModel):
    audio_intent: List[str]
    sound_sources: List[str]
    ambience: List[str]
    tempo: str
    must_include: List[str]
    must_avoid: List[str]


class TextConstraints(BaseModel):
    must_include: List[str]
    must_avoid: List[str]
    keywords: List[str]
    length: str


class UnifiedPlan(BaseModel):
    scene_summary: str
    domain: str
    core_semantics: CoreSemantics
    style_controls: StyleControls
    image_constraints: ImageConstraints
    audio_constraints: AudioConstraints
    text_constraints: TextConstraints
    complexity_level: ComplexityLevel = ComplexityLevel.low
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    notes: str | None = None


SemanticPlan = UnifiedPlan
