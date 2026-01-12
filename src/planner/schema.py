from __future__ import annotations

from typing import List

from pydantic import BaseModel


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


SemanticPlan = UnifiedPlan
