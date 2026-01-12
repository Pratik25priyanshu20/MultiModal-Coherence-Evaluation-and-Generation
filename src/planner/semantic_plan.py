from dataclasses import dataclass
from typing import List, Literal


SceneSetting = Literal["urban", "natural", "indoor"]
TimeOfDay = Literal["day", "night", "sunset"]
Weather = Literal["clear", "rain", "fog", "wind"]
Mood = Literal["calm", "tense", "futuristic", "melancholic"]
Motion = Literal["static", "slow", "dynamic"]


@dataclass
class Scene:
    setting: SceneSetting
    time: TimeOfDay
    weather: Weather


@dataclass
class SemanticPlan:
    scene: Scene
    visual_elements: List[str]
    audio_elements: List[str]
    mood: Mood
    motion: Motion