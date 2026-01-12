from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.planner.schema import SemanticPlan

OUT_PATH = Path("evaluation/gold_dataset/samples.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_sample(sample: dict) -> None:
    with OUT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_pilot() -> None:
    """
    Manually curated pilot samples.
    Start small. Quality > quantity.
    """

    plan = SemanticPlan(
        scene_summary="A calm forest road at dawn with light fog",
        domain="mixed_general",
        primary_entities=["forest road"],
        secondary_entities=["trees", "fog"],
        visual_attributes=["soft lighting", "green tones", "early morning"],
        style="cinematic",
        mood_emotion=["calm"],
        narrative_tone="peaceful",
        audio_intent=["ambient", "natural"],
        audio_elements=["birds chirping", "soft wind"],
        must_include=["forest road"],
        must_avoid=["vehicles"],
    )

    sample = {
        "id": str(uuid.uuid4()),
        "source": "wikimedia",
        "domain": "mixed_general",
        "text_prompt": "A peaceful forest road at dawn with mist and birds",
        "semantic_plan": plan.model_dump(),
        "image_path": "data/processed/images/forest_road.jpg",
        "audio_path": "data/processed/audio/forest_ambient.wav",
        "tags": {
            "scene_type": "nature",
            "mood": ["calm"],
            "environment": "forest",
        },
        "human_criteria": {
            "entities_present": True,
            "mood_matches": True,
            "audio_matches_environment": True,
            "no_major_contradictions": True,
        },
        "notes": "Clear low-risk baseline sample",
    }

    write_sample(sample)
    print("✅ Gold pilot sample written.")


if __name__ == "__main__":
    build_pilot()
