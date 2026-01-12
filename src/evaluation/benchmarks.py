from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.coherence.msci import compute_msci_v0
from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.planner.schema_to_text import plan_to_canonical_text

GOLD_PATH = Path("evaluation/gold_dataset/samples.jsonl")


def load_gold_samples() -> List[Dict]:
    with GOLD_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def evaluate_gold_samples():
    embedder = AlignedEmbedder(target_dim=512)

    results = []

    for sample in load_gold_samples():
        plan_text = plan_to_canonical_text(sample["semantic_plan"])
        e_text = embedder.embed_text(plan_text)

        e_image = None
        e_audio = None

        if sample["image_path"]:
            e_image = embedder.embed_image(sample["image_path"])
        if sample["audio_path"]:
            e_audio = embedder.embed_audio(sample["audio_path"])

        msci = compute_msci_v0(
            e_text,
            e_image,
            e_audio,
            include_image_audio=True,
        )

        results.append(
            {
                "id": sample["id"],
                "msci": msci.msci,
                "st_i": msci.st_i,
                "st_a": msci.st_a,
                "si_a": msci.si_a,
                "human": sample["human_criteria"],
            }
        )

    return results
