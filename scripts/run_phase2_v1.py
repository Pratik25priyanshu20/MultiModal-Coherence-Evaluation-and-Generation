from __future__ import annotations

import json
import os

from src.generators.audio.generator import AudioGenerator
from src.generators.image.generator import ImageRetrievalGenerator
from src.generators.text.generator import TextGenerator
from src.orchestrator.request_flow import Orchestrator
from src.planner.council import PlannerCouncil


def main() -> None:
    council = PlannerCouncil()

    text_gen = TextGenerator(model_name="distilgpt2", max_new_tokens=120)
    image_gen = ImageRetrievalGenerator(index_path="data/embeddings/image_index.npz")
    audio_gen = AudioGenerator(device="cpu")

    orch = Orchestrator(
        council=council,
        text_gen=text_gen,
        image_gen=image_gen,
        audio_gen=audio_gen,
        msci_threshold=0.42,
        max_attempts=4,
        runs_dir="runs",
    )

    prompt = os.getenv(
        "USER_PROMPT_OVERRIDE",
        "A rainy neon-lit city street at night with reflections on wet pavement.",
    )
    out = orch.run(prompt)

    if out.narrative_structured:
        print("\nNARRATIVE (STRUCTURED JSON):")
        print(json.dumps(out.narrative_structured, indent=2, ensure_ascii=False))
        print("\nNARRATIVE (PARAGRAPH):")
        print(out.narrative_text)
    else:
        print("\nNARRATIVE:")
        print(out.narrative_text)

    if out.coherence:
        print("\nCOHERENCE (PHASE-3B):")
        print(json.dumps(out.coherence, indent=2, ensure_ascii=False))

    print("\n=== PHASE-2 V1 OUTPUT ===")
    print("run_id:", out.run_id)
    print("attempts:", out.attempts)
    print("scores:", out.scores)
    print("drift:", out.drift)
    print("\n# paths")
    print("image_path:", out.image_path)
    print("audio_path:", out.audio_path)
    print("run_json:", f"runs/{out.run_id}/logs/run.json")


if __name__ == "__main__":
    main()
