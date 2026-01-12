import json
import sys
from pathlib import Path

from src.data.audiocaps_loader import AudioCapsLoader
from src.data.laion_loader import LaionPromptLoader
from src.data.wikimedia_loader import WikimediaSampleLoader
from src.coherence.coherence_engine import CoherenceEngine

OUTPUT_DIR = Path("runs/dataset_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "wikimedia"
    if dataset == "wikimedia":
        loader = WikimediaSampleLoader()
    elif dataset == "audiocaps":
        loader = AudioCapsLoader()
    elif dataset == "laion":
        loader = LaionPromptLoader()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    samples = loader.load()
    print(f"Loaded {len(samples)} samples from {dataset}")

    engine = CoherenceEngine()
    results = []

    for idx, sample in enumerate(samples, start=1):
        print(f"Evaluating sample: {idx} ({sample['id']})")

        result = engine.evaluate(
            text=sample["caption"],
            image_path=str(sample["image_path"]) if sample["image_path"] else None,
            audio_path=str(sample["audio_path"]) if sample["audio_path"] else None,
        )
        scores = result["scores"]
        coherence = result["coherence"]

        results.append(
            {
                "id": sample["id"],
                "prompt": sample["caption"],
                "scores": scores,
                "base_score": coherence["base_score"],
                "final_score": coherence["final_score"],
                "classification": coherence["classification"]["label"],
                "weakest_metric": coherence["classification"]["weakest_metric"],
            }
        )

    out_file = OUTPUT_DIR / f"{dataset}_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
