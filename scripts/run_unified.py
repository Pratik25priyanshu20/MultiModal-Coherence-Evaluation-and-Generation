import os

from src.pipeline.generate_and_evaluate import generate_and_evaluate


if __name__ == "__main__":
    prompt = os.environ.get(
        "PROMPT",
        "A rainy neon-lit city street at night with reflections on wet pavement",
    )

    bundle = generate_and_evaluate(prompt=prompt, out_dir="runs/unified", use_ollama=True)

    print("\n=== UNIFIED RUN COMPLETE ===")
    print("run_id:", bundle.run_id)
    print("image:", bundle.image_path)
    print("audio:", bundle.audio_path)
    print("scores:", bundle.scores)
    print("classification:", bundle.coherence.get("classification"))
    print("bundle.json:", f"runs/unified/{bundle.run_id}/bundle.json")
