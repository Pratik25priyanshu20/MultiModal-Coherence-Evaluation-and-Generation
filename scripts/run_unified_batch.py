from pathlib import Path
import json
from statistics import mean

from src.pipeline.generate_and_evaluate import generate_and_evaluate

PROMPTS = [
    "A quiet beach at night with gentle waves and distant wind",
    "A rainy neon-lit city street at night with reflections on wet pavement",
    "A calm forest at dawn with birdsong and soft mist",
    "A futuristic city skyline at sunset with flying vehicles",
]

N_RUNS_PER_PROMPT = 10

OUT_DIR = Path("runs/unified_batch_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_results = []

for prompt in PROMPTS:
    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("=" * 60)

    prompt_results = []

    for i in range(N_RUNS_PER_PROMPT):
        print(f"\n--- Run {i+1}/{N_RUNS_PER_PROMPT} ---")

        bundle = generate_and_evaluate(
            prompt=prompt,
            out_dir=str(OUT_DIR),
            use_ollama=True,
        )

        record = {
            "prompt": prompt,
            "run_id": bundle.run_id,
            "scores": bundle.scores,
            "classification": bundle.coherence.get("classification"),
        }

        prompt_results.append(record)
        all_results.append(record)

    # prompt-level summary
    msci_values = [r["scores"]["msci"] for r in prompt_results]
    print(
        f"\nMSCI summary for prompt:\n"
        f"avg={mean(msci_values):.4f}, "
        f"min={min(msci_values):.4f}, "
        f"max={max(msci_values):.4f}"
    )

# Save everything
with (OUT_DIR / "raw_results.json").open("w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved raw results to {OUT_DIR / 'raw_results.json'}")