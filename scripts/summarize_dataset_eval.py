import json
from collections import Counter
from pathlib import Path

import numpy as np

INPUT_FILE = Path("runs/dataset_eval/wikimedia_results.json")
OUTPUT_FILE = Path("artifacts/dataset_summary.json")

with INPUT_FILE.open("r", encoding="utf-8") as f:
    results = json.load(f)

base_scores = [r["base_score"] for r in results]
final_scores = [r["final_score"] for r in results]
labels = [r["classification"] for r in results]

summary = {
    "num_samples": len(results),
    "base_score_mean": float(np.mean(base_scores)),
    "base_score_std": float(np.std(base_scores)),
    "final_score_mean": float(np.mean(final_scores)),
    "final_score_std": float(np.std(final_scores)),
    "classification_counts": dict(Counter(labels)),
    "classification_percentages": {
        k: v / len(labels) for k, v in Counter(labels).items()
    },
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved dataset summary to", OUTPUT_FILE)
