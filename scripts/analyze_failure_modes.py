import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

INPUT_FILE = Path("runs/dataset_eval/wikimedia_results.json")
OUTPUT_FILE = Path("artifacts/failure_analysis.json")

with INPUT_FILE.open("r", encoding="utf-8") as f:
    results = json.load(f)

weakest = Counter()
scores_by_metric = defaultdict(list)
failed_samples = []

for r in results:
    wm = r.get("weakest_metric")
    if wm:
        weakest[wm] += 1

    for k in ["msci", "st_i", "st_a", "si_a"]:
        if k in r["scores"]:
            scores_by_metric[k].append(r["scores"][k])

    if r["classification"] != "HIGH_COHERENCE":
        failed_samples.append(
            {
                "id": r["id"],
                "classification": r["classification"],
                "weakest_metric": r.get("weakest_metric"),
            }
        )

analysis = {
    "weakest_metric_counts": dict(weakest),
    "average_scores_by_metric": {
        k: float(np.mean(v)) for k, v in scores_by_metric.items()
    },
    "failed_samples": failed_samples,
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2)

print("Saved failure analysis to", OUTPUT_FILE)
