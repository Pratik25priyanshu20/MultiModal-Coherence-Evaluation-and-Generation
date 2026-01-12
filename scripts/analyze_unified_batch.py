import json
from pathlib import Path
from collections import Counter

import numpy as np

DATA_PATH = Path("runs/unified_batch/raw_results.json")

with DATA_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

scores = {k: [] for k in ["msci", "st_i", "st_a", "si_a"]}
labels = []

for r in data:
    for k in scores:
        scores[k].append(r["scores"][k])
    labels.append(r["classification"]["label"])

report = {
    "num_runs": len(data),
    "mean_scores": {k: float(np.mean(v)) for k, v in scores.items()},
    "std_scores": {k: float(np.std(v)) for k, v in scores.items()},
    "classification_counts": dict(Counter(labels)),
    "classification_percentages": {
        k: v / len(data) for k, v in Counter(labels).items()
    },
}

OUT_FILE = Path("artifacts/unified_batch_report.json")
OUT_FILE.parent.mkdir(exist_ok=True)

with OUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Saved aggregate report to", OUT_FILE)
