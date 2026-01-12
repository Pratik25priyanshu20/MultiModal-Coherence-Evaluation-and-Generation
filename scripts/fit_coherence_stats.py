import json
from pathlib import Path
import numpy as np

from src.coherence.thresholds import freeze_thresholds

RUNS_DIR = Path("runs")
OUT_FILE = Path("artifacts/coherence_stats.json")

METRIC_KEYS = ["msci", "st_i", "st_a", "si_a"]
metrics = {k: [] for k in METRIC_KEYS}


def extract_scores(obj):
    """
    Recursively search for a dict containing all metric keys.
    Returns the FIRST valid scores dict found.
    """
    if isinstance(obj, dict):
        if all(k in obj for k in METRIC_KEYS):
            return obj
        for v in obj.values():
            found = extract_scores(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = extract_scores(item)
            if found:
                return found
    return None


for run_file in RUNS_DIR.glob("*/logs/run.json"):
    with open(run_file, encoding="utf-8") as f:
        data = json.load(f)

    scores = extract_scores(data)
    if not scores:
        continue

    for k in METRIC_KEYS:
        v = scores.get(k)
        if isinstance(v, (int, float)):
            metrics[k].append(v)

stats = {}
for k, values in metrics.items():
    if len(values) >= 5:
        stats[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

OUT_FILE.parent.mkdir(exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("✅ Saved coherence stats to", OUT_FILE)
print("📊 Collected counts:", {k: len(v) for k, v in metrics.items()})
freeze_thresholds(
    stats_path=str(OUT_FILE),
    notes="Thresholds calibrated on recent runs using adaptive statistics.",
)
print("✅ Saved frozen thresholds to artifacts/thresholds_frozen.json")
