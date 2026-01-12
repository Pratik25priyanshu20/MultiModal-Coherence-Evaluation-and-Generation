from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RUNS_DIR = Path("runs")


def main() -> None:
    labels = Counter()
    total = 0

    for run_file in RUNS_DIR.glob("*/logs/run.json"):
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        coherence = data.get("final", {}).get("coherence", {})
        classification = coherence.get("classification", {})
        label = classification.get("label")
        if not label:
            continue

        labels[label] += 1
        total += 1

    if total == 0:
        print("No runs found.")
        return

    accepted = labels.get("HIGH_COHERENCE", 0) + labels.get("LOCAL_MODALITY_WEAKNESS", 0)
    regen = labels.get("MODALITY_FAILURE", 0) + labels.get("GLOBAL_FAILURE", 0)

    print("Run count:", total)
    print("Label counts:", dict(labels))
    print("Acceptance rate:", round(accepted / total, 3))
    print("Regen rate:", round(regen / total, 3))


if __name__ == "__main__":
    main()
