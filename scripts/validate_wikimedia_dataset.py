import json
from pathlib import Path

base = Path("data/wikimedia")

with (base / "samples.json").open("r", encoding="utf-8") as f:
    samples = json.load(f)

errors = 0
for s in samples:
    if not (base / s["image"]).exists():
        print("Missing image:", s["image"])
        errors += 1
    if s.get("audio") and not (base / s["audio"]).exists():
        print("Missing audio:", s["audio"])
        errors += 1

print(f"Checked {len(samples)} samples, errors: {errors}")
