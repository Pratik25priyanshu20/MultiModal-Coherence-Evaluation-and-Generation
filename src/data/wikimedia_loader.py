import json
from pathlib import Path


class WikimediaSampleLoader:
    def __init__(self, base_dir: str = "data/wikimedia"):
        self.base_dir = Path(base_dir)
        self.samples_file = self.base_dir / "samples.json"

    def load(self):
        print("Wikimedia loader starting")
        if not self.samples_file.exists():
            raise FileNotFoundError(f"Missing {self.samples_file}")

        with self.samples_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        elif isinstance(data, dict):
            data = list(data.values())

        samples = []
        for item in data:
            print("Checking item:", item.get("id"))
            samples.append(
                {
                    "id": item["id"],
                    "caption": item["caption"],
                    "image_path": self.base_dir / item["image"],
                    "audio_path": (
                        self.base_dir / item["audio"] if item.get("audio") else None
                    ),
                }
            )

        print("Wikimedia loader finished. Samples:", len(samples))
        return samples
