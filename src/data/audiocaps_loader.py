import json
from pathlib import Path


class AudioCapsLoader:
    def __init__(self, base_dir: str = "data/audiocaps"):
        self.base_dir = Path(base_dir)
        self.samples_file = self.base_dir / "samples.json"

    def load(self):
        if not self.samples_file.exists():
            raise FileNotFoundError(f"Missing {self.samples_file}")

        with self.samples_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data:
            audio_path = self.base_dir / item["audio"]
            if not audio_path.exists():
                print(f"⚠️ Missing audio file: {audio_path}")
                continue

            samples.append(
                {
                    "id": item["id"],
                    "caption": item["caption"],
                    "image_path": None,
                    "audio_path": audio_path,
                }
            )

        print(f"AudioCaps loader finished. Samples: {len(samples)}")
        return samples