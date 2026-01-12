import json
import random
from pathlib import Path
from typing import Optional


class LaionPromptLoader:
    def __init__(
        self,
        base_dir: str = "data/laion",
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Load LAION captions from samples.json.
        
        Args:
            base_dir: Directory containing samples.json
            max_samples: Maximum number of samples to return (None = all)
            seed: Random seed for deterministic shuffling (if shuffle=True)
            shuffle: Whether to shuffle samples before capping
        """
        self.base_dir = Path(base_dir)
        self.samples_file = self.base_dir / "samples.json"
        self.max_samples = max_samples
        self.seed = seed
        self.shuffle = shuffle

    def load(self):
        if not self.samples_file.exists():
            raise FileNotFoundError(f"Missing {self.samples_file}")

        with self.samples_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data:
            samples.append({
                "id": item["id"],
                "caption": item["caption"],
                "image_path": None,
                "audio_path": None
            })

        # Shuffle if requested
        if self.shuffle:
            rng = random.Random(self.seed) if self.seed is not None else random
            rng.shuffle(samples)

        # Apply max_samples cap
        if self.max_samples is not None:
            samples = samples[:self.max_samples]

        print(f"LAION loader finished. Samples: {len(samples)}")
        return samples
