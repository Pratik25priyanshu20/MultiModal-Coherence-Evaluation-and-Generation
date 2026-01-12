from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

METRICS = ["st_i", "st_a", "si_a", "msci"]


class CoherenceStats:
    """
    Learns statistical thresholds from previous run logs.
    """

    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, float]] = {}

    def fit(self, runs: List[Dict]) -> None:
        """
        Learn mean, std, and percentiles from past runs.
        """
        for metric in METRICS:
            values = [
                r["scores"][metric]
                for r in runs
                if metric in r.get("scores", {})
            ]

            if not values:
                continue

            arr = np.array(values)

            self.stats[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
            }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CoherenceStats":
        obj = cls()
        obj.stats = json.loads(path.read_text())
        return obj
