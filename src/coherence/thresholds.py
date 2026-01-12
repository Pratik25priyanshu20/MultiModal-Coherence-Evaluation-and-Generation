from datetime import datetime
from pathlib import Path
import json


class AdaptiveThresholds:
    def __init__(self, stats_path: str = "artifacts/coherence_stats.json"):
        self.stats_path = Path(stats_path)
        self.stats = self._load_stats()

    def _load_stats(self):
        if not self.stats_path.exists():
            raise RuntimeError("Coherence stats file not found")

        with open(self.stats_path, encoding="utf-8") as f:
            return json.load(f)

    def bands(self, metric: str):
        """
        Returns (good, weak) thresholds.
        """
        s = self.stats[metric]
        mean = s["mean"]
        std = s["std"]

        good = mean - 0.5 * std
        weak = mean - 1.0 * std

        return good, weak

    def classify_value(self, metric: str, value: float):
        good, weak = self.bands(metric)

        if value >= good:
            return "GOOD"
        if value >= weak:
            return "WEAK"
        return "FAIL"


def freeze_thresholds(
    stats_path: str = "artifacts/coherence_stats.json",
    out_path: str = "artifacts/thresholds_frozen.json",
    version: str = "v1.0",
    notes: str | None = None,
) -> dict:
    thresholds = AdaptiveThresholds(stats_path=stats_path)
    metrics = list(thresholds.stats.keys())

    frozen_thresholds = {
        metric: {
            "good": thresholds.bands(metric)[0],
            "weak": thresholds.bands(metric)[1],
            "mean": thresholds.stats[metric]["mean"],
            "std": thresholds.stats[metric]["std"],
            "count": thresholds.stats[metric].get("count"),
        }
        for metric in metrics
    }

    policy = {
        "HIGH_COHERENCE": "accept",
        "LOCAL_MODALITY_WEAKNESS": "accept_with_note",
        "MODALITY_FAILURE": "regenerate",
        "GLOBAL_FAILURE": "regenerate",
    }

    default_notes = "Thresholds calibrated using adaptive statistics."
    frozen = {
        "version": version,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "metrics": metrics,
        "policy": policy,
        "thresholds": frozen_thresholds,
        "notes": notes or default_notes,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen
