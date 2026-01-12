from __future__ import annotations

from typing import Dict

from src.coherence.stats import CoherenceStats


class CoherenceProfiler:
    """
    Assigns coherence bands to a run using learned statistics.
    """

    def __init__(self, stats: CoherenceStats):
        self.stats = stats.stats

    def band(self, metric: str, value: float) -> str:
        s = self.stats.get(metric)
        if not s:
            return "unknown"

        if value < s["p25"]:
            return "low"
        if value < s["p75"]:
            return "medium"
        return "high"

    def profile(self, scores: Dict[str, float]) -> Dict[str, str]:
        """
        Return coherence band per metric.
        """
        return {
            metric: self.band(metric, value)
            for metric, value in scores.items()
            if metric in self.stats
        }
