from dataclasses import dataclass
from typing import Dict

from src.coherence.thresholds import AdaptiveThresholds


@dataclass
class CoherenceResult:
    label: str
    reason: str
    weakest_metric: str | None
    penalties: Dict[str, float]


class CoherenceClassifier:
    def __init__(self, thresholds: AdaptiveThresholds):
        self.t = thresholds

    def classify(self, scores: Dict[str, float], global_drift: bool) -> CoherenceResult:
        if not scores:
            return CoherenceResult(
                label="UNKNOWN",
                reason="Insufficient metrics for classification",
                weakest_metric=None,
                penalties={},
            )

        statuses = {m: self.t.classify_value(m, v) for m, v in scores.items()}

        fails = [m for m, s in statuses.items() if s == "FAIL"]
        weaks = [m for m, s in statuses.items() if s == "WEAK"]

        penalties: Dict[str, float] = {}
        weakest = min(scores, key=lambda m: scores[m]) if scores else None

        if statuses.get("msci") == "FAIL" or len(fails) >= 2:
            penalties["global_drift"] = 0.18
            label = "GLOBAL_FAILURE"
            reason = "Semantic alignment failed across modalities"
        elif len(fails) == 1:
            penalties["weak_modality"] = 0.12
            label = "MODALITY_FAILURE"
            reason = f"Failure in modality: {fails[0]}"
            weakest = fails[0]
        elif len(weaks) >= 1:
            penalties["weak_modality"] = 0.06
            label = "LOCAL_MODALITY_WEAKNESS"
            reason = f"Weak coherence in modality: {weaks[0]}"
        else:
            label = "HIGH_COHERENCE"
            reason = "Strong cross-modal semantic agreement"

        return CoherenceResult(
            label=label,
            reason=reason,
            weakest_metric=weakest,
            penalties=penalties,
        )
