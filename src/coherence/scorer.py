from src.coherence.classifier import CoherenceClassifier
from src.coherence.thresholds import AdaptiveThresholds


class CoherenceScorer:
    def __init__(self):
        self.thresholds = AdaptiveThresholds()
        self.classifier = CoherenceClassifier(self.thresholds)

    def score(self, scores: dict, global_drift: bool):
        weights = {"msci": 0.35, "st_i": 0.20, "st_a": 0.20, "si_a": 0.25}
        valid_scores = {k: v for k, v in scores.items() if v is not None}

        total = sum(weights[k] for k in valid_scores if k in weights)
        if total > 0:
            base = sum(valid_scores[k] * weights[k] for k in valid_scores if k in weights) / total
        else:
            base = 0.0

        result = self.classifier.classify(valid_scores, global_drift)

        total_penalty = sum(result.penalties.values())
        final = max(0.0, base - total_penalty)

        return {
            "base_score": round(base, 4),
            "final_score": round(final, 4),
            "penalties": result.penalties,
            "classification": {
                "label": result.label,
                "reason": result.reason,
                "weakest_metric": result.weakest_metric,
            },
        }
