"""
Phase 4: Normalized scorer using calibration config.

This module provides normalized scoring using calibration parameters
derived from perturbation experiments.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any


class NormalizedScorer:
    """
    Score normalization and calibration based on perturbation distributions.
    """
    
    def __init__(self, calibration_config_path: Optional[str] = None):
        """
        Initialize with calibration config.
        
        Args:
            calibration_config_path: Path to calibration_config.json.
                                    If None, uses default location.
        """
        if calibration_config_path is None:
            calibration_config_path = "runs/calibration/calibration_config.json"
        
        self.config_path = Path(calibration_config_path)
        self.calibration = self._load_calibration()
    
    def _load_calibration(self) -> Dict[str, Any]:
        """Load calibration config."""
        if not self.config_path.exists():
            # Return defaults if calibration not available
            return {
                "normalization": {},
                "thresholds": {},
                "separation_analysis": {},
            }
        
        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def normalize_score(self, metric: str, raw_value: float) -> float:
        """
        Normalize a raw similarity score using calibration parameters.
        
        Uses z-score normalization: (value - mean) / std
        """
        norm_params = self.calibration.get("normalization", {}).get(metric)
        if not norm_params:
            # No calibration available, return raw value
            return raw_value
        
        mean_val = norm_params.get("mean", 0.0)
        std_val = norm_params.get("std", 1.0)
        
        if std_val < 1e-6:
            return raw_value
        
        normalized = (raw_value - mean_val) / std_val
        return float(normalized)
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize all scores in a scores dict."""
        normalized = {}
        for metric, value in scores.items():
            if value is not None:
                normalized[metric] = self.normalize_score(metric, value)
            else:
                normalized[metric] = None
        return normalized
    
    def get_threshold(self, metric: str, level: str = "low") -> Optional[float]:
        """
        Get calibrated threshold for a metric.
        
        Args:
            metric: Metric name (msci, st_i, st_a, si_a)
            level: Threshold level ("low" or "very_low")
        """
        thresholds = self.calibration.get("thresholds", {}).get(metric, {})
        return thresholds.get(level)
    
    def classify_score(self, metric: str, raw_value: float) -> str:
        """
        Classify a raw score using calibrated thresholds.
        
        Returns: "GOOD", "WEAK", or "FAIL"
        """
        low_threshold = self.get_threshold(metric, "low")
        very_low_threshold = self.get_threshold(metric, "very_low")
        
        if low_threshold is None or very_low_threshold is None:
            # Fallback to simple heuristic
            if raw_value > 0.3:
                return "GOOD"
            elif raw_value > 0.1:
                return "WEAK"
            else:
                return "FAIL"
        
        if raw_value >= low_threshold:
            return "GOOD"
        elif raw_value >= very_low_threshold:
            return "WEAK"
        else:
            return "FAIL"
    
    def is_calibrated(self) -> bool:
        """Check if calibration config is loaded and valid."""
        return bool(self.calibration.get("normalization"))


def apply_normalization_to_results(
    results: Dict[str, Any],
    calibration_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply normalization to a results dict (e.g., from raw_results.json).
    
    Adds normalized_scores and calibrated_classification fields.
    """
    scorer = NormalizedScorer(calibration_config_path)
    
    scores = results.get("scores", {})
    normalized_scores = scorer.normalize_scores(scores)
    
    # Classify using calibrated thresholds
    classifications = {}
    for metric, value in scores.items():
        if value is not None:
            classifications[metric] = scorer.classify_score(metric, value)
    
    return {
        **results,
        "normalized_scores": normalized_scores,
        "calibrated_classification": classifications,
    }
