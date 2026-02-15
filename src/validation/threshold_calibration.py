"""
MSCI Threshold Calibration

Calibrates MSCI thresholds using ROC analysis to find optimal
classification boundaries for "coherent" vs "incoherent" samples.

Key analyses:
- ROC curve: MSCI as classifier
- AUC (Area Under Curve)
- Optimal threshold via Youden's J statistic
- Precision-Recall analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class CalibrationResult:
    """Result of threshold calibration."""
    optimal_threshold: float
    youden_j: float
    auc: float
    sensitivity_at_optimal: float  # True positive rate
    specificity_at_optimal: float  # True negative rate
    precision_at_optimal: float
    f1_at_optimal: float
    roc_curve: Dict[str, List[float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimal_threshold": self.optimal_threshold,
            "youden_j": self.youden_j,
            "auc": self.auc,
            "sensitivity_at_optimal": self.sensitivity_at_optimal,
            "specificity_at_optimal": self.specificity_at_optimal,
            "precision_at_optimal": self.precision_at_optimal,
            "f1_at_optimal": self.f1_at_optimal,
            "roc_curve": self.roc_curve,
        }


class ThresholdCalibrator:
    """
    Calibrates MSCI thresholds for coherence classification.

    Uses human judgments as the validation target to find optimal
    MSCI threshold that maximizes discrimination between coherent
    and incoherent samples. Note: human judgments serve as the
    best available reference, not absolute ground truth.
    """

    def __init__(self, human_threshold: float = 0.6):
        """
        Initialize calibrator.

        Args:
            human_threshold: Human score above which sample is "coherent"
                            (e.g., 0.6 = 3/5 or higher on Likert scale)
        """
        self.human_threshold = human_threshold

    def compute_roc_curve(
        self,
        msci_scores: List[float],
        human_scores: List[float],
        n_thresholds: int = 100,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute ROC curve points.

        Args:
            msci_scores: MSCI scores (predictor)
            human_scores: Human scores (validation target, normalized 0-1)
            n_thresholds: Number of threshold points

        Returns:
            Tuple of (thresholds, tpr_list, fpr_list)
        """
        # Binarize human scores: 1 = coherent, 0 = incoherent
        y_true = [1 if h >= self.human_threshold else 0 for h in human_scores]

        # Generate thresholds
        min_msci = min(msci_scores)
        max_msci = max(msci_scores)
        thresholds = np.linspace(min_msci - 0.01, max_msci + 0.01, n_thresholds)

        tpr_list = []  # True positive rate (sensitivity)
        fpr_list = []  # False positive rate (1 - specificity)

        for threshold in thresholds:
            # Predict: 1 if MSCI >= threshold
            y_pred = [1 if m >= threshold else 0 for m in msci_scores]

            # Compute confusion matrix elements
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

            # Rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return list(thresholds), tpr_list, fpr_list

    def compute_auc(
        self,
        fpr_list: List[float],
        tpr_list: List[float],
    ) -> float:
        """
        Compute Area Under ROC Curve using trapezoidal rule.

        Args:
            fpr_list: False positive rates
            tpr_list: True positive rates

        Returns:
            AUC value
        """
        # Sort by FPR for proper integration
        sorted_points = sorted(zip(fpr_list, tpr_list))
        sorted_fpr = [p[0] for p in sorted_points]
        sorted_tpr = [p[1] for p in sorted_points]

        # Trapezoidal integration
        auc = 0.0
        for i in range(1, len(sorted_fpr)):
            auc += (sorted_fpr[i] - sorted_fpr[i-1]) * (sorted_tpr[i] + sorted_tpr[i-1]) / 2

        return auc

    def find_optimal_threshold(
        self,
        thresholds: List[float],
        tpr_list: List[float],
        fpr_list: List[float],
    ) -> Tuple[float, float, int]:
        """
        Find optimal threshold using Youden's J statistic.

        J = sensitivity + specificity - 1 = TPR - FPR

        Args:
            thresholds: MSCI threshold values
            tpr_list: True positive rates
            fpr_list: False positive rates

        Returns:
            Tuple of (optimal_threshold, youden_j, optimal_index)
        """
        youden_j = [tpr - fpr for tpr, fpr in zip(tpr_list, fpr_list)]
        optimal_idx = int(np.argmax(youden_j))

        return thresholds[optimal_idx], youden_j[optimal_idx], optimal_idx

    def calibrate(
        self,
        msci_scores: List[float],
        human_scores: List[float],
    ) -> CalibrationResult:
        """
        Perform full threshold calibration.

        Args:
            msci_scores: MSCI scores
            human_scores: Human coherence scores (normalized 0-1)

        Returns:
            CalibrationResult with optimal threshold and metrics
        """
        if len(msci_scores) != len(human_scores):
            raise ValueError("Score lists must have same length")

        if len(msci_scores) < 10:
            raise ValueError("Need at least 10 samples for calibration")

        # Compute ROC curve
        thresholds, tpr_list, fpr_list = self.compute_roc_curve(
            msci_scores, human_scores
        )

        # Compute AUC
        auc = self.compute_auc(fpr_list, tpr_list)

        # Find optimal threshold
        optimal_threshold, youden_j, opt_idx = self.find_optimal_threshold(
            thresholds, tpr_list, fpr_list
        )

        # Compute metrics at optimal threshold
        sensitivity = tpr_list[opt_idx]
        specificity = 1 - fpr_list[opt_idx]

        # Precision and F1 at optimal threshold
        y_true = [1 if h >= self.human_threshold else 0 for h in human_scores]
        y_pred = [1 if m >= optimal_threshold else 0 for m in msci_scores]

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return CalibrationResult(
            optimal_threshold=optimal_threshold,
            youden_j=youden_j,
            auc=auc,
            sensitivity_at_optimal=sensitivity,
            specificity_at_optimal=specificity,
            precision_at_optimal=precision,
            f1_at_optimal=f1,
            roc_curve={
                "thresholds": thresholds,
                "tpr": tpr_list,
                "fpr": fpr_list,
            },
        )

    def calibrate_from_human_eval(
        self,
        human_eval_path: Path,
    ) -> CalibrationResult:
        """
        Calibrate from human evaluation session.

        Args:
            human_eval_path: Path to human evaluation session JSON

        Returns:
            CalibrationResult
        """
        from src.evaluation.human_eval_schema import EvaluationSession

        session = EvaluationSession.load(Path(human_eval_path))

        msci_scores = []
        human_scores = []

        # Build sample_id -> msci mapping
        sample_msci = {s.sample_id: s.msci_score for s in session.samples if s.msci_score}

        for eval in session.evaluations:
            if eval.is_rerating:
                continue
            if eval.sample_id not in sample_msci:
                continue

            msci_scores.append(sample_msci[eval.sample_id])
            human_scores.append(eval.weighted_score())

        return self.calibrate(msci_scores, human_scores)

    def evaluate_thresholds(
        self,
        msci_scores: List[float],
        human_scores: List[float],
        thresholds: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate classification performance at multiple thresholds.

        Args:
            msci_scores: MSCI scores
            human_scores: Human scores
            thresholds: Thresholds to evaluate

        Returns:
            Dict mapping threshold to performance metrics
        """
        y_true = [1 if h >= self.human_threshold else 0 for h in human_scores]
        results = {}

        for threshold in thresholds:
            y_pred = [1 if m >= threshold else 0 for m in msci_scores]

            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

            accuracy = (tp + tn) / len(y_true) if y_true else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[f"{threshold:.3f}"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
            }

        return results

    def generate_report(
        self,
        calibration_result: CalibrationResult,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate calibration report.

        Args:
            calibration_result: Result from calibrate()
            output_path: Optional path to save report

        Returns:
            Complete calibration report
        """
        report = {
            "analysis_type": "MSCI Threshold Calibration",
            "purpose": "Find optimal MSCI threshold for coherence classification",
            "method": "ROC analysis with Youden's J optimization",
            "human_threshold": self.human_threshold,
            "results": calibration_result.to_dict(),
        }

        # AUC interpretation
        auc = calibration_result.auc
        if auc >= 0.9:
            auc_interp = "Excellent discrimination"
        elif auc >= 0.8:
            auc_interp = "Good discrimination"
        elif auc >= 0.7:
            auc_interp = "Acceptable discrimination"
        elif auc >= 0.6:
            auc_interp = "Poor discrimination"
        else:
            auc_interp = "Failed discrimination (no better than chance)"

        report["interpretation"] = {
            "auc_interpretation": auc_interp,
            "optimal_threshold": calibration_result.optimal_threshold,
            "threshold_usage": (
                f"Samples with MSCI >= {calibration_result.optimal_threshold:.3f} "
                f"should be classified as 'coherent'"
            ),
            "expected_performance": {
                "sensitivity": f"{calibration_result.sensitivity_at_optimal:.1%} of coherent samples correctly identified",
                "specificity": f"{calibration_result.specificity_at_optimal:.1%} of incoherent samples correctly rejected",
                "precision": f"{calibration_result.precision_at_optimal:.1%} of 'coherent' predictions are correct",
            },
        }

        # Recommendations
        if auc >= 0.7:
            report["recommendations"] = [
                f"Use MSCI threshold of {calibration_result.optimal_threshold:.3f} for binary classification",
                "MSCI provides meaningful discrimination between coherent and incoherent samples",
            ]
        else:
            report["recommendations"] = [
                "MSCI alone may not reliably distinguish coherent from incoherent samples",
                "Consider combining MSCI with other metrics",
                "Human evaluation may be necessary for borderline cases",
            ]

        if output_path:
            # Exclude full ROC curve from saved file to reduce size
            report_to_save = report.copy()
            if "roc_curve" in report_to_save.get("results", {}):
                report_to_save["results"] = report_to_save["results"].copy()
                del report_to_save["results"]["roc_curve"]
                report_to_save["results"]["roc_curve_note"] = "Excluded from file (100 points)"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(report_to_save, f, indent=2, ensure_ascii=False)

        return report
