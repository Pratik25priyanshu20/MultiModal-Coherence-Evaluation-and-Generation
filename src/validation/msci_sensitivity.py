"""
MSCI Sensitivity Analysis

Tests whether MSCI is sensitive to controlled semantic perturbations.
This addresses RQ1: "Is MSCI sensitive to controlled semantic perturbations?"

Key tests:
- Perturbation gradient: 0%, 25%, 50%, 75%, 100% semantic mismatch
- Expected: monotonic MSCI decrease with increasing perturbation
- If not monotonic: MSCI may be unreliable
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class PerturbationLevel:
    """A single perturbation level result."""
    level: float  # 0.0 = baseline, 1.0 = complete mismatch
    label: str
    msci_scores: List[float] = field(default_factory=list)
    n_samples: int = 0

    @property
    def mean_msci(self) -> float:
        """Mean MSCI at this perturbation level."""
        return np.mean(self.msci_scores) if self.msci_scores else 0.0

    @property
    def std_msci(self) -> float:
        """Standard deviation of MSCI."""
        return np.std(self.msci_scores, ddof=1) if len(self.msci_scores) > 1 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "label": self.label,
            "n_samples": len(self.msci_scores),
            "mean_msci": self.mean_msci,
            "std_msci": self.std_msci,
            "min_msci": min(self.msci_scores) if self.msci_scores else None,
            "max_msci": max(self.msci_scores) if self.msci_scores else None,
        }


@dataclass
class PerturbationGradient:
    """Results from a perturbation gradient analysis."""
    levels: List[PerturbationLevel]
    is_monotonic: bool
    spearman_correlation: float
    spearman_p: float
    linear_slope: float
    r_squared: float
    sensitivity_score: float  # 0-1, how well MSCI tracks perturbation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "levels": [l.to_dict() for l in self.levels],
            "is_monotonic": self.is_monotonic,
            "spearman_correlation": self.spearman_correlation,
            "spearman_p": self.spearman_p,
            "linear_slope": self.linear_slope,
            "r_squared": self.r_squared,
            "sensitivity_score": self.sensitivity_score,
        }


class MSCISensitivityAnalyzer:
    """
    Analyzes MSCI sensitivity to semantic perturbations.

    RQ1: "Is MSCI sensitive to controlled semantic perturbations?"
    H0: MSCI(baseline) = MSCI(perturbed)
    H1: MSCI(baseline) > MSCI(perturbed)
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def analyze_perturbation_gradient(
        self,
        baseline_scores: List[float],
        perturbed_scores_by_level: Dict[float, List[float]],
    ) -> PerturbationGradient:
        """
        Analyze MSCI response to perturbation gradient.

        Args:
            baseline_scores: MSCI scores for unperturbed samples
            perturbed_scores_by_level: Dict mapping perturbation level (0-1) to MSCI scores

        Returns:
            PerturbationGradient with analysis results
        """
        # Build perturbation levels
        levels = [
            PerturbationLevel(
                level=0.0,
                label="baseline",
                msci_scores=baseline_scores,
            )
        ]

        for level, scores in sorted(perturbed_scores_by_level.items()):
            levels.append(
                PerturbationLevel(
                    level=level,
                    label=f"{int(level * 100)}% perturbation",
                    msci_scores=scores,
                )
            )

        # Check monotonicity (MSCI should decrease as perturbation increases)
        means = [l.mean_msci for l in levels]
        is_monotonic = all(means[i] >= means[i + 1] for i in range(len(means) - 1))

        # Compute Spearman correlation between perturbation level and MSCI
        all_levels = []
        all_scores = []
        for level in levels:
            for score in level.msci_scores:
                all_levels.append(level.level)
                all_scores.append(score)

        if len(all_scores) >= 3:
            spearman_result = stats.spearmanr(all_levels, all_scores)
            spearman_rho = spearman_result.correlation
            spearman_p = spearman_result.pvalue

            # Linear regression for slope
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                all_levels, all_scores
            )
            r_squared = r_value ** 2
        else:
            spearman_rho = 0.0
            spearman_p = 1.0
            slope = 0.0
            r_squared = 0.0

        # Sensitivity score: combination of correlation strength and monotonicity
        sensitivity = 0.0
        if spearman_rho < 0:  # Negative correlation expected
            sensitivity = abs(spearman_rho)
            if not is_monotonic:
                sensitivity *= 0.5  # Penalty for non-monotonicity

        return PerturbationGradient(
            levels=levels,
            is_monotonic=is_monotonic,
            spearman_correlation=spearman_rho,
            spearman_p=spearman_p,
            linear_slope=slope,
            r_squared=r_squared,
            sensitivity_score=sensitivity,
        )

    def paired_sensitivity_test(
        self,
        baseline_scores: List[float],
        perturbed_scores: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Perform paired t-test for sensitivity.

        Tests H0: MSCI(baseline) = MSCI(perturbed)
        vs    H1: MSCI(baseline) > MSCI(perturbed)

        Args:
            baseline_scores: MSCI scores for baseline (same prompts)
            perturbed_scores: MSCI scores for perturbed (same prompts)
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        from src.experiments.statistical_analysis import paired_ttest, compute_effect_size

        if len(baseline_scores) != len(perturbed_scores):
            raise ValueError("Baseline and perturbed must have same length for paired test")

        # One-sided t-test (baseline > perturbed)
        result = paired_ttest(
            baseline_scores,
            perturbed_scores,
            alpha=alpha,
            alternative="greater",
        )

        # Effect size
        effect_size = compute_effect_size(baseline_scores, perturbed_scores, paired=True)

        # Descriptive stats
        baseline_mean = np.mean(baseline_scores)
        perturbed_mean = np.mean(perturbed_scores)
        mean_drop = baseline_mean - perturbed_mean
        percent_drop = (mean_drop / baseline_mean * 100) if baseline_mean > 0 else 0

        return {
            "test": "paired_t_test_one_sided",
            "hypothesis": "H1: MSCI(baseline) > MSCI(perturbed)",
            "n": len(baseline_scores),
            "baseline_mean": baseline_mean,
            "perturbed_mean": perturbed_mean,
            "mean_drop": mean_drop,
            "percent_drop": percent_drop,
            "t_statistic": result.statistic,
            "p_value": result.p_value,
            "effect_size_d": effect_size,
            "significant": result.significant,
            "interpretation": self._interpret_sensitivity(
                result.significant, effect_size, percent_drop
            ),
        }

    def _interpret_sensitivity(
        self,
        significant: bool,
        effect_size: float,
        percent_drop: float,
    ) -> str:
        """Generate interpretation of sensitivity test."""
        if not significant:
            return "MSCI is NOT significantly sensitive to this perturbation (H0 not rejected)"

        if effect_size > 0.8:
            strength = "highly"
        elif effect_size > 0.5:
            strength = "moderately"
        else:
            strength = "weakly"

        return (
            f"MSCI is {strength} sensitive to perturbation "
            f"(d={effect_size:.2f}, {percent_drop:.1f}% drop)"
        )

    def analyze_from_experiment_results(
        self,
        results_path: Path,
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity from experiment results JSON.

        Args:
            results_path: Path to experiment_results.json

        Returns:
            Sensitivity analysis results
        """
        with Path(results_path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract MSCI scores by condition
        raw_results = data.get("raw_results", [])
        scores_by_condition: Dict[str, List[float]] = {}

        for result in raw_results:
            if not result.get("success"):
                continue
            condition = result.get("condition", "")
            msci = result.get("scores", {}).get("msci")
            if msci is not None and condition:
                if condition not in scores_by_condition:
                    scores_by_condition[condition] = []
                scores_by_condition[condition].append(msci)

        # Analyze perturbations
        analyses = {}

        # Find baseline conditions
        baseline_conditions = [c for c in scores_by_condition if "baseline" in c]

        for baseline_cond in baseline_conditions:
            mode = baseline_cond.replace("_baseline", "")
            baseline_scores = scores_by_condition[baseline_cond]

            # Find corresponding perturbation conditions
            for cond, scores in scores_by_condition.items():
                if cond == baseline_cond:
                    continue
                if not cond.startswith(mode):
                    continue

                # Perform sensitivity test
                n = min(len(baseline_scores), len(scores))
                if n >= 3:
                    test_result = self.paired_sensitivity_test(
                        baseline_scores[:n], scores[:n]
                    )
                    analyses[f"{baseline_cond}_vs_{cond}"] = test_result

        return {
            "source": str(results_path),
            "conditions_analyzed": list(scores_by_condition.keys()),
            "sensitivity_tests": analyses,
            "summary": self._summarize_sensitivity(analyses),
        }

    def _summarize_sensitivity(self, analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize sensitivity results."""
        if not analyses:
            return {"conclusion": "No sensitivity tests performed"}

        n_significant = sum(1 for a in analyses.values() if a.get("significant"))
        n_total = len(analyses)

        avg_effect = np.mean([
            a.get("effect_size_d", 0) for a in analyses.values()
        ])
        avg_drop = np.mean([
            a.get("percent_drop", 0) for a in analyses.values()
        ])

        if n_significant == n_total and avg_effect > 0.5:
            verdict = "STRONG SENSITIVITY: MSCI reliably detects perturbations"
        elif n_significant > n_total / 2:
            verdict = "MODERATE SENSITIVITY: MSCI detects most perturbations"
        elif n_significant > 0:
            verdict = "WEAK SENSITIVITY: MSCI detects some perturbations"
        else:
            verdict = "NO SENSITIVITY: MSCI fails to detect perturbations"

        return {
            "n_tests": n_total,
            "n_significant": n_significant,
            "sensitivity_rate": n_significant / n_total if n_total > 0 else 0,
            "average_effect_size": avg_effect,
            "average_percent_drop": avg_drop,
            "verdict": verdict,
        }

    def generate_report(
        self,
        gradient_result: Optional[PerturbationGradient] = None,
        sensitivity_tests: Optional[Dict[str, Dict]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive sensitivity report.

        Args:
            gradient_result: Optional perturbation gradient analysis
            sensitivity_tests: Optional dict of sensitivity test results
            output_path: Optional path to save report

        Returns:
            Complete sensitivity report
        """
        report = {
            "analysis_type": "MSCI Sensitivity Analysis",
            "research_question": "RQ1: Is MSCI sensitive to controlled semantic perturbations?",
            "hypothesis": {
                "H0": "MSCI(baseline) = MSCI(perturbed)",
                "H1": "MSCI(baseline) > MSCI(perturbed)",
            },
        }

        if gradient_result:
            report["gradient_analysis"] = gradient_result.to_dict()
            report["gradient_verdict"] = (
                "PASS: MSCI shows monotonic decrease with perturbation"
                if gradient_result.is_monotonic and gradient_result.sensitivity_score > 0.5
                else "FAIL: MSCI does not reliably track perturbation level"
            )

        if sensitivity_tests:
            report["sensitivity_tests"] = sensitivity_tests
            report["summary"] = self._summarize_sensitivity(sensitivity_tests)

        # Overall RQ1 verdict
        verdicts = []
        if gradient_result:
            verdicts.append(gradient_result.sensitivity_score > 0.5)
        if sensitivity_tests:
            summary = report.get("summary", {})
            verdicts.append(summary.get("sensitivity_rate", 0) > 0.5)

        if verdicts:
            report["rq1_verdict"] = (
                "SUPPORTED: MSCI is sensitive to semantic perturbations"
                if all(verdicts)
                else "PARTIALLY SUPPORTED: Mixed sensitivity results"
                if any(verdicts)
                else "NOT SUPPORTED: MSCI is not reliably sensitive"
            )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return report
