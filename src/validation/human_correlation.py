"""
Human Correlation Analysis

Analyzes correlation between MSCI scores and human judgments.
This addresses RQ3: "Does MSCI correlate with human judgments of multimodal coherence?"

Key analyses:
- Spearman rank correlation (for ordinal human ratings)
- Pearson correlation (for continuous relationship)
- Per-dimension correlations (text-image, text-audio, image-audio)
- Agreement analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""
    variable1: str
    variable2: str
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    n: int
    ci_lower: float
    ci_upper: float
    significant: bool
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable1": self.variable1,
            "variable2": self.variable2,
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "n": self.n,
            "ci_95": [self.ci_lower, self.ci_upper],
            "significant": self.significant,
            "interpretation": self.interpretation,
        }


class HumanCorrelationAnalyzer:
    """
    Analyzes correlation between MSCI and human judgments.

    RQ3: "Does MSCI correlate with human judgments of multimodal coherence?"
    H0: ρ(MSCI, human) ≤ 0
    H1: ρ(MSCI, human) > 0
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compute_correlation(
        self,
        msci_scores: List[float],
        human_scores: List[float],
        var1_name: str = "MSCI",
        var2_name: str = "Human",
    ) -> CorrelationResult:
        """
        Compute correlation with confidence interval.

        Args:
            msci_scores: MSCI scores
            human_scores: Human coherence scores (normalized to 0-1)
            var1_name: Name for first variable
            var2_name: Name for second variable

        Returns:
            CorrelationResult with all statistics
        """
        if len(msci_scores) != len(human_scores):
            raise ValueError("Score lists must have same length")

        n = len(msci_scores)
        if n < 3:
            return CorrelationResult(
                variable1=var1_name,
                variable2=var2_name,
                spearman_rho=0.0,
                spearman_p=1.0,
                pearson_r=0.0,
                pearson_p=1.0,
                n=n,
                ci_lower=-1.0,
                ci_upper=1.0,
                significant=False,
                interpretation="Insufficient data (N < 3)",
            )

        # Spearman correlation (better for ordinal human ratings)
        spearman = stats.spearmanr(msci_scores, human_scores)

        # Pearson correlation
        pearson = stats.pearsonr(msci_scores, human_scores)

        # Confidence interval for Spearman (using Fisher z-transformation)
        z = np.arctanh(spearman.correlation)
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = np.tanh(z - z_crit * se_z)
        ci_upper = np.tanh(z + z_crit * se_z)

        # Significance (one-tailed test: ρ > 0)
        significant = spearman.pvalue / 2 < self.alpha and spearman.correlation > 0

        # Interpretation
        interpretation = self._interpret_correlation(
            spearman.correlation, spearman.pvalue / 2, significant
        )

        return CorrelationResult(
            variable1=var1_name,
            variable2=var2_name,
            spearman_rho=float(spearman.correlation),
            spearman_p=float(spearman.pvalue),
            pearson_r=float(pearson.statistic),
            pearson_p=float(pearson.pvalue),
            n=n,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            significant=significant,
            interpretation=interpretation,
        )

    def _interpret_correlation(
        self,
        rho: float,
        p_one_tailed: float,
        significant: bool,
    ) -> str:
        """Generate interpretation of correlation."""
        if not significant:
            if p_one_tailed >= self.alpha:
                return f"No significant positive correlation (ρ={rho:.3f}, p={p_one_tailed:.4f})"
            else:
                return f"Significant negative correlation (unexpected; ρ={rho:.3f})"

        abs_rho = abs(rho)
        if abs_rho >= 0.7:
            strength = "strong"
        elif abs_rho >= 0.5:
            strength = "moderate-strong"
        elif abs_rho >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        return f"Significant {strength} positive correlation (ρ={rho:.3f}, p={p_one_tailed:.4f})"

    def analyze_from_human_eval(
        self,
        human_eval_path: Path,
        msci_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze correlation from human evaluation session.

        Args:
            human_eval_path: Path to human evaluation session JSON
            msci_scores: Optional dict of sample_id -> MSCI score

        Returns:
            Comprehensive correlation analysis
        """
        from src.evaluation.human_eval_schema import EvaluationSession

        session = EvaluationSession.load(Path(human_eval_path))

        # Build sample ID -> MSCI mapping from session if not provided
        if msci_scores is None:
            msci_scores = {}
            for sample in session.samples:
                if sample.msci_score is not None:
                    msci_scores[sample.sample_id] = sample.msci_score

        # Collect paired data
        pairs: List[Dict[str, Any]] = []

        for eval in session.evaluations:
            if eval.is_rerating:
                continue
            if eval.sample_id not in msci_scores:
                continue

            pairs.append({
                "sample_id": eval.sample_id,
                "msci": msci_scores[eval.sample_id],
                "human_weighted": eval.weighted_score(),
                "human_overall": eval.overall_coherence / 5.0,  # Normalize
                "human_ti": eval.text_image_coherence / 5.0,
                "human_ta": eval.text_audio_coherence / 5.0,
                "human_ia": eval.image_audio_coherence / 5.0,
            })

        if len(pairs) < 3:
            return {
                "error": "Insufficient paired data",
                "n_pairs": len(pairs),
            }

        # Extract arrays
        msci = [p["msci"] for p in pairs]
        human_weighted = [p["human_weighted"] for p in pairs]
        human_overall = [p["human_overall"] for p in pairs]
        human_ti = [p["human_ti"] for p in pairs]
        human_ta = [p["human_ta"] for p in pairs]
        human_ia = [p["human_ia"] for p in pairs]

        # Compute correlations
        results = {
            "n_pairs": len(pairs),
            "overall_correlation": self.compute_correlation(
                msci, human_weighted, "MSCI", "Human Weighted Score"
            ).to_dict(),
            "overall_rating_correlation": self.compute_correlation(
                msci, human_overall, "MSCI", "Human Overall Rating"
            ).to_dict(),
            "per_dimension": {
                "text_image": self.compute_correlation(
                    msci, human_ti, "MSCI", "Human Text-Image"
                ).to_dict(),
                "text_audio": self.compute_correlation(
                    msci, human_ta, "MSCI", "Human Text-Audio"
                ).to_dict(),
                "image_audio": self.compute_correlation(
                    msci, human_ia, "MSCI", "Human Image-Audio"
                ).to_dict(),
            },
        }

        # RQ3 verdict
        main_corr = results["overall_correlation"]
        results["rq3_verdict"] = self._rq3_verdict(main_corr)

        return results

    def _rq3_verdict(self, correlation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate RQ3 verdict from correlation result."""
        rho = correlation["spearman_rho"]
        p = correlation["spearman_p"]
        significant = correlation["significant"]

        if significant and rho > 0.3:
            verdict = "SUPPORTED"
            explanation = (
                f"MSCI shows significant positive correlation with human judgments "
                f"(ρ={rho:.3f}, p={p/2:.4f}). MSCI is a valid proxy for human-perceived coherence."
            )
        elif significant and rho > 0:
            verdict = "WEAKLY SUPPORTED"
            explanation = (
                f"MSCI shows significant but weak correlation with human judgments "
                f"(ρ={rho:.3f}). MSCI captures some aspects of human-perceived coherence."
            )
        elif not significant and rho > 0:
            verdict = "NOT SUPPORTED"
            explanation = (
                f"No significant correlation between MSCI and human judgments "
                f"(ρ={rho:.3f}, p={p/2:.4f}). MSCI may not reliably reflect human perception."
            )
        else:
            verdict = "CONTRADICTED"
            explanation = (
                f"Unexpected negative correlation (ρ={rho:.3f}). "
                f"MSCI may be inversely related to human perception."
            )

        return {
            "verdict": verdict,
            "explanation": explanation,
            "threshold_met": significant and rho > 0.3,
            "rho": rho,
            "p_value": p / 2,  # One-tailed
        }

    def analyze_disagreements(
        self,
        pairs: List[Dict[str, Any]],
        threshold: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Analyze cases where MSCI and human judgments disagree.

        Args:
            pairs: List of dicts with 'msci' and 'human_weighted' keys
            threshold: Disagreement threshold (normalized)

        Returns:
            Analysis of disagreement patterns
        """
        disagreements = []

        for pair in pairs:
            msci = pair.get("msci", 0)
            human = pair.get("human_weighted", 0)
            diff = msci - human

            if abs(diff) > threshold:
                disagreements.append({
                    "sample_id": pair.get("sample_id"),
                    "msci": msci,
                    "human": human,
                    "difference": diff,
                    "type": "MSCI_overestimates" if diff > 0 else "MSCI_underestimates",
                })

        n_total = len(pairs)
        n_disagree = len(disagreements)

        overestimates = [d for d in disagreements if d["type"] == "MSCI_overestimates"]
        underestimates = [d for d in disagreements if d["type"] == "MSCI_underestimates"]

        return {
            "n_total": n_total,
            "n_disagreements": n_disagree,
            "disagreement_rate": n_disagree / n_total if n_total > 0 else 0,
            "n_overestimates": len(overestimates),
            "n_underestimates": len(underestimates),
            "mean_overestimate": (
                np.mean([d["difference"] for d in overestimates])
                if overestimates else 0
            ),
            "mean_underestimate": (
                np.mean([abs(d["difference"]) for d in underestimates])
                if underestimates else 0
            ),
            "samples": disagreements,
        }

    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive human correlation report.

        Args:
            analysis_results: Results from analyze_from_human_eval
            output_path: Optional path to save report

        Returns:
            Complete correlation report
        """
        report = {
            "analysis_type": "MSCI-Human Correlation Analysis",
            "research_question": "RQ3: Does MSCI correlate with human judgments?",
            "hypothesis": {
                "H0": "ρ(MSCI, human) ≤ 0",
                "H1": "ρ(MSCI, human) > 0",
                "threshold": "ρ > 0.3 for meaningful validity",
            },
            "results": analysis_results,
        }

        # Add recommendations based on results
        verdict = analysis_results.get("rq3_verdict", {})
        if verdict.get("verdict") == "SUPPORTED":
            report["recommendations"] = [
                "MSCI can be used as a proxy for human coherence judgments",
                "Consider using MSCI for automated evaluation at scale",
            ]
        elif verdict.get("verdict") == "WEAKLY SUPPORTED":
            report["recommendations"] = [
                "MSCI provides some signal but should not be sole metric",
                "Consider combining MSCI with other metrics or human spot-checks",
                "Investigate which dimensions MSCI captures well vs poorly",
            ]
        else:
            report["recommendations"] = [
                "MSCI may not reliably reflect human perception",
                "Consider revising MSCI weights or embedding approach",
                "Human evaluation remains necessary for validation",
                "Investigate failure modes to improve MSCI",
            ]

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return report
