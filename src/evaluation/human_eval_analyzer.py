"""
Human Evaluation Analysis Module

Analyzes human evaluation data to compute:
- Intra-rater reliability (Cohen's kappa for self-agreement)
- Inter-rater reliability (Krippendorff's alpha for multi-rater agreement)
- Descriptive statistics
- Correlation with MSCI scores (aggregated across raters)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from scipy import stats

from src.evaluation.human_eval_schema import (
    EvaluationSession,
    HumanEvaluation,
    ReliabilityMetrics,
)


def compute_cohens_kappa(ratings1: List[int], ratings2: List[int]) -> float:
    """
    Compute Cohen's kappa for two sets of ratings.

    Args:
        ratings1: First set of ratings
        ratings2: Second set of ratings (same samples, different time)

    Returns:
        Cohen's kappa coefficient
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating lists must have the same length")

    n = len(ratings1)
    if n == 0:
        return 0.0

    # Create confusion matrix
    categories = sorted(set(ratings1) | set(ratings2))
    k = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    confusion = np.zeros((k, k))
    for r1, r2 in zip(ratings1, ratings2):
        confusion[cat_to_idx[r1], cat_to_idx[r2]] += 1

    # Compute observed agreement
    p_o = np.trace(confusion) / n

    # Compute expected agreement by chance
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    p_e = np.sum(row_sums * col_sums) / (n * n)

    # Cohen's kappa
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def compute_weighted_kappa(
    ratings1: List[int], ratings2: List[int], weights: str = "quadratic"
) -> float:
    """
    Compute weighted Cohen's kappa for ordinal data.

    Args:
        ratings1: First set of ratings
        ratings2: Second set of ratings
        weights: "linear" or "quadratic" weighting scheme

    Returns:
        Weighted kappa coefficient
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating lists must have the same length")

    n = len(ratings1)
    if n == 0:
        return 0.0

    # Get all categories
    all_ratings = set(ratings1) | set(ratings2)
    min_cat, max_cat = min(all_ratings), max(all_ratings)
    categories = list(range(min_cat, max_cat + 1))
    k = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Create confusion matrix
    confusion = np.zeros((k, k))
    for r1, r2 in zip(ratings1, ratings2):
        confusion[cat_to_idx[r1], cat_to_idx[r2]] += 1

    # Create weight matrix
    weight_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "linear":
                weight_matrix[i, j] = abs(i - j) / (k - 1)
            else:  # quadratic
                weight_matrix[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    # Normalize confusion matrix
    confusion = confusion / n

    # Marginal distributions
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)

    # Expected matrix
    expected = np.outer(row_sums, col_sums)

    # Weighted observed and expected
    w_observed = np.sum(weight_matrix * confusion)
    w_expected = np.sum(weight_matrix * expected)

    if w_expected == 0:
        return 1.0
    return 1 - (w_observed / w_expected)


def compute_intra_rater_reliability(
    session: EvaluationSession,
) -> Optional[ReliabilityMetrics]:
    """
    Compute intra-rater reliability from re-rated samples.

    Args:
        session: Evaluation session containing evaluations

    Returns:
        ReliabilityMetrics or None if no re-ratings available
    """
    # Find paired evaluations (first rating vs re-rating)
    first_ratings: Dict[str, HumanEvaluation] = {}
    reratings: Dict[str, HumanEvaluation] = {}

    for eval in session.evaluations:
        if eval.sample_id in session.rerating_sample_ids:
            if eval.is_rerating:
                reratings[eval.sample_id] = eval
            else:
                first_ratings[eval.sample_id] = eval

    # Get paired samples
    paired_ids = set(first_ratings.keys()) & set(reratings.keys())
    if not paired_ids:
        return None

    # Extract rating pairs for each dimension
    dimensions = [
        "text_image_coherence",
        "text_audio_coherence",
        "image_audio_coherence",
        "overall_coherence",
    ]

    all_first = []
    all_second = []

    for sample_id in paired_ids:
        first = first_ratings[sample_id]
        second = reratings[sample_id]

        for dim in dimensions:
            all_first.append(getattr(first, dim))
            all_second.append(getattr(second, dim))

    # Compute metrics
    kappa = compute_cohens_kappa(all_first, all_second)
    weighted_kappa = compute_weighted_kappa(all_first, all_second, weights="quadratic")

    # Simple agreement
    agreements = sum(1 for f, s in zip(all_first, all_second) if f == s)
    percent_agreement = agreements / len(all_first) * 100

    # Mean absolute difference
    mad = np.mean([abs(f - s) for f, s in zip(all_first, all_second)])

    return ReliabilityMetrics(
        kappa=kappa,
        percent_agreement=percent_agreement,
        weighted_kappa=weighted_kappa,
        mean_absolute_difference=mad,
        n_reratings=len(paired_ids),
    )


@dataclass
class HumanEvalSummary:
    """Summary statistics for human evaluations."""
    n_samples: int
    n_evaluations: int

    # Per-dimension statistics
    text_image_mean: float
    text_image_std: float
    text_audio_mean: float
    text_audio_std: float
    image_audio_mean: float
    image_audio_std: float
    overall_mean: float
    overall_std: float

    # Aggregated scores
    mean_weighted_score: float
    std_weighted_score: float

    # Reliability
    reliability: Optional[ReliabilityMetrics]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_evaluations": self.n_evaluations,
            "text_image": {"mean": self.text_image_mean, "std": self.text_image_std},
            "text_audio": {"mean": self.text_audio_mean, "std": self.text_audio_std},
            "image_audio": {"mean": self.image_audio_mean, "std": self.image_audio_std},
            "overall": {"mean": self.overall_mean, "std": self.overall_std},
            "weighted_score": {"mean": self.mean_weighted_score, "std": self.std_weighted_score},
            "reliability": self.reliability.to_dict() if self.reliability else None,
        }


def compute_human_eval_summary(session: EvaluationSession) -> HumanEvalSummary:
    """
    Compute summary statistics for human evaluations.

    Args:
        session: Evaluation session

    Returns:
        HumanEvalSummary with descriptive statistics
    """
    # Filter out re-ratings for summary stats
    evals = [e for e in session.evaluations if not e.is_rerating]

    if not evals:
        raise ValueError("No evaluations found in session")

    # Extract ratings
    ti = [e.text_image_coherence for e in evals]
    ta = [e.text_audio_coherence for e in evals]
    ia = [e.image_audio_coherence for e in evals]
    overall = [e.overall_coherence for e in evals]
    weighted = [e.weighted_score() for e in evals]

    # Compute reliability
    reliability = compute_intra_rater_reliability(session)

    return HumanEvalSummary(
        n_samples=len(set(e.sample_id for e in evals)),
        n_evaluations=len(evals),
        text_image_mean=np.mean(ti),
        text_image_std=np.std(ti),
        text_audio_mean=np.mean(ta),
        text_audio_std=np.std(ta),
        image_audio_mean=np.mean(ia),
        image_audio_std=np.std(ia),
        overall_mean=np.mean(overall),
        overall_std=np.std(overall),
        mean_weighted_score=np.mean(weighted),
        std_weighted_score=np.std(weighted),
        reliability=reliability,
    )


def compute_human_msci_correlation(
    session: EvaluationSession,
    msci_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute correlation between human ratings and MSCI scores.

    Args:
        session: Evaluation session with human ratings
        msci_scores: Optional dict mapping sample_id to MSCI score.
                    If None, uses msci_score from sample metadata.

    Returns:
        Dictionary with correlation statistics
    """
    # Get paired human scores and MSCI scores
    human_weighted = []
    human_overall = []
    msci_values = []

    sample_msci = {}
    if msci_scores:
        sample_msci = msci_scores
    else:
        # Try to extract from session samples
        for sample in session.samples:
            if sample.msci_score is not None:
                sample_msci[sample.sample_id] = sample.msci_score

    for eval in session.evaluations:
        if eval.is_rerating:
            continue

        if eval.sample_id in sample_msci:
            human_weighted.append(eval.weighted_score())
            human_overall.append(eval.overall_coherence / 5.0)  # Normalize to 0-1
            msci_values.append(sample_msci[eval.sample_id])

    if len(msci_values) < 3:
        return {
            "error": "Insufficient paired data for correlation",
            "n_paired": len(msci_values),
        }

    # Spearman correlation (for ordinal human ratings)
    spearman_weighted = stats.spearmanr(msci_values, human_weighted)
    spearman_overall = stats.spearmanr(msci_values, human_overall)

    # Pearson correlation
    pearson_weighted = stats.pearsonr(msci_values, human_weighted)
    pearson_overall = stats.pearsonr(msci_values, human_overall)

    return {
        "n_paired": len(msci_values),
        "msci_vs_weighted_human": {
            "spearman_rho": spearman_weighted.correlation,
            "spearman_p": spearman_weighted.pvalue,
            "pearson_r": pearson_weighted.statistic,
            "pearson_p": pearson_weighted.pvalue,
        },
        "msci_vs_overall_human": {
            "spearman_rho": spearman_overall.correlation,
            "spearman_p": spearman_overall.pvalue,
            "pearson_r": pearson_overall.statistic,
            "pearson_p": pearson_overall.pvalue,
        },
        "interpretation": _interpret_correlation(spearman_weighted.correlation, spearman_weighted.pvalue),
    }


def _interpret_correlation(rho: float, p: float, alpha: float = 0.05) -> str:
    """Generate human-readable interpretation of correlation."""
    if p >= alpha:
        return f"No significant correlation (ρ={rho:.3f}, p={p:.4f} ≥ {alpha})"

    strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.6 else "strong"
    direction = "positive" if rho > 0 else "negative"

    return f"Significant {strength} {direction} correlation (ρ={rho:.3f}, p={p:.4f})"


def analyze_by_condition(session: EvaluationSession) -> Dict[str, Dict[str, Any]]:
    """
    Analyze human ratings grouped by experimental condition.

    Args:
        session: Evaluation session

    Returns:
        Dictionary with statistics per condition
    """
    # Group evaluations by condition
    by_condition: Dict[str, List[HumanEvaluation]] = defaultdict(list)

    # Create sample_id to condition mapping
    sample_to_condition = {s.sample_id: s.condition for s in session.samples}

    for eval in session.evaluations:
        if eval.is_rerating:
            continue
        condition = sample_to_condition.get(eval.sample_id, "unknown")
        by_condition[condition].append(eval)

    results = {}

    for condition, evals in by_condition.items():
        if not evals:
            continue

        weighted = [e.weighted_score() for e in evals]
        overall = [e.overall_coherence for e in evals]

        results[condition] = {
            "n": len(evals),
            "weighted_score": {
                "mean": np.mean(weighted),
                "std": np.std(weighted),
                "median": np.median(weighted),
            },
            "overall_coherence": {
                "mean": np.mean(overall),
                "std": np.std(overall),
                "median": np.median(overall),
            },
        }

    return results


def generate_analysis_report(
    session: EvaluationSession,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive analysis report.

    Args:
        session: Evaluation session
        output_path: Optional path to save JSON report

    Returns:
        Dictionary with complete analysis
    """
    report = {
        "session_id": session.session_id,
        "evaluator_id": session.evaluator_id,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "summary": compute_human_eval_summary(session).to_dict(),
        "by_condition": analyze_by_condition(session),
        "msci_correlation": compute_human_msci_correlation(session),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


# =========================================================================
# Multi-rater analysis (RQ3)
# =========================================================================

def compute_krippendorff_alpha(
    data_matrix: np.ndarray,
    level: str = "ordinal",
) -> float:
    """
    Compute Krippendorff's alpha for inter-rater reliability.

    Args:
        data_matrix: Shape (n_raters, n_items). Use np.nan for missing values.
        level: "nominal", "ordinal", or "interval" measurement level.

    Returns:
        Krippendorff's alpha coefficient (-1 to 1, >0.667 acceptable).
    """
    n_raters, n_items = data_matrix.shape

    # Collect all valid value pairs within each item
    # D_o = observed disagreement, D_e = expected disagreement
    all_values = []
    pairs_observed = []

    for item in range(n_items):
        values = data_matrix[:, item]
        valid = values[~np.isnan(values)]
        if len(valid) < 2:
            continue
        all_values.extend(valid)
        # All pairs within this item
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pairs_observed.append((valid[i], valid[j]))

    if not pairs_observed:
        return 0.0

    all_values = np.array(all_values)

    # Distance function
    if level == "nominal":
        def dist(a, b):
            return 0.0 if a == b else 1.0
    elif level == "ordinal":
        # For ordinal: use squared rank difference
        unique_vals = np.sort(np.unique(all_values))
        val_to_rank = {v: i for i, v in enumerate(unique_vals)}
        def dist(a, b):
            return (val_to_rank[a] - val_to_rank[b]) ** 2
    else:  # interval
        def dist(a, b):
            return (a - b) ** 2

    # Observed disagreement
    D_o = np.mean([dist(a, b) for a, b in pairs_observed])

    # Expected disagreement (all possible pairs from marginal distribution)
    n_total = len(all_values)
    D_e_sum = 0.0
    count = 0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            D_e_sum += dist(all_values[i], all_values[j])
            count += 1

    D_e = D_e_sum / count if count > 0 else 0.0

    if D_e == 0:
        return 1.0  # Perfect agreement if no variance

    alpha = 1.0 - D_o / D_e
    return alpha


def aggregate_multi_rater_sessions(
    sessions: List[EvaluationSession],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate evaluations across multiple raters for the same samples.

    Args:
        sessions: List of completed evaluation sessions (same sample set).

    Returns:
        Dictionary mapping sample_id to aggregated scores.
    """
    # Collect all evaluations per sample (excluding re-ratings)
    by_sample: Dict[str, List[HumanEvaluation]] = defaultdict(list)

    for session in sessions:
        for ev in session.evaluations:
            if ev.is_rerating:
                continue
            by_sample[ev.sample_id].append(ev)

    # Compute mean scores per sample
    aggregated = {}
    for sample_id, evals in by_sample.items():
        ti = [e.text_image_coherence for e in evals]
        ta = [e.text_audio_coherence for e in evals]
        ia = [e.image_audio_coherence for e in evals]
        overall = [e.overall_coherence for e in evals]
        weighted = [e.weighted_score() for e in evals]

        aggregated[sample_id] = {
            "n_raters": len(evals),
            "text_image": {"mean": float(np.mean(ti)), "std": float(np.std(ti))},
            "text_audio": {"mean": float(np.mean(ta)), "std": float(np.std(ta))},
            "image_audio": {"mean": float(np.mean(ia)), "std": float(np.std(ia))},
            "overall": {"mean": float(np.mean(overall)), "std": float(np.std(overall))},
            "weighted_score": {"mean": float(np.mean(weighted)), "std": float(np.std(weighted))},
            "evaluator_ids": [e.evaluator_id for e in evals],
        }

    return aggregated


def compute_inter_rater_reliability(
    sessions: List[EvaluationSession],
) -> Dict[str, Any]:
    """
    Compute inter-rater reliability across multiple evaluators.

    Args:
        sessions: List of evaluation sessions (same sample set).

    Returns:
        Dictionary with Krippendorff's alpha per dimension and overall.
    """
    # Get common sample IDs (rated by all raters)
    sample_sets = []
    for session in sessions:
        ids = {e.sample_id for e in session.evaluations if not e.is_rerating}
        sample_sets.append(ids)

    common_ids = sorted(set.intersection(*sample_sets)) if sample_sets else []

    if len(common_ids) < 3:
        return {"error": "Too few common samples for reliability analysis",
                "n_common": len(common_ids)}

    n_raters = len(sessions)
    n_items = len(common_ids)
    id_to_idx = {sid: i for i, sid in enumerate(common_ids)}

    dimensions = {
        "text_image": "text_image_coherence",
        "text_audio": "text_audio_coherence",
        "image_audio": "image_audio_coherence",
        "overall": "overall_coherence",
    }

    results = {"n_raters": n_raters, "n_common_samples": n_items}

    for dim_name, attr_name in dimensions.items():
        matrix = np.full((n_raters, n_items), np.nan)

        for rater_idx, session in enumerate(sessions):
            for ev in session.evaluations:
                if ev.is_rerating:
                    continue
                if ev.sample_id in id_to_idx:
                    matrix[rater_idx, id_to_idx[ev.sample_id]] = getattr(ev, attr_name)

        alpha = compute_krippendorff_alpha(matrix, level="ordinal")
        results[dim_name] = {
            "krippendorff_alpha": round(alpha, 4),
            "interpretation": _interpret_alpha(alpha),
        }

    # Weighted score dimension
    w_matrix = np.full((n_raters, n_items), np.nan)
    for rater_idx, session in enumerate(sessions):
        for ev in session.evaluations:
            if ev.is_rerating:
                continue
            if ev.sample_id in id_to_idx:
                w_matrix[rater_idx, id_to_idx[ev.sample_id]] = ev.weighted_score()

    alpha_w = compute_krippendorff_alpha(w_matrix, level="interval")
    results["weighted_score"] = {
        "krippendorff_alpha": round(alpha_w, 4),
        "interpretation": _interpret_alpha(alpha_w),
    }

    return results


def _interpret_alpha(alpha: float) -> str:
    """Interpret Krippendorff's alpha value."""
    if alpha >= 0.80:
        return "good agreement"
    elif alpha >= 0.667:
        return "acceptable agreement"
    elif alpha >= 0.40:
        return "moderate agreement"
    else:
        return "poor agreement"


def compute_multi_rater_msci_correlation(
    sessions: List[EvaluationSession],
    sample_msci: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute Spearman correlation between average human scores and MSCI.

    Args:
        sessions: List of evaluation sessions.
        sample_msci: Mapping sample_id -> MSCI score.

    Returns:
        Correlation statistics with bootstrap 95% CI.
    """
    aggregated = aggregate_multi_rater_sessions(sessions)

    human_scores = []
    msci_scores = []

    for sample_id, agg in aggregated.items():
        if sample_id in sample_msci:
            human_scores.append(agg["weighted_score"]["mean"])
            msci_scores.append(sample_msci[sample_id])

    if len(human_scores) < 5:
        return {"error": "Too few paired samples", "n_paired": len(human_scores)}

    human_arr = np.array(human_scores)
    msci_arr = np.array(msci_scores)

    # Spearman rho
    spearman = stats.spearmanr(msci_arr, human_arr)
    # Pearson r
    pearson = stats.pearsonr(msci_arr, human_arr)

    # Bootstrap 95% CI for Spearman rho
    n_boot = 10000
    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(human_arr), size=len(human_arr), replace=True)
        r, _ = stats.spearmanr(msci_arr[idx], human_arr[idx])
        boot_rhos.append(r)
    ci_lower = float(np.percentile(boot_rhos, 2.5))
    ci_upper = float(np.percentile(boot_rhos, 97.5))

    return {
        "n_paired": len(human_scores),
        "spearman_rho": round(float(spearman.correlation), 4),
        "spearman_p": float(spearman.pvalue),
        "spearman_95ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "pearson_r": round(float(pearson.statistic), 4),
        "pearson_p": float(pearson.pvalue),
        "interpretation": _interpret_correlation(
            float(spearman.correlation), float(spearman.pvalue)
        ),
    }
