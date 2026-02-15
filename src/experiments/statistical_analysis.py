"""
Statistical Analysis Module

Provides rigorous statistical tools for hypothesis testing:
- Paired t-tests for within-subject designs
- Effect sizes (Cohen's d)
- Confidence intervals
- Multiple comparison correction
- Power analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """
    Result of a statistical test.

    Includes all information needed for scientific reporting:
    - Test statistic and p-value
    - Effect size with interpretation
    - Confidence interval
    - Sample statistics
    """
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n: int
    mean_diff: float
    std_diff: float
    significant: bool
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "n": self.n,
            "mean_diff": self.mean_diff,
            "std_diff": self.std_diff,
            "significant": self.significant,
            "interpretation": self.interpretation,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        sig_marker = "*" if self.significant else ""
        return (
            f"{self.test_name}: t({self.n-1})={self.statistic:.3f}, "
            f"p={self.p_value:.4f}{sig_marker}, "
            f"{self.effect_size_name}={self.effect_size:.3f}, "
            f"95% CI [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


def paired_ttest(
    condition1: List[float],
    condition2: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalResult:
    """
    Perform paired samples t-test.

    For testing H0: μ1 = μ2 vs H1: μ1 ≠ μ2 (or one-sided alternatives)

    Args:
        condition1: Scores from first condition
        condition2: Scores from second condition (same subjects)
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"

    Returns:
        StatisticalResult with all test statistics
    """
    if len(condition1) != len(condition2):
        raise ValueError("Conditions must have same length for paired test")

    n = len(condition1)
    if n < 2:
        raise ValueError("Need at least 2 observations")

    c1 = np.array(condition1)
    c2 = np.array(condition2)
    differences = c1 - c2

    # Perform t-test
    result = stats.ttest_rel(c1, c2, alternative=alternative)

    # Effect size (Cohen's d for paired samples)
    d = compute_effect_size(condition1, condition2, paired=True)

    # Confidence interval for mean difference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    # Interpretation
    significant = result.pvalue < alpha
    interpretation = _interpret_effect_size(d)

    return StatisticalResult(
        test_name="Paired t-test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=float(d),
        effect_size_name="Cohen's d",
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=1 - alpha,
        n=n,
        mean_diff=float(mean_diff),
        std_diff=float(std_diff),
        significant=significant,
        interpretation=interpretation,
    )


def independent_ttest(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> StatisticalResult:
    """
    Perform independent samples t-test.

    Args:
        group1: Scores from first group
        group2: Scores from second group
        alpha: Significance level
        equal_var: Assume equal variances (use Welch's t-test if False)
        alternative: "two-sided", "greater", or "less"

    Returns:
        StatisticalResult with all test statistics
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    # Perform t-test
    result = stats.ttest_ind(g1, g2, equal_var=equal_var, alternative=alternative)

    # Effect size (Cohen's d)
    d = compute_effect_size(group1, group2, paired=False)

    # Confidence interval for difference in means
    mean_diff = np.mean(g1) - np.mean(g2)
    n1, n2 = len(g1), len(g2)

    if equal_var:
        # Pooled standard error
        pooled_var = ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
        se = np.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch-Satterthwaite approximation
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        se = np.sqrt(var1/n1 + var2/n2)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    significant = result.pvalue < alpha
    interpretation = _interpret_effect_size(d)

    return StatisticalResult(
        test_name="Independent t-test" + ("" if equal_var else " (Welch's)"),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size=float(d),
        effect_size_name="Cohen's d",
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=1 - alpha,
        n=n1 + n2,
        mean_diff=float(mean_diff),
        std_diff=float(se * np.sqrt(n1 + n2)),  # Approximate pooled SD
        significant=significant,
        interpretation=interpretation,
    )


def compute_effect_size(
    group1: List[float],
    group2: List[float],
    paired: bool = True,
) -> float:
    """
    Compute Cohen's d effect size.

    For paired data: d = mean(diff) / std(diff)
    For independent: d = (mean1 - mean2) / pooled_std

    Args:
        group1: First group/condition scores
        group2: Second group/condition scores
        paired: Whether data is paired

    Returns:
        Cohen's d effect size
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    if paired:
        differences = g1 - g2
        d = np.mean(differences) / np.std(differences, ddof=1)
    else:
        n1, n2 = len(g1), len(g2)
        var1 = np.var(g1, ddof=1)
        var2 = np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(g1) - np.mean(g2)) / pooled_std

    return float(d)


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d according to conventional thresholds."""
    abs_d = abs(d)
    if abs_d < 0.2:
        size = "negligible"
    elif abs_d < 0.5:
        size = "small"
    elif abs_d < 0.8:
        size = "medium"
    else:
        size = "large"

    direction = "positive" if d > 0 else "negative" if d < 0 else "no"
    return f"{size} {direction} effect"


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for mean.

    Args:
        data: Sample data
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    arr = np.array(data)
    n = len(arr)
    mean = np.mean(arr)
    se = stats.sem(arr)

    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return float(mean), float(ci_lower), float(ci_upper)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[float, List[bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate

    Returns:
        Tuple of (corrected_alpha, list of significant results)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in p_values]

    return corrected_alpha, significant


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Holm-Bonferroni (step-down) correction.

    More powerful than standard Bonferroni while controlling FWER.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate

    Returns:
        Tuple of (adjusted_p_values, list of significant results)
    """
    n = len(p_values)
    indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[indices]

    adjusted_p = np.zeros(n)
    significant = [False] * n

    for i, idx in enumerate(indices):
        adjusted_p[idx] = sorted_p[i] * (n - i)

    # Enforce monotonicity
    adjusted_p = np.minimum.accumulate(adjusted_p[np.argsort(indices)][::-1])[::-1]
    adjusted_p = np.minimum(adjusted_p, 1.0)

    significant = [p < alpha for p in adjusted_p]

    return list(adjusted_p), significant


def power_analysis_paired_ttest(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Compute required sample size for paired t-test.

    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired statistical power

    Returns:
        Required sample size (N)
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n = ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval.

    Non-parametric CI that makes no distributional assumptions.
    Uses BCa (bias-corrected and accelerated) percentile method.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (default 0.95 for 95% CI)
        statistic: "mean" or "median"
        seed: Random seed for reproducibility

    Returns:
        Dictionary with point estimate, ci_lower, ci_upper, se
    """
    arr = np.array(data)
    n = len(arr)
    rng = np.random.default_rng(seed)

    stat_fn = np.mean if statistic == "mean" else np.median
    observed = float(stat_fn(arr))

    # Generate bootstrap distribution
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    # BCa correction: bias correction factor
    z0 = stats.norm.ppf(np.mean(boot_stats < observed))

    # Acceleration factor (jackknife)
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(arr, i)
        jackknife_stats[i] = stat_fn(jack_sample)
    jack_mean = np.mean(jackknife_stats)
    num = np.sum((jack_mean - jackknife_stats) ** 3)
    den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # Adjusted percentiles
    alpha = 1 - confidence
    z_lower = stats.norm.ppf(alpha / 2)
    z_upper = stats.norm.ppf(1 - alpha / 2)

    p_lower = stats.norm.cdf(z0 + (z0 + z_lower) / (1 - a * (z0 + z_lower)))
    p_upper = stats.norm.cdf(z0 + (z0 + z_upper) / (1 - a * (z0 + z_upper)))

    # Clamp to valid range
    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)

    ci_lower = float(np.percentile(boot_stats, p_lower * 100))
    ci_upper = float(np.percentile(boot_stats, p_upper * 100))

    return {
        "estimate": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(boot_stats)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "method": "BCa bootstrap",
    }


def bootstrap_ci_diff(
    group1: List[float],
    group2: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    paired: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap CI for the difference between two groups.

    Args:
        group1: First group scores
        group2: Second group scores
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level
        paired: Whether data is paired (same subjects)
        seed: Random seed

    Returns:
        Dictionary with mean difference, ci_lower, ci_upper
    """
    g1 = np.array(group1)
    g2 = np.array(group2)
    rng = np.random.default_rng(seed)

    if paired:
        diffs = g1 - g2
        n = len(diffs)
        observed_diff = float(np.mean(diffs))

        boot_diffs = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(diffs, size=n, replace=True)
            boot_diffs[i] = np.mean(sample)
    else:
        n1, n2 = len(g1), len(g2)
        observed_diff = float(np.mean(g1) - np.mean(g2))

        boot_diffs = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            s1 = rng.choice(g1, size=n1, replace=True)
            s2 = rng.choice(g2, size=n2, replace=True)
            boot_diffs[i] = np.mean(s1) - np.mean(s2)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_diffs, (alpha / 2) * 100))
    ci_upper = float(np.percentile(boot_diffs, (1 - alpha / 2) * 100))

    return {
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(boot_diffs)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def descriptive_stats(data: List[float]) -> Dict[str, float]:
    """
    Compute descriptive statistics for a sample.

    Args:
        data: Sample data

    Returns:
        Dictionary with mean, std, median, min, max, N, bootstrap CI
    """
    arr = np.array(data)
    boot = bootstrap_ci(list(arr))
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "se": float(stats.sem(arr)),
        "ci_lower_95": boot["ci_lower"],
        "ci_upper_95": boot["ci_upper"],
    }


def shapiro_wilk_test(
    data: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Shapiro-Wilk test for normality.

    Args:
        data: Sample data (paired differences for paired tests)
        alpha: Significance level

    Returns:
        Dictionary with W statistic, p-value, and whether normality holds
    """
    arr = np.array(data)
    w, p = stats.shapiro(arr)
    return {
        "test_name": "Shapiro-Wilk",
        "W": float(w),
        "p_value": float(p),
        "normal": bool(p > alpha),
        "alpha": alpha,
    }


def wilcoxon_signed_rank(
    condition1: List[float],
    condition2: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        condition1: Scores from first condition
        condition2: Scores from second condition (same subjects)
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"

    Returns:
        Dictionary with test statistic, p-value, and rank-biserial correlation
    """
    c1 = np.array(condition1)
    c2 = np.array(condition2)
    diff = c1 - c2

    result = stats.wilcoxon(diff, alternative=alternative)
    n = len(diff)

    # Rank-biserial correlation as effect size
    r_rb = 1 - (2 * float(result.statistic)) / (n * (n + 1) / 2)

    return {
        "test_name": "Wilcoxon signed-rank",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size": float(r_rb),
        "effect_size_name": "rank-biserial r",
        "n": n,
        "significant": bool(result.pvalue < alpha),
        "alpha": alpha,
    }


def cohens_d_ci(
    d: float,
    n: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Approximate 95% confidence interval for Cohen's d.

    Uses the formula: SE(d) = sqrt(1/n + d^2 / (2*n))

    Args:
        d: Cohen's d point estimate
        n: Sample size (number of pairs for paired tests)
        alpha: Significance level

    Returns:
        Dictionary with d, ci_lower, ci_upper
    """
    se = np.sqrt(1 / n + d**2 / (2 * n))
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "d": float(d),
        "ci_lower": float(d - z * se),
        "ci_upper": float(d + z * se),
        "se": float(se),
    }


def compare_all_pairs(
    conditions: Dict[str, List[float]],
    alpha: float = 0.05,
    paired: bool = True,
    correction: str = "holm",
) -> Dict[str, StatisticalResult]:
    """
    Compare all pairs of conditions with multiple comparison correction.

    Args:
        conditions: Dictionary mapping condition names to scores
        alpha: Family-wise error rate
        paired: Whether data is paired
        correction: "bonferroni" or "holm"

    Returns:
        Dictionary of comparison results
    """
    condition_names = list(conditions.keys())
    n_conditions = len(condition_names)

    results = {}
    p_values = []
    comparison_keys = []

    # Perform all pairwise comparisons
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            name1, name2 = condition_names[i], condition_names[j]
            key = f"{name1}_vs_{name2}"

            if paired:
                result = paired_ttest(
                    conditions[name1], conditions[name2], alpha=alpha
                )
            else:
                result = independent_ttest(
                    conditions[name1], conditions[name2], alpha=alpha
                )

            results[key] = result
            p_values.append(result.p_value)
            comparison_keys.append(key)

    # Apply correction
    if correction == "bonferroni":
        corrected_alpha, significant = bonferroni_correction(p_values, alpha)
        adjusted_p = [p * len(p_values) for p in p_values]
    else:  # holm
        adjusted_p, significant = holm_bonferroni_correction(p_values, alpha)

    # Update results with corrected significance
    for key, adj_p, sig in zip(comparison_keys, adjusted_p, significant):
        result = results[key]
        # Create new result with adjusted values
        results[key] = StatisticalResult(
            test_name=result.test_name + f" ({correction}-corrected)",
            statistic=result.statistic,
            p_value=min(adj_p, 1.0),  # Original p-value replaced with adjusted
            effect_size=result.effect_size,
            effect_size_name=result.effect_size_name,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            confidence_level=result.confidence_level,
            n=result.n,
            mean_diff=result.mean_diff,
            std_diff=result.std_diff,
            significant=sig,
            interpretation=result.interpretation,
        )

    return results


def spearman_correlation(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute Spearman rank correlation with confidence interval.

    Args:
        x: First variable
        y: Second variable
        alpha: Significance level

    Returns:
        Dictionary with correlation, p-value, CI, and interpretation
    """
    result = stats.spearmanr(x, y)
    rho = result.correlation
    p = result.pvalue
    n = len(x)

    # Fisher z-transformation for CI
    z = np.arctanh(rho)
    se_z = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_z_lower = z - z_crit * se_z
    ci_z_upper = z + z_crit * se_z
    ci_lower = np.tanh(ci_z_lower)
    ci_upper = np.tanh(ci_z_upper)

    # Interpretation
    abs_rho = abs(rho)
    if abs_rho < 0.1:
        strength = "negligible"
    elif abs_rho < 0.3:
        strength = "weak"
    elif abs_rho < 0.5:
        strength = "moderate"
    elif abs_rho < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if rho > 0 else "negative"
    significant = p < alpha

    return {
        "rho": float(rho),
        "p_value": float(p),
        "n": n,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": 1 - alpha,
        "significant": significant,
        "interpretation": f"{'Significant' if significant else 'Non-significant'} {strength} {direction} correlation",
    }
