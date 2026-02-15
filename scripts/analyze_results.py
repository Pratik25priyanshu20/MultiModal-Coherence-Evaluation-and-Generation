"""
Statistical Analysis for RQ1 and RQ2 Experiments.

Reads experiment results JSON and produces:
- Descriptive statistics per condition
- Paired t-tests with Cohen's d
- Holm-Bonferroni correction for multiple comparisons
- Summary tables
- Verdict on each research question

Usage:
    python scripts/analyze_results.py runs/rq1/rq1_results.json
    python scripts/analyze_results.py runs/rq2/rq2_results.json
    python scripts/analyze_results.py runs/rq1/rq1_results.json --format latex
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.statistical_analysis import (
    paired_ttest,
    compute_effect_size,
    holm_bonferroni_correction,
    descriptive_stats,
    compute_confidence_interval,
    power_analysis_paired_ttest,
    bootstrap_ci,
    bootstrap_ci_diff,
    shapiro_wilk_test,
    wilcoxon_signed_rank,
    cohens_d_ci,
)


def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def aggregate_by_prompt(
    results: List[Dict[str, Any]],
    group_key: str,
    value_key: str = "msci",
) -> Dict[str, List[float]]:
    """
    Aggregate scores by prompt, averaging across seeds.

    Returns {condition_or_mode: [per-prompt average scores]}
    For paired tests, each prompt must appear in both conditions.
    """
    # Collect (prompt_id, seed) → score per group
    raw = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get(value_key) is not None and "error" not in r:
            group = r[group_key]
            pid = r["prompt_id"]
            raw[group][pid].append(r[value_key])

    # Average across seeds per prompt
    aggregated = {}
    for group, prompt_scores in raw.items():
        aggregated[group] = {
            pid: np.mean(scores) for pid, scores in prompt_scores.items()
        }

    # Find common prompts across all groups
    all_groups = list(aggregated.keys())
    if not all_groups:
        return {}

    common_prompts = set(aggregated[all_groups[0]].keys())
    for group in all_groups[1:]:
        common_prompts &= set(aggregated[group].keys())

    common_prompts = sorted(common_prompts)

    # Build aligned lists
    result = {}
    for group in all_groups:
        result[group] = [aggregated[group][pid] for pid in common_prompts]

    return result


def analyze_rq1(results: List[Dict[str, Any]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    RQ1 Analysis: Is MSCI sensitive to perturbations?

    Tests:
    - H1: MSCI(baseline) > MSCI(wrong_image)
    - H2: MSCI(baseline) > MSCI(wrong_audio)
    """
    print("\n" + "=" * 90)
    print("RQ1 ANALYSIS: IS MSCI SENSITIVE TO CONTROLLED SEMANTIC PERTURBATIONS?")
    print("=" * 90)

    # Aggregate by prompt (average across seeds)
    msci_by_condition = aggregate_by_prompt(results, group_key="condition", value_key="msci")
    sti_by_condition = aggregate_by_prompt(results, group_key="condition", value_key="st_i")
    sta_by_condition = aggregate_by_prompt(results, group_key="condition", value_key="st_a")

    if "baseline" not in msci_by_condition:
        print("  ERROR: No baseline results found.")
        return {"error": "No baseline results"}

    n_prompts = len(msci_by_condition["baseline"])
    print(f"\n  Prompts with complete data: {n_prompts}")

    # Descriptive statistics
    print("\n  DESCRIPTIVE STATISTICS (MSCI)")
    print(f"  {'Condition':<16} {'N':>4} {'Mean':>8} {'Std':>8} {'95% CI':>20} {'Median':>8}")
    print(f"  {'-'*16} {'-'*4} {'-'*8} {'-'*8} {'-'*20} {'-'*8}")

    desc = {}
    for cond in ["baseline", "wrong_image", "wrong_audio"]:
        if cond in msci_by_condition:
            d = descriptive_stats(msci_by_condition[cond])
            desc[cond] = d
            ci_str = f"[{d['ci_lower_95']:.4f}, {d['ci_upper_95']:.4f}]"
            print(
                f"  {cond:<16} {d['n']:>4} {d['mean']:>8.4f} {d['std']:>8.4f} "
                f"{ci_str:>20} {d['median']:>8.4f}"
            )

    # Sub-metric descriptives
    for metric_name, metric_data in [("st_i", sti_by_condition), ("st_a", sta_by_condition)]:
        print(f"\n  DESCRIPTIVE STATISTICS ({metric_name})")
        print(f"  {'Condition':<16} {'Mean':>8} {'Std':>8}")
        print(f"  {'-'*16} {'-'*8} {'-'*8}")
        for cond in ["baseline", "wrong_image", "wrong_audio"]:
            if cond in metric_data:
                vals = metric_data[cond]
                print(f"  {cond:<16} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")

    # Hypothesis tests
    print("\n  HYPOTHESIS TESTS (paired t-test, one-sided: baseline > perturbation)")
    print(f"  {'-'*90}")

    test_results = []
    comparisons = []
    normality_tests = []
    nonparametric_tests = []
    effect_size_cis = []

    for pert in ["wrong_image", "wrong_audio"]:
        if pert not in msci_by_condition:
            continue

        baseline_vals = msci_by_condition["baseline"]
        pert_vals = msci_by_condition[pert]
        diff = [b - p for b, p in zip(baseline_vals, pert_vals)]

        # Normality check on paired differences
        sw = shapiro_wilk_test(diff, alpha=alpha)
        normality_tests.append({"comparison": f"baseline vs {pert}", **sw})
        normal_str = "PASS" if sw["normal"] else "FAIL"
        print(f"\n  Shapiro-Wilk (baseline - {pert}): W = {sw['W']:.4f}, p = {sw['p_value']:.4f} → {normal_str}")

        # Paired t-test
        result = paired_ttest(
            baseline_vals,
            pert_vals,
            alpha=alpha,
            alternative="greater",
        )
        test_results.append(result)
        comparisons.append(f"baseline vs {pert}")

        # Cohen's d CI
        d_ci = cohens_d_ci(result.effect_size, result.n, alpha=alpha)
        effect_size_cis.append({"comparison": f"baseline vs {pert}", **d_ci})

        # Wilcoxon signed-rank (non-parametric backup)
        wsr = wilcoxon_signed_rank(
            baseline_vals,
            pert_vals,
            alpha=alpha,
            alternative="greater",
        )
        nonparametric_tests.append({"comparison": f"baseline vs {pert}", **wsr})

        boot_diff = bootstrap_ci_diff(
            baseline_vals,
            pert_vals,
            paired=True,
        )

        sig = "*" if result.significant else ""
        print(f"  baseline vs {pert}:")
        print(f"    t({result.n-1}) = {result.statistic:.3f}, p = {result.p_value:.6f}{sig}")
        print(f"    Cohen's d = {result.effect_size:.3f} [{d_ci['ci_lower']:.2f}, {d_ci['ci_upper']:.2f}] ({result.interpretation})")
        print(f"    Mean Δ = {result.mean_diff:+.4f} ± {result.std_diff:.4f}")
        print(f"    95% CI (parametric)  [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"    95% CI (bootstrap)   [{boot_diff['ci_lower']:.4f}, {boot_diff['ci_upper']:.4f}]")
        wsig = "*" if wsr["significant"] else ""
        print(f"    Wilcoxon W = {wsr['statistic']:.1f}, p = {wsr['p_value']:.2e}{wsig}, r_rb = {wsr['effect_size']:.3f}")

    # Multiple comparison correction (Holm-Bonferroni)
    holm_results = []
    if len(test_results) > 1:
        p_values = [r.p_value for r in test_results]
        adjusted_p, significant = holm_bonferroni_correction(p_values, alpha)

        print(f"\n  HOLM-BONFERRONI CORRECTION (α={alpha})")
        print(f"  {'Comparison':<30} {'Raw p':>12} {'Adj p':>12} {'Sig':>5}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*5}")
        for comp, raw_p, adj_p, sig in zip(comparisons, p_values, adjusted_p, significant):
            print(f"  {comp:<30} {raw_p:>12.2e} {adj_p:>12.2e} {'*' if sig else ''}")
            holm_results.append({"comparison": comp, "raw_p": raw_p, "adjusted_p": adj_p, "significant": sig})

    # Power analysis
    if test_results:
        observed_d = np.mean([abs(r.effect_size) for r in test_results])
        required_n = power_analysis_paired_ttest(observed_d if observed_d > 0.1 else 0.5)
        print(f"\n  POWER ANALYSIS")
        print(f"    Observed avg |d| = {observed_d:.3f}")
        print(f"    Current N = {n_prompts}")
        print(f"    Required N (for observed d, power=0.80) = {required_n}")
        print(f"    Adequately powered: {'Yes' if n_prompts >= required_n else 'No'}")

    # Verdict
    print("\n  " + "=" * 70)
    all_sig = all(r.significant for r in test_results) if test_results else False
    all_medium = all(abs(r.effect_size) >= 0.5 for r in test_results) if test_results else False

    if all_sig and all_medium:
        verdict = "SUPPORTED — MSCI is sensitive to perturbations (p < 0.05, |d| >= 0.5)"
    elif all_sig:
        verdict = "PARTIALLY SUPPORTED — Significant but small effect sizes"
    elif test_results:
        verdict = "NOT SUPPORTED — Perturbations do not reliably affect MSCI"
    else:
        verdict = "INCONCLUSIVE — Insufficient data"

    print(f"  VERDICT: {verdict}")
    print("  " + "=" * 70)

    return {
        "descriptive": desc,
        "tests": [r.to_dict() for r in test_results],
        "normality_tests": normality_tests,
        "nonparametric_tests": nonparametric_tests,
        "effect_size_cis": effect_size_cis,
        "holm_bonferroni": holm_results,
        "comparisons": comparisons,
        "verdict": verdict,
        "n_prompts": n_prompts,
    }


def analyze_rq2(results: List[Dict[str, Any]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    RQ2 Analysis: Does planning improve alignment?

    Tests:
    - H1: MSCI(planner) > MSCI(direct)
    - H2: MSCI(council) > MSCI(direct)
    - H3: MSCI(extended_prompt) > MSCI(direct)
    - Control: MSCI(council) vs MSCI(extended_prompt) [structure vs tokens]
    """
    print("\n" + "=" * 90)
    print("RQ2 ANALYSIS: DOES STRUCTURED PLANNING IMPROVE CROSS-MODAL ALIGNMENT?")
    print("=" * 90)

    msci_by_mode = aggregate_by_prompt(results, group_key="mode", value_key="msci")
    sti_by_mode = aggregate_by_prompt(results, group_key="mode", value_key="st_i")
    sta_by_mode = aggregate_by_prompt(results, group_key="mode", value_key="st_a")

    modes = sorted(msci_by_mode.keys())
    if "direct" not in msci_by_mode:
        print("  ERROR: No direct mode results found.")
        return {"error": "No direct results"}

    n_prompts = len(msci_by_mode["direct"])
    print(f"\n  Prompts with complete data: {n_prompts}")

    # Descriptive statistics
    print("\n  DESCRIPTIVE STATISTICS (MSCI)")
    print(f"  {'Mode':<20} {'N':>4} {'Mean':>8} {'Std':>8} {'95% CI':>20} {'Median':>8}")
    print(f"  {'-'*20} {'-'*4} {'-'*8} {'-'*8} {'-'*20} {'-'*8}")

    desc = {}
    for mode in modes:
        d = descriptive_stats(msci_by_mode[mode])
        desc[mode] = d
        ci_str = f"[{d['ci_lower_95']:.4f}, {d['ci_upper_95']:.4f}]"
        print(
            f"  {mode:<20} {d['n']:>4} {d['mean']:>8.4f} {d['std']:>8.4f} "
            f"{ci_str:>20} {d['median']:>8.4f}"
        )

    # Sub-metrics
    for metric_name, metric_data in [("st_i", sti_by_mode), ("st_a", sta_by_mode)]:
        print(f"\n  DESCRIPTIVE STATISTICS ({metric_name})")
        print(f"  {'Mode':<20} {'Mean':>8} {'Std':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8}")
        for mode in modes:
            if mode in metric_data:
                vals = metric_data[mode]
                print(f"  {mode:<20} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")

    # Hypothesis tests: each planner mode vs direct
    print("\n  HYPOTHESIS TESTS (paired t-test, two-sided)")
    print(f"  {'-'*90}")

    test_results = []
    comparisons = []
    normality_tests = []
    nonparametric_tests = []
    effect_size_cis = []
    planner_modes = [m for m in modes if m != "direct"]

    for mode in planner_modes:
        mode_vals = msci_by_mode[mode]
        direct_vals = msci_by_mode["direct"]
        diff = [m - d for m, d in zip(mode_vals, direct_vals)]

        # Normality check on paired differences
        sw = shapiro_wilk_test(diff, alpha=alpha)
        normality_tests.append({"comparison": f"{mode} vs direct", **sw})
        normal_str = "PASS" if sw["normal"] else "FAIL"
        print(f"\n  Shapiro-Wilk ({mode} - direct): W = {sw['W']:.4f}, p = {sw['p_value']:.4f} → {normal_str}")

        result = paired_ttest(
            mode_vals,
            direct_vals,
            alpha=alpha,
            alternative="two-sided",
        )
        test_results.append(result)
        comparisons.append(f"{mode} vs direct")

        # Cohen's d CI
        d_ci = cohens_d_ci(result.effect_size, result.n, alpha=alpha)
        effect_size_cis.append({"comparison": f"{mode} vs direct", **d_ci})

        # Wilcoxon signed-rank (non-parametric backup)
        wsr = wilcoxon_signed_rank(
            mode_vals,
            direct_vals,
            alpha=alpha,
            alternative="two-sided",
        )
        nonparametric_tests.append({"comparison": f"{mode} vs direct", **wsr})

        boot_diff = bootstrap_ci_diff(
            mode_vals,
            direct_vals,
            paired=True,
        )

        sig = "*" if result.significant else ""
        print(f"  {mode} vs direct:")
        print(f"    t({result.n-1}) = {result.statistic:.3f}, p = {result.p_value:.6f}{sig}")
        print(f"    Cohen's d = {result.effect_size:.3f} [{d_ci['ci_lower']:.2f}, {d_ci['ci_upper']:.2f}] ({result.interpretation})")
        print(f"    Mean Δ = {result.mean_diff:+.4f}")
        print(f"    95% CI (bootstrap)   [{boot_diff['ci_lower']:.4f}, {boot_diff['ci_upper']:.4f}]")
        wsig = "*" if wsr["significant"] else ""
        print(f"    Wilcoxon W = {wsr['statistic']:.1f}, p = {wsr['p_value']:.2e}{wsig}, r_rb = {wsr['effect_size']:.3f}")

    # Control comparison: council vs extended_prompt (structure vs tokens)
    if "council" in msci_by_mode and "extended_prompt" in msci_by_mode:
        print(f"\n  CONTROL: Structure vs Token Budget")
        council_vals = msci_by_mode["council"]
        ext_vals = msci_by_mode["extended_prompt"]
        diff_ctrl = [c - e for c, e in zip(council_vals, ext_vals)]

        sw_ctrl = shapiro_wilk_test(diff_ctrl, alpha=alpha)
        normality_tests.append({"comparison": "council vs extended_prompt", **sw_ctrl})

        result = paired_ttest(
            council_vals,
            ext_vals,
            alpha=alpha,
            alternative="two-sided",
        )
        test_results.append(result)
        comparisons.append("council vs extended_prompt")

        d_ci_ctrl = cohens_d_ci(result.effect_size, result.n, alpha=alpha)
        effect_size_cis.append({"comparison": "council vs extended_prompt", **d_ci_ctrl})

        wsr_ctrl = wilcoxon_signed_rank(council_vals, ext_vals, alpha=alpha, alternative="two-sided")
        nonparametric_tests.append({"comparison": "council vs extended_prompt", **wsr_ctrl})

        sig = "*" if result.significant else ""
        print(f"  council vs extended_prompt:")
        print(f"    t({result.n-1}) = {result.statistic:.3f}, p = {result.p_value:.6f}{sig}")
        print(f"    Cohen's d = {result.effect_size:.3f} [{d_ci_ctrl['ci_lower']:.2f}, {d_ci_ctrl['ci_upper']:.2f}] ({result.interpretation})")
        print(f"    Mean Δ = {result.mean_diff:+.4f}")
        wsig = "*" if wsr_ctrl["significant"] else ""
        print(f"    Wilcoxon W = {wsr_ctrl['statistic']:.1f}, p = {wsr_ctrl['p_value']:.2e}{wsig}, r_rb = {wsr_ctrl['effect_size']:.3f}")

    # Multiple comparison correction (Holm-Bonferroni)
    holm_results = []
    if len(test_results) > 1:
        p_values = [r.p_value for r in test_results]
        adjusted_p, significant = holm_bonferroni_correction(p_values, alpha)

        print(f"\n  HOLM-BONFERRONI CORRECTION (α={alpha})")
        print(f"  {'Comparison':<35} {'Raw p':>12} {'Adj p':>12} {'Sig':>5}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*5}")
        for comp, raw_p, adj_p, sig in zip(comparisons, p_values, adjusted_p, significant):
            print(f"  {comp:<35} {raw_p:>12.4f} {adj_p:>12.4f} {'*' if sig else ''}")
            holm_results.append({"comparison": comp, "raw_p": raw_p, "adjusted_p": adj_p, "significant": sig})

    # Power analysis for null result
    print(f"\n  SENSITIVITY ANALYSIS (post-hoc power)")
    from scipy.stats import t as t_dist, nct as nct_dist
    n_pow = n_prompts
    df_pow = n_pow - 1
    t_crit_pow = t_dist.ppf(1 - alpha / 2, df_pow)
    # Find d at 80% power
    d_range = np.arange(0.01, 2.0, 0.01)
    for target_power in [0.80, 0.90]:
        for d_check in d_range:
            ncp = d_check * np.sqrt(n_pow)
            power = 1 - nct_dist.cdf(t_crit_pow, df_pow, ncp) + nct_dist.cdf(-t_crit_pow, df_pow, ncp)
            if power >= target_power:
                print(f"    Minimum detectable d at {int(target_power*100)}% power (N={n_pow}): d = {d_check:.2f}")
                break

    # Verdict
    print("\n  " + "=" * 70)
    planner_tests = [r for r, c in zip(test_results, comparisons) if "direct" in c]
    any_sig = any(r.significant for r in planner_tests) if planner_tests else False
    all_sig = all(r.significant for r in planner_tests) if planner_tests else False

    # Check for positive significant effects (improvement) vs negative (degradation)
    sig_positive = [r for r, c in zip(test_results, comparisons) if r.significant and "direct" in c and r.effect_size > 0]
    sig_negative = [r for r, c in zip(test_results, comparisons) if r.significant and "direct" in c and r.effect_size < 0]

    if all_sig and all(r.effect_size > 0 for r in planner_tests):
        verdict = "SUPPORTED — Planning improves alignment across all modes"
    elif sig_positive:
        sig_modes = [c.split(" vs ")[0] for r, c in zip(test_results, comparisons) if r.significant and "direct" in c and r.effect_size > 0]
        verdict = f"PARTIALLY SUPPORTED — Improvement seen for: {', '.join(sig_modes)}"
    elif sig_negative:
        deg_modes = [c.split(" vs ")[0] for r, c in zip(test_results, comparisons) if r.significant and "direct" in c and r.effect_size < 0]
        verdict = f"NOT SUPPORTED — Planning degrades alignment for: {', '.join(deg_modes)}"
    elif planner_tests:
        verdict = "NOT SUPPORTED — Planning does not significantly improve alignment"
    else:
        verdict = "INCONCLUSIVE — Insufficient data"

    print(f"  VERDICT: {verdict}")
    print("  " + "=" * 70)

    return {
        "descriptive": desc,
        "tests": [r.to_dict() for r in test_results],
        "normality_tests": normality_tests,
        "nonparametric_tests": nonparametric_tests,
        "effect_size_cis": effect_size_cis,
        "holm_bonferroni": holm_results,
        "comparisons": comparisons,
        "verdict": verdict,
        "n_prompts": n_prompts,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze RQ1/RQ2 experiment results")
    parser.add_argument("results_file", help="Path to results JSON (rq1_results.json or rq2_results.json)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--out", help="Save analysis to JSON file")
    args = parser.parse_args()

    data = load_results(args.results_file)
    experiment = data.get("experiment", "")
    results = data.get("results", [])

    print(f"\nLoaded {len(results)} results from: {args.results_file}")
    print(f"Experiment: {experiment}")
    print(f"Config: {json.dumps(data.get('config', {}), indent=2)}")

    # Detect RQ type
    if "RQ1" in experiment or "condition" in results[0]:
        analysis = analyze_rq1(results, alpha=args.alpha)
    elif "RQ2" in experiment or "mode" in results[0]:
        analysis = analyze_rq2(results, alpha=args.alpha)
    else:
        # Try both if ambiguous
        if any(r.get("condition") for r in results):
            analysis = analyze_rq1(results, alpha=args.alpha)
        else:
            analysis = analyze_rq2(results, alpha=args.alpha)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
