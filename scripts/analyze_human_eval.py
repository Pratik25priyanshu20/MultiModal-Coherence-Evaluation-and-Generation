#!/usr/bin/env python3
"""
Human Evaluation Analysis Script

Analyzes human evaluation sessions to compute:
- Intra-rater reliability (Cohen's kappa)
- Summary statistics
- Correlation with MSCI scores
- Per-condition analysis

Usage:
    python scripts/analyze_human_eval.py --session evaluation/human_eval_sessions/session_xxx.json
    python scripts/analyze_human_eval.py --all-sessions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.human_eval_schema import EvaluationSession
from src.evaluation.human_eval_analyzer import (
    compute_human_eval_summary,
    compute_human_msci_correlation,
    compute_intra_rater_reliability,
    analyze_by_condition,
    generate_analysis_report,
)


def print_summary(summary: dict):
    """Print formatted summary statistics."""
    print("\n" + "=" * 60)
    print("HUMAN EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nSamples evaluated: {summary['n_samples']}")
    print(f"Total evaluations: {summary['n_evaluations']}")

    print("\n--- Per-Dimension Statistics (1-5 Likert) ---")
    for dim in ['text_image', 'text_audio', 'image_audio', 'overall']:
        data = summary[dim]
        print(f"  {dim:15}: mean={data['mean']:.2f}, std={data['std']:.2f}")

    print("\n--- Weighted Score (0-1, MSCI-comparable) ---")
    ws = summary['weighted_score']
    print(f"  Mean: {ws['mean']:.4f}")
    print(f"  Std:  {ws['std']:.4f}")


def print_reliability(reliability: dict):
    """Print intra-rater reliability metrics."""
    print("\n" + "=" * 60)
    print("INTRA-RATER RELIABILITY (Self-Consistency)")
    print("=" * 60)

    if reliability is None:
        print("\n  No re-ratings available for reliability analysis.")
        return

    print(f"\n  N re-rated samples: {reliability['n_reratings']}")
    print(f"  Cohen's kappa:      {reliability['kappa']:.3f}")
    print(f"  Weighted kappa:     {reliability['weighted_kappa']:.3f}")
    print(f"  Percent agreement:  {reliability['percent_agreement']:.1f}%")
    print(f"  Mean |difference|:  {reliability['mean_absolute_difference']:.2f}")

    # Interpretation
    kappa = reliability['kappa']
    if kappa >= 0.80:
        interpretation = "Almost perfect agreement"
    elif kappa >= 0.70:
        interpretation = "Substantial agreement (acceptable)"
    elif kappa >= 0.60:
        interpretation = "Moderate agreement (borderline)"
    elif kappa >= 0.40:
        interpretation = "Fair agreement (needs improvement)"
    else:
        interpretation = "Poor agreement (unreliable)"

    print(f"\n  Interpretation: {interpretation}")

    threshold_met = reliability['is_acceptable']
    print(f"  Threshold (κ ≥ 0.70): {'✓ MET' if threshold_met else '✗ NOT MET'}")


def print_correlation(correlation: dict):
    """Print MSCI correlation results."""
    print("\n" + "=" * 60)
    print("MSCI vs HUMAN CORRELATION (RQ3)")
    print("=" * 60)

    if "error" in correlation:
        print(f"\n  {correlation['error']}")
        print(f"  Paired samples available: {correlation['n_paired']}")
        return

    print(f"\n  Paired samples: {correlation['n_paired']}")

    print("\n--- MSCI vs Weighted Human Score ---")
    w = correlation['msci_vs_weighted_human']
    print(f"  Spearman ρ: {w['spearman_rho']:.4f} (p={w['spearman_p']:.4f})")
    print(f"  Pearson r:  {w['pearson_r']:.4f} (p={w['pearson_p']:.4f})")

    print("\n--- MSCI vs Overall Human Rating ---")
    o = correlation['msci_vs_overall_human']
    print(f"  Spearman ρ: {o['spearman_rho']:.4f} (p={o['spearman_p']:.4f})")
    print(f"  Pearson r:  {o['pearson_r']:.4f} (p={o['pearson_p']:.4f})")

    print(f"\n  Interpretation: {correlation['interpretation']}")

    # RQ3 verdict
    rho = w['spearman_rho']
    p = w['spearman_p']
    print("\n  RQ3 Verdict:")
    if p < 0.05 and rho > 0.3:
        print("    ✓ H1 SUPPORTED: MSCI correlates positively with human judgments")
    elif p < 0.05 and rho > 0:
        print("    ~ WEAK SUPPORT: Significant but weak positive correlation")
    elif p >= 0.05:
        print("    ✗ H0 NOT REJECTED: No significant correlation found")
    else:
        print("    ✗ UNEXPECTED: Negative correlation (MSCI may be inverted)")


def print_condition_analysis(by_condition: dict):
    """Print per-condition analysis."""
    print("\n" + "=" * 60)
    print("ANALYSIS BY CONDITION (RQ2)")
    print("=" * 60)

    if not by_condition:
        print("\n  No condition data available.")
        return

    # Sort conditions
    conditions = sorted(by_condition.keys())

    print("\n  Condition               N    WeightedScore    Overall")
    print("  " + "-" * 55)

    for cond in conditions:
        data = by_condition[cond]
        ws = data['weighted_score']
        ov = data['overall_coherence']
        print(f"  {cond:22} {data['n']:3}   {ws['mean']:.4f}±{ws['std']:.3f}   "
              f"{ov['mean']:.2f}±{ov['std']:.2f}")

    # Compare planner vs direct (if both present)
    planner_conds = [c for c in conditions if c.startswith('planner_baseline')]
    direct_conds = [c for c in conditions if c.startswith('direct_baseline')]

    if planner_conds and direct_conds:
        print("\n--- Planner vs Direct (Baseline Only) ---")
        planner_ws = by_condition[planner_conds[0]]['weighted_score']['mean']
        direct_ws = by_condition[direct_conds[0]]['weighted_score']['mean']
        diff = planner_ws - direct_ws
        print(f"  Planner: {planner_ws:.4f}")
        print(f"  Direct:  {direct_ws:.4f}")
        print(f"  Difference: {diff:+.4f}")
        if diff > 0:
            print("  → Planner shows higher human-rated coherence")
        else:
            print("  → Direct shows higher human-rated coherence")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze human evaluation results"
    )

    parser.add_argument(
        "--session",
        type=str,
        help="Path to session JSON file",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Analyze all sessions in evaluation/human_eval_sessions/",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save analysis report (JSON)",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default="evaluation/human_eval_sessions",
        help="Directory containing session files (for --all-sessions)",
    )

    args = parser.parse_args()

    if not args.session and not args.all_sessions:
        parser.print_help()
        print("\nError: Specify --session or --all-sessions")
        sys.exit(1)

    sessions_to_analyze = []

    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            print(f"Error: Session file not found: {session_path}")
            sys.exit(1)
        sessions_to_analyze.append(session_path)

    if args.all_sessions:
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            print(f"Error: Session directory not found: {session_dir}")
            sys.exit(1)
        sessions_to_analyze.extend(sorted(session_dir.glob("session_*.json")))

    if not sessions_to_analyze:
        print("No session files found.")
        sys.exit(1)

    # Analyze each session
    for session_path in sessions_to_analyze:
        print(f"\n{'#' * 70}")
        print(f"# ANALYZING: {session_path.name}")
        print(f"{'#' * 70}")

        try:
            session = EvaluationSession.load(session_path)
        except Exception as e:
            print(f"Error loading session: {e}")
            continue

        if not session.evaluations:
            print("  No evaluations in this session.")
            continue

        # Generate report
        report = generate_analysis_report(session)

        # Print results
        print_summary(report['summary'])
        print_reliability(report['summary'].get('reliability'))
        print_correlation(report['msci_correlation'])
        print_condition_analysis(report['by_condition'])

        # Save report if requested
        if args.output:
            output_path = Path(args.output)
            if len(sessions_to_analyze) > 1:
                output_path = output_path.with_stem(
                    f"{output_path.stem}_{session_path.stem}"
                )
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n  Report saved to: {output_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
