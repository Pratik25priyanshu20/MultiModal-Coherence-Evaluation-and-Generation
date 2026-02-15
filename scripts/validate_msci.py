#!/usr/bin/env python3
"""
MSCI Validation Script

Validates the Multimodal Semantic Coherence Index through:
1. Sensitivity analysis (RQ1)
2. Human correlation analysis (RQ3)
3. Threshold calibration

Usage:
    python scripts/validate_msci.py --experiment-results runs/experiment/results.json
    python scripts/validate_msci.py --human-eval evaluation/human_eval_sessions/session.json
    python scripts/validate_msci.py --full-validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.msci_sensitivity import MSCISensitivityAnalyzer
from src.validation.human_correlation import HumanCorrelationAnalyzer
from src.validation.threshold_calibration import ThresholdCalibrator


def run_sensitivity_analysis(experiment_path: Path, output_dir: Path):
    """Run MSCI sensitivity analysis from experiment results."""
    print(f"\n{'=' * 70}")
    print("RQ1: MSCI SENSITIVITY ANALYSIS")
    print(f"{'=' * 70}")

    analyzer = MSCISensitivityAnalyzer()
    results = analyzer.analyze_from_experiment_results(experiment_path)

    # Print results
    print(f"\nConditions analyzed: {results['conditions_analyzed']}")
    print("\nSensitivity Tests:")

    for test_name, test_result in results.get("sensitivity_tests", {}).items():
        sig = "***" if test_result.get("significant") else ""
        print(f"\n  {test_name}:")
        print(f"    Baseline mean:  {test_result['baseline_mean']:.4f}")
        print(f"    Perturbed mean: {test_result['perturbed_mean']:.4f}")
        print(f"    Mean drop:      {test_result['mean_drop']:.4f} ({test_result['percent_drop']:.1f}%)")
        print(f"    Effect size d:  {test_result['effect_size_d']:.3f}")
        print(f"    p-value:        {test_result['p_value']:.4f} {sig}")
        print(f"    → {test_result['interpretation']}")

    # Summary
    summary = results.get("summary", {})
    print(f"\n{'=' * 50}")
    print("SENSITIVITY SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Tests performed: {summary.get('n_tests', 0)}")
    print(f"  Significant: {summary.get('n_significant', 0)} ({summary.get('sensitivity_rate', 0):.1%})")
    print(f"  Average effect: d={summary.get('average_effect_size', 0):.3f}")
    print(f"  Average drop: {summary.get('average_percent_drop', 0):.1f}%")
    print(f"\n  VERDICT: {summary.get('verdict', 'N/A')}")

    # Save report
    report = analyzer.generate_report(sensitivity_tests=results.get("sensitivity_tests"))
    report_path = output_dir / "sensitivity_analysis.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    return results


def run_human_correlation(human_eval_path: Path, output_dir: Path):
    """Run MSCI-human correlation analysis."""
    print(f"\n{'=' * 70}")
    print("RQ3: MSCI-HUMAN CORRELATION ANALYSIS")
    print(f"{'=' * 70}")

    analyzer = HumanCorrelationAnalyzer()
    results = analyzer.analyze_from_human_eval(human_eval_path)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return results

    # Print results
    print(f"\nPaired samples: {results['n_pairs']}")

    print("\n--- Overall Correlation ---")
    overall = results["overall_correlation"]
    print(f"  Spearman ρ: {overall['spearman_rho']:.4f} (p={overall['spearman_p']:.4f})")
    print(f"  Pearson r:  {overall['pearson_r']:.4f} (p={overall['pearson_p']:.4f})")
    print(f"  95% CI:     [{overall['ci_95'][0]:.4f}, {overall['ci_95'][1]:.4f}]")
    print(f"  → {overall['interpretation']}")

    print("\n--- Per-Dimension Correlations ---")
    for dim_name, dim_result in results["per_dimension"].items():
        sig = "*" if dim_result.get("significant") else ""
        print(f"  {dim_name}: ρ={dim_result['spearman_rho']:.4f} (p={dim_result['spearman_p']:.4f}){sig}")

    # RQ3 Verdict
    verdict = results.get("rq3_verdict", {})
    print(f"\n{'=' * 50}")
    print("RQ3 VERDICT")
    print(f"{'=' * 50}")
    print(f"  Status: {verdict.get('verdict', 'N/A')}")
    print(f"  {verdict.get('explanation', '')}")
    print(f"  Threshold (ρ > 0.3): {'MET' if verdict.get('threshold_met') else 'NOT MET'}")

    # Save report
    report = analyzer.generate_report(results)
    report_path = output_dir / "human_correlation.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    return results


def run_threshold_calibration(human_eval_path: Path, output_dir: Path):
    """Run MSCI threshold calibration."""
    print(f"\n{'=' * 70}")
    print("MSCI THRESHOLD CALIBRATION")
    print(f"{'=' * 70}")

    calibrator = ThresholdCalibrator(human_threshold=0.6)

    try:
        result = calibrator.calibrate_from_human_eval(human_eval_path)
    except ValueError as e:
        print(f"\nError: {e}")
        return None

    # Print results
    print(f"\n--- ROC Analysis ---")
    print(f"  AUC: {result.auc:.4f}")

    if result.auc >= 0.9:
        print("  Interpretation: Excellent discrimination")
    elif result.auc >= 0.8:
        print("  Interpretation: Good discrimination")
    elif result.auc >= 0.7:
        print("  Interpretation: Acceptable discrimination")
    else:
        print("  Interpretation: Poor discrimination")

    print(f"\n--- Optimal Threshold (Youden's J) ---")
    print(f"  Threshold: {result.optimal_threshold:.4f}")
    print(f"  Youden's J: {result.youden_j:.4f}")

    print(f"\n--- Performance at Optimal Threshold ---")
    print(f"  Sensitivity: {result.sensitivity_at_optimal:.1%}")
    print(f"  Specificity: {result.specificity_at_optimal:.1%}")
    print(f"  Precision: {result.precision_at_optimal:.1%}")
    print(f"  F1 Score: {result.f1_at_optimal:.4f}")

    print(f"\n--- Recommendation ---")
    print(f"  Use MSCI >= {result.optimal_threshold:.4f} to classify as 'coherent'")

    # Save report
    report = calibrator.generate_report(result)
    report_path = output_dir / "threshold_calibration.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport saved to: {report_path}")

    return result


def run_full_validation(
    experiment_path: Path,
    human_eval_path: Path,
    output_dir: Path,
):
    """Run complete MSCI validation suite."""
    print(f"\n{'#' * 70}")
    print("# COMPREHENSIVE MSCI VALIDATION")
    print(f"{'#' * 70}")

    results = {
        "experiment_source": str(experiment_path),
        "human_eval_source": str(human_eval_path),
    }

    # RQ1: Sensitivity
    print("\n[1/3] Running sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(experiment_path, output_dir)
    results["rq1_sensitivity"] = sensitivity.get("summary", {})

    # RQ3: Human correlation
    print("\n[2/3] Running human correlation analysis...")
    correlation = run_human_correlation(human_eval_path, output_dir)
    results["rq3_correlation"] = correlation.get("rq3_verdict", {})

    # Threshold calibration
    print("\n[3/3] Running threshold calibration...")
    calibration = run_threshold_calibration(human_eval_path, output_dir)
    if calibration:
        results["calibration"] = {
            "optimal_threshold": calibration.optimal_threshold,
            "auc": calibration.auc,
            "f1": calibration.f1_at_optimal,
        }

    # Overall verdict
    print(f"\n{'=' * 70}")
    print("OVERALL MSCI VALIDATION VERDICT")
    print(f"{'=' * 70}")

    verdicts = []

    # RQ1
    sens_rate = sensitivity.get("summary", {}).get("sensitivity_rate", 0)
    if sens_rate > 0.5:
        verdicts.append(("RQ1 Sensitivity", "PASS", f"{sens_rate:.0%} of perturbations detected"))
    else:
        verdicts.append(("RQ1 Sensitivity", "FAIL", f"Only {sens_rate:.0%} detected"))

    # RQ3
    rq3_verdict = correlation.get("rq3_verdict", {}).get("verdict", "N/A")
    rq3_rho = correlation.get("rq3_verdict", {}).get("rho", 0)
    if rq3_verdict in ["SUPPORTED", "WEAKLY SUPPORTED"]:
        verdicts.append(("RQ3 Correlation", "PASS", f"ρ = {rq3_rho:.3f}"))
    else:
        verdicts.append(("RQ3 Correlation", "FAIL", f"ρ = {rq3_rho:.3f}"))

    # Calibration
    if calibration:
        if calibration.auc >= 0.7:
            verdicts.append(("Calibration", "PASS", f"AUC = {calibration.auc:.3f}"))
        else:
            verdicts.append(("Calibration", "FAIL", f"AUC = {calibration.auc:.3f}"))

    for name, status, detail in verdicts:
        print(f"  {name}: {status} ({detail})")

    n_pass = sum(1 for _, s, _ in verdicts if s == "PASS")
    n_total = len(verdicts)

    print(f"\n  Overall: {n_pass}/{n_total} validation criteria met")

    if n_pass == n_total:
        print("\n  ✓ MSCI IS VALIDATED as a reliable coherence metric")
    elif n_pass > 0:
        print("\n  ~ MSCI is PARTIALLY VALIDATED - use with caution")
    else:
        print("\n  ✗ MSCI validation FAILED - metric may be unreliable")

    # Save combined report
    results["verdicts"] = [
        {"criterion": n, "status": s, "detail": d}
        for n, s, d in verdicts
    ]
    results["overall_pass_rate"] = n_pass / n_total if n_total > 0 else 0

    combined_path = output_dir / "validation_summary.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nCombined report saved to: {combined_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate MSCI metric"
    )

    parser.add_argument(
        "--experiment-results",
        type=str,
        help="Path to experiment results JSON (for sensitivity analysis)",
    )
    parser.add_argument(
        "--human-eval",
        type=str,
        help="Path to human evaluation session JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/validation",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Run only sensitivity analysis",
    )
    parser.add_argument(
        "--correlation-only",
        action="store_true",
        help="Run only human correlation analysis",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run only threshold calibration",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run complete validation suite",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to run
    if args.full_validation:
        if not args.experiment_results or not args.human_eval:
            print("Error: --full-validation requires both --experiment-results and --human-eval")
            sys.exit(1)
        run_full_validation(
            Path(args.experiment_results),
            Path(args.human_eval),
            output_dir,
        )

    elif args.sensitivity_only:
        if not args.experiment_results:
            print("Error: --sensitivity-only requires --experiment-results")
            sys.exit(1)
        run_sensitivity_analysis(Path(args.experiment_results), output_dir)

    elif args.correlation_only:
        if not args.human_eval:
            print("Error: --correlation-only requires --human-eval")
            sys.exit(1)
        run_human_correlation(Path(args.human_eval), output_dir)

    elif args.calibration_only:
        if not args.human_eval:
            print("Error: --calibration-only requires --human-eval")
            sys.exit(1)
        run_threshold_calibration(Path(args.human_eval), output_dir)

    else:
        # Default: run what we can based on provided inputs
        if args.experiment_results:
            run_sensitivity_analysis(Path(args.experiment_results), output_dir)
        if args.human_eval:
            run_human_correlation(Path(args.human_eval), output_dir)
            run_threshold_calibration(Path(args.human_eval), output_dir)

        if not args.experiment_results and not args.human_eval:
            parser.print_help()
            print("\nError: Provide at least one of --experiment-results or --human-eval")
            sys.exit(1)


if __name__ == "__main__":
    main()
