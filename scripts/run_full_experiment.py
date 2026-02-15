#!/usr/bin/env python3
"""
Full Experiment Runner

Executes complete controlled experiments with:
- Stratified prompt sampling
- Multiple conditions (modes × perturbations)
- Statistical analysis with hypothesis testing
- Comprehensive result reporting

Usage:
    python scripts/run_full_experiment.py --preset rq2_planning
    python scripts/run_full_experiment.py --config experiments/my_config.json
    python scripts/run_full_experiment.py --n-prompts 50 --modes direct single_planner
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.config import (
    ExperimentConfig,
    PlannerMode,
    PerturbationType,
    PRESETS,
)
from src.experiments.statistical_analysis import (
    paired_ttest,
    compare_all_pairs,
    descriptive_stats,
    spearman_correlation,
)
from src.experiments.prompt_sampler import PromptSampler, create_experiment_prompts
from src.pipeline.generate_and_evaluate import generate_and_evaluate
from src.utils.seed import set_global_seed


def run_single_condition(
    prompt: str,
    mode: str,
    perturbation: str,
    seed: int,
    output_dir: Path,
    use_ollama: bool = True,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Run a single experimental condition.

    Returns:
        Dictionary with scores, coherence, and metadata
    """
    try:
        bundle = generate_and_evaluate(
            prompt=prompt,
            out_dir=str(output_dir),
            use_ollama=use_ollama,
            deterministic=deterministic,
            seed=seed,
            mode=mode,
            condition=perturbation,
        )

        return {
            "success": True,
            "run_id": bundle.run_id,
            "scores": bundle.scores,
            "coherence": bundle.coherence,
            "semantic_drift": bundle.semantic_drift,
            "meta": bundle.meta,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def run_experiment(
    config: ExperimentConfig,
    prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run complete experiment according to configuration.

    Args:
        config: Experiment configuration
        prompts: Optional list of prompts (will sample if not provided)

    Returns:
        Complete experiment results with statistical analysis
    """
    # Validate configuration
    warnings = config.validate()
    for warning in warnings:
        print(f"WARNING: {warning}")

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"{config.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(output_dir / "config.json")

    # Get prompts
    if prompts is None:
        print(f"Sampling {config.n_prompts} prompts...")
        sampler = PromptSampler(seed=config.base_seed)
        sampled = sampler.sample_stratified(n_total=config.n_prompts)
        prompts = [p.text for p in sampled]
        sampler.save_sample_set(sampled, output_dir / "prompts.json")

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'=' * 70}")
    print(f"Prompts: {len(prompts)}")
    print(f"Seeds per prompt: {config.n_seeds}")
    print(f"Modes: {[m.value for m in config.modes]}")
    print(f"Perturbations: {[p.value for p in config.perturbations]}")
    print(f"Total runs: {config.total_runs}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # Collect results by condition
    results_by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_results: List[Dict[str, Any]] = []

    total_runs = 0
    successful_runs = 0

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")

        for seed_offset in range(config.n_seeds):
            seed = config.base_seed + prompt_idx * 100 + seed_offset

            for mode in config.modes:
                for perturbation in config.perturbations:
                    condition_key = f"{mode.value}_{perturbation.value}"
                    total_runs += 1

                    if config.deterministic:
                        set_global_seed(seed)

                    print(f"  [{condition_key}] seed={seed}...", end=" ")

                    result = run_single_condition(
                        prompt=prompt,
                        mode=mode.value,
                        perturbation=perturbation.value,
                        seed=seed,
                        output_dir=output_dir / condition_key,
                        use_ollama=config.use_ollama,
                        deterministic=config.deterministic,
                    )

                    result["prompt"] = prompt
                    result["prompt_idx"] = prompt_idx
                    result["seed"] = seed
                    result["mode"] = mode.value
                    result["perturbation"] = perturbation.value
                    result["condition"] = condition_key

                    if result["success"]:
                        msci = result["scores"].get("msci", 0)
                        print(f"MSCI={msci:.4f}")
                        successful_runs += 1
                    else:
                        print(f"ERROR: {result['error'][:50]}")

                    results_by_condition[condition_key].append(result)
                    all_results.append(result)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE: {successful_runs}/{total_runs} runs successful")
    print(f"{'=' * 70}")

    # Extract MSCI scores by condition for statistical analysis
    msci_by_condition: Dict[str, List[float]] = {}
    for condition, results in results_by_condition.items():
        scores = [
            r["scores"]["msci"]
            for r in results
            if r["success"] and "msci" in r.get("scores", {})
        ]
        msci_by_condition[condition] = scores

    # Compute descriptive statistics
    print("\n--- Descriptive Statistics ---")
    condition_stats: Dict[str, Dict[str, float]] = {}
    for condition, scores in msci_by_condition.items():
        if scores:
            stats = descriptive_stats(scores)
            condition_stats[condition] = stats
            print(f"  {condition}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")

    # Statistical tests
    statistical_results: Dict[str, Any] = {}

    # Test RQ1: MSCI sensitivity (baseline vs perturbed)
    print("\n--- RQ1: MSCI Sensitivity ---")
    for mode in config.modes:
        baseline_key = f"{mode.value}_baseline"
        baseline_scores = msci_by_condition.get(baseline_key, [])

        for perturbation in config.perturbations:
            if perturbation == PerturbationType.BASELINE:
                continue

            perturbed_key = f"{mode.value}_{perturbation.value}"
            perturbed_scores = msci_by_condition.get(perturbed_key, [])

            if len(baseline_scores) >= 2 and len(perturbed_scores) >= 2:
                # Ensure same length for paired test
                n = min(len(baseline_scores), len(perturbed_scores))
                result = paired_ttest(baseline_scores[:n], perturbed_scores[:n])
                test_key = f"{baseline_key}_vs_{perturbed_key}"
                statistical_results[test_key] = result.to_dict()

                sig = "*" if result.significant else ""
                print(f"  {test_key}: d={result.effect_size:.3f}, p={result.p_value:.4f}{sig}")

    # Test RQ2: Planning effect (direct vs planner for baseline)
    print("\n--- RQ2: Planning Effect ---")
    baseline_conditions = {
        c: scores for c, scores in msci_by_condition.items()
        if c.endswith("_baseline")
    }

    if len(baseline_conditions) >= 2:
        comparisons = compare_all_pairs(baseline_conditions, paired=True)
        for key, result in comparisons.items():
            statistical_results[key] = result.to_dict()
            sig = "*" if result.significant else ""
            print(f"  {key}: d={result.effect_size:.3f}, p={result.p_value:.4f}{sig}")

    # Compile final report
    report = {
        "experiment": config.to_dict(),
        "timestamp": timestamp,
        "n_prompts": len(prompts),
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
        "condition_statistics": condition_stats,
        "statistical_tests": statistical_results,
        "raw_results": all_results,
    }

    # Save results
    results_path = output_dir / "experiment_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {results_path}")

    # Generate summary
    _print_summary(report)

    return report


def _print_summary(report: Dict[str, Any]):
    """Print formatted experiment summary."""
    print(f"\n{'=' * 70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")

    stats = report.get("condition_statistics", {})
    tests = report.get("statistical_tests", {})

    # Best and worst conditions
    if stats:
        sorted_conditions = sorted(stats.items(), key=lambda x: x[1].get("mean", 0), reverse=True)
        print("\nConditions ranked by mean MSCI:")
        for i, (cond, s) in enumerate(sorted_conditions, 1):
            print(f"  {i}. {cond}: {s['mean']:.4f} ± {s['std']:.4f}")

    # Significant findings
    significant_tests = {k: v for k, v in tests.items() if v.get("significant")}
    if significant_tests:
        print("\nSignificant findings (p < 0.05):")
        for test, result in significant_tests.items():
            print(f"  - {test}: d={result['effect_size']:.3f}, p={result['p_value']:.4f}")
    else:
        print("\nNo significant findings at α=0.05")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled multimodal coherence experiments"
    )

    # Configuration options
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration JSON",
    )

    # Override options
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--n-prompts", type=int, help="Number of prompts")
    parser.add_argument("--n-seeds", type=int, help="Seeds per prompt")
    parser.add_argument("--modes", nargs="+", help="Planner modes")
    parser.add_argument("--perturbations", nargs="+", help="Perturbation types")
    parser.add_argument("--seed", type=int, help="Base random seed")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Prompts
    parser.add_argument("--prompts-file", type=str, help="File with prompts")
    parser.add_argument("--prompts", nargs="+", help="Direct prompt list")

    # Execution
    parser.add_argument("--no-ollama", action="store_true", help="Don't use Ollama")

    args = parser.parse_args()

    # Build configuration
    if args.preset:
        config = PRESETS[args.preset]
    elif args.config:
        config = ExperimentConfig.load(Path(args.config))
    else:
        config = ExperimentConfig()

    # Apply overrides
    if args.name:
        config.name = args.name
    if args.n_prompts:
        config.n_prompts = args.n_prompts
    if args.n_seeds:
        config.n_seeds = args.n_seeds
    if args.modes:
        config.modes = [PlannerMode(m) for m in args.modes]
    if args.perturbations:
        config.perturbations = [PerturbationType(p) for p in args.perturbations]
    if args.seed:
        config.base_seed = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_ollama:
        config.use_ollama = False

    # Load prompts
    prompts = None
    if args.prompts:
        prompts = args.prompts
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            elif "prompts" in data:
                prompts = [p["text"] if isinstance(p, dict) else p for p in data["prompts"]]

    # Run experiment
    run_experiment(config, prompts)


if __name__ == "__main__":
    main()
