#!/usr/bin/env python3
"""
Ablation Study Runner

Runs ablation experiments to answer:
"Is Council-Lite's benefit from structure or just more prompting?"

Four conditions:
1. Direct: No planning
2. Single Planner: 1 LLM call
3. Council-Lite: 3 LLM calls (multi-agent)
4. Extended Prompt: 1 LLM call with 3× tokens

Usage:
    python scripts/run_ablation_study.py --n-prompts 50
    python scripts/run_ablation_study.py --conditions direct single_planner council
    python scripts/run_ablation_study.py --prompts-file prompts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.ablation_runner import (
    AblationRunner,
    AblationStudyConfig,
    ABLATION_CONDITIONS,
)
from src.experiments.prompt_sampler import PromptSampler, create_experiment_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Run Council-Lite ablation study"
    )

    # Study configuration
    parser.add_argument(
        "--name",
        type=str,
        default="council_lite_ablation",
        help="Study name",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(ABLATION_CONDITIONS.keys()),
        default=["direct", "single_planner", "council", "extended_prompt"],
        help="Conditions to test",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=50,
        help="Number of prompts",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Seeds per prompt for replication",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/ablation_study",
        help="Output directory",
    )

    # Prompts
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="JSON file with prompts",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Direct list of prompts",
    )

    # Execution
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Don't use Ollama (use API instead)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (5 prompts, 1 seed)",
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick:
        args.n_prompts = 5
        args.n_seeds = 1
        args.conditions = ["direct", "single_planner"]
        print("Quick test mode: 5 prompts, 1 seed, 2 conditions")

    # Create configuration
    config = AblationStudyConfig(
        name=args.name,
        conditions=args.conditions,
        n_prompts=args.n_prompts,
        n_seeds=args.n_seeds,
        base_seed=args.seed,
        output_dir=args.output_dir,
        use_ollama=not args.no_ollama,
        deterministic=True,
    )

    # Load or generate prompts
    if args.prompts:
        prompts = args.prompts
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            elif "prompts" in data:
                prompts = [p["text"] if isinstance(p, dict) else p for p in data["prompts"]]
            else:
                raise ValueError(f"Unexpected format in {args.prompts_file}")
    else:
        # Generate stratified prompts
        print(f"Generating {args.n_prompts} stratified prompts...")
        sampled = create_experiment_prompts(n_prompts=args.n_prompts, seed=args.seed)
        prompts = [p.text for p in sampled]

    # Validate
    if len(prompts) < args.n_prompts:
        print(f"Warning: Only {len(prompts)} prompts available, requested {args.n_prompts}")

    # Print study info
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Name: {config.name}")
    print(f"Conditions: {config.conditions}")
    for cond in config.conditions:
        desc = ABLATION_CONDITIONS[cond].description
        print(f"  - {cond}: {desc}")
    print(f"Prompts: {len(prompts)} (requested: {args.n_prompts})")
    print(f"Seeds per prompt: {config.n_seeds}")
    print(f"Total runs: {config.total_runs}")
    print(f"Output: {config.output_dir}")
    print(f"{'=' * 70}")

    # Confirm
    response = input("\nProceed with ablation study? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Run study
    runner = AblationRunner(config)
    report = runner.run_study(prompts)

    # Print conclusions
    print(f"\n{'=' * 70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'=' * 70}")

    ablation = report.get("ablation_analysis", {})
    rqs = ablation.get("research_questions", {})

    if "planning_effect" in rqs:
        pe = rqs["planning_effect"]
        print(f"\n1. Planning Effect:")
        print(f"   Direct: {pe['direct_mean']:.4f}")
        print(f"   Single Planner: {pe['single_planner_mean']:.4f}")
        print(f"   Δ = {pe['difference']:+.4f}")
        print(f"   → {pe['interpretation']}")

    if "council_structure" in rqs:
        cs = rqs["council_structure"]
        print(f"\n2. Council Structure:")
        print(f"   Single Planner: {cs['single_planner_mean']:.4f}")
        print(f"   Council: {cs['council_mean']:.4f}")
        print(f"   Δ = {cs['difference']:+.4f}")
        print(f"   → {cs['interpretation']}")

    if "token_control" in rqs:
        tc = rqs["token_control"]
        print(f"\n3. Token Control (KEY QUESTION):")
        print(f"   Extended Prompt (3× tokens): {tc['extended_prompt_mean']:.4f}")
        print(f"   Council (3 calls): {tc['council_mean']:.4f}")
        print(f"   Δ = {tc['difference']:+.4f}")
        print(f"   → {tc['interpretation']}")

    conclusions = ablation.get("conclusions", [])
    if conclusions:
        print(f"\n{'=' * 70}")
        print("MAIN CONCLUSION:")
        for conclusion in conclusions:
            print(f"  {conclusion}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
