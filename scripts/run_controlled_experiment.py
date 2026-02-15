"""
Controlled Experiment Runner

Implements exactly 6 experimental conditions:
1. Direct + Baseline
2. Direct + Wrong Image
3. Direct + Wrong Audio
4. Planner + Baseline
5. Planner + Wrong Image
6. Planner + Wrong Audio

This addresses the question: "Is this structure or just more prompting?"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from src.pipeline.generate_and_evaluate import generate_and_evaluate
from src.utils.seed import set_global_seed


# Experimental conditions
CONDITIONS = [
    ("direct", "baseline"),
    ("direct", "wrong_image"),
    ("direct", "wrong_audio"),
    ("planner", "baseline"),
    ("planner", "wrong_image"),
    ("planner", "wrong_audio"),
]


def run_controlled_experiment(
    prompt: str,
    out_dir: str = "runs/controlled_experiment",
    seed: int = 42,
    use_ollama: bool = True,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Run all 6 experimental conditions for a single prompt.
    
    Returns:
        Dictionary with results for each condition
    """
    if deterministic:
        set_global_seed(seed)
    
    results = {}
    
    print(f"\n{'=' * 60}")
    print(f"CONTROLLED EXPERIMENT: {prompt[:60]}...")
    print(f"{'=' * 60}")
    
    for mode, condition in CONDITIONS:
        condition_key = f"{mode}_{condition}"
        print(f"\n[{condition_key}] Running...")
        
        try:
            bundle = generate_and_evaluate(
                prompt=prompt,
                out_dir=out_dir,
                use_ollama=use_ollama,
                deterministic=deterministic,
                seed=seed,
                mode=mode,
                condition=condition,
            )
            
            results[condition_key] = {
                "run_id": bundle.run_id,
                "prompt": prompt,
                "mode": mode,
                "condition": condition,
                "scores": bundle.scores,
                "coherence": bundle.coherence,
                "semantic_drift": bundle.semantic_drift,
                "meta": bundle.meta,
            }
            
            print(f"  ✓ MSCI: {bundle.scores.get('msci', 0):.4f}")
            print(f"  ✓ st_i: {bundle.scores.get('st_i', 0):.4f}")
            print(f"  ✓ st_a: {bundle.scores.get('st_a', 0):.4f}")
            print(f"  ✓ si_a: {bundle.scores.get('si_a', 0):.4f}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results[condition_key] = {
                "error": str(e),
                "prompt": prompt,
                "mode": mode,
                "condition": condition,
            }
    
    return results


def run_batch_controlled_experiment(
    prompts: List[str],
    out_dir: str = "runs/controlled_experiment",
    seed: int = 42,
    use_ollama: bool = True,
    deterministic: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run controlled experiments for multiple prompts.
    
    Returns:
        Dictionary organized by condition, containing results for all prompts
    """
    all_results = {condition_key: [] for mode, condition in CONDITIONS for condition_key in [f"{mode}_{condition}"]}
    
    for idx, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"PROMPT {idx + 1}/{len(prompts)}")
        print(f"{'=' * 60}")
        
        results = run_controlled_experiment(
            prompt=prompt,
            out_dir=out_dir,
            seed=seed + idx,  # Different seed per prompt for reproducibility
            use_ollama=use_ollama,
            deterministic=deterministic,
        )
        
        # Organize by condition
        for condition_key, result in results.items():
            all_results[condition_key].append(result)
    
    return all_results


def main():
    """Main entry point for controlled experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run controlled experiment")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--prompts-file", type=str, help="JSON file with list of prompts")
    parser.add_argument("--out-dir", type=str, default="runs/controlled_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ollama", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        # Default test prompts
        prompts = [
            "A calm foggy forest at dawn with distant birds and soft wind",
            "A rainy neon-lit city street at night with reflections on wet pavement",
        ]
    
    # Run experiments
    results = run_batch_controlled_experiment(
        prompts=prompts,
        out_dir=args.out_dir,
        seed=args.seed,
        use_ollama=args.use_ollama,
        deterministic=True,
    )
    
    # Save results
    out_path = Path(args.out_dir) / "controlled_experiment_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Total conditions: {len(CONDITIONS)}")
    print(f"Total runs: {len(prompts) * len(CONDITIONS)}")


if __name__ == "__main__":
    main()
