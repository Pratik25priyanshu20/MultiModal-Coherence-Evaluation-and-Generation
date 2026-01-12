"""
Phase 3: Verify generator conditioning consistency.

Checks that all generators use ONLY semantic plan, not raw prompt.
Run this to audit conditioning strictness across a batch of runs.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from src.pipeline.conditioning_validator import detect_prompt_leakage, check_plan_completeness


def analyze_bundle(bundle_path: Path) -> Dict[str, Any]:
    """Analyze a single bundle for conditioning issues."""
    with bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    
    original_prompt = bundle.get("prompt", "")
    plan = bundle.get("meta", {}).get("semantic_plan", {})
    modality_prompts = bundle.get("meta", {}).get("modality_prompts", {})
    
    leakage_check = detect_prompt_leakage(
        original_prompt=original_prompt,
        plan=plan,
        text_prompt=modality_prompts.get("text_prompt", ""),
        image_prompt=modality_prompts.get("image_prompt", ""),
        audio_prompt=modality_prompts.get("audio_prompt", ""),
    )
    
    completeness = check_plan_completeness(plan)
    
    return {
        "run_id": bundle.get("run_id"),
        "prompt": original_prompt[:60] + "..." if len(original_prompt) > 60 else original_prompt,
        "has_leakage": leakage_check["has_leakage"],
        "leakage_details": leakage_check["leakage_details"],
        "plan_coverage": leakage_check["plan_coverage"],
        "plan_completeness": completeness,
        "msci": bundle.get("scores", {}).get("msci"),
    }


def main():
    results_dir = Path("runs/laion_eval")
    
    # Find all bundle.json files
    bundle_files = list(results_dir.glob("**/bundle.json"))
    
    if not bundle_files:
        print(f"No bundle.json files found in {results_dir}")
        print("Run run_laion_eval.py first to generate bundles.")
        return
    
    print("=" * 60)
    print("Conditioning Consistency Verification")
    print("=" * 60)
    print(f"Analyzing {len(bundle_files)} bundles...")
    print()
    
    results = []
    leakage_count = 0
    incomplete_plans = 0
    
    for bundle_path in bundle_files:
        try:
            analysis = analyze_bundle(bundle_path)
            results.append(analysis)
            
            if analysis["has_leakage"]:
                leakage_count += 1
            if not analysis["plan_completeness"]["is_complete"]:
                incomplete_plans += 1
        except Exception as e:
            print(f"Error analyzing {bundle_path}: {e}")
    
    # Save analysis
    analysis_file = results_dir / "conditioning_analysis.json"
    with analysis_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved analysis to {analysis_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONDITIONING VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total bundles analyzed: {len(results)}")
    print(f"Bundles with prompt leakage: {leakage_count} ({leakage_count/len(results)*100:.1f}%)")
    print(f"Bundles with incomplete plans: {incomplete_plans} ({incomplete_plans/len(results)*100:.1f}%)")
    print()
    
    # Show examples with leakage
    leakage_examples = [r for r in results if r["has_leakage"]]
    if leakage_examples:
        print("Examples with prompt leakage:")
        for ex in leakage_examples[:5]:
            print(f"  Run: {ex['run_id']}")
            print(f"    Prompt: {ex['prompt']}")
            leakage = ex["leakage_details"]
            if leakage["text"]:
                print(f"    Text leakage: {leakage['text']}")
            if leakage["image"]:
                print(f"    Image leakage: {leakage['image']}")
            if leakage["audio"]:
                print(f"    Audio leakage: {leakage['audio']}")
            print()
    
    # Show plan completeness stats
    avg_coverage = sum(r["plan_coverage"] for r in results) / len(results) if results else 0
    print(f"Average plan coverage: {avg_coverage:.2%}")
    
    completeness_stats = {
        "has_scene": sum(1 for r in results if r["plan_completeness"]["has_scene"]),
        "has_visual": sum(1 for r in results if r["plan_completeness"]["has_visual"]),
        "has_audio": sum(1 for r in results if r["plan_completeness"]["has_audio"]),
        "has_entities": sum(1 for r in results if r["plan_completeness"]["has_entities"]),
    }
    print("\nPlan completeness breakdown:")
    for key, count in completeness_stats.items():
        print(f"  {key}: {count}/{len(results)} ({count/len(results)*100:.1f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
