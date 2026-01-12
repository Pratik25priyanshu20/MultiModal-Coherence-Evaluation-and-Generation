"""
Phase 4: Metric calibration and threshold derivation.

This script:
1. Runs batch perturbation experiments to build distributions
2. Computes normalization parameters from baseline distributions
3. Analyzes separation between baseline and wrong-modality distributions
4. Derives calibrated thresholds based on separation
5. Saves calibration config for use by coherence engine
"""

import json
import random
from pathlib import Path
from statistics import mean, stdev, median
from typing import Dict, List, Any, Tuple
import numpy as np

from src.pipeline.generate_and_evaluate import generate_and_evaluate


def run_perturbation_batch(
    prompts: List[str],
    n_per_prompt: int = 5,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run perturbation experiments for multiple prompts.
    
    Returns:
        {
            "baseline": [...],
            "wrong_image": [...],
            "wrong_audio": [...],
            "wrong_text": [...]
        }
    """
    rng = random.Random(seed)
    results = {
        "baseline": [],
        "wrong_image": [],
        "wrong_audio": [],
        "wrong_text": [],
    }
    
    failed_prompts = []
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {prompt_idx+1}/{len(prompts)}: {prompt[:60]}...")
        print('='*60)
        
        prompt_failed = False
        
        for run_idx in range(n_per_prompt):
            run_seed = seed + prompt_idx * 1000 + run_idx
            
            try:
                # BASELINE
                print(f"\n[Run {run_idx+1}/{n_per_prompt}] Baseline...")
                baseline = generate_and_evaluate(
                    prompt=prompt,
                    out_dir="runs/calibration",
                    use_ollama=True,
                    deterministic=True,
                    seed=run_seed,
                )
                results["baseline"].append({
                    "prompt": prompt,
                    "scores": baseline.scores,
                    "run_id": baseline.run_id,
                })
                
                # WRONG IMAGE
                print(f"[Run {run_idx+1}/{n_per_prompt}] Wrong image...")
                wrong_img_bundle = generate_and_evaluate(
                    prompt=prompt,
                    out_dir="runs/calibration",
                    use_ollama=True,
                    deterministic=True,
                    seed=run_seed + 10000,  # Different seed for wrong variant
                )
                # Replace with random image
                images = list(Path("data/processed/images").glob("*.png"))
                if images and baseline.image_path != str(images[0]):
                    wrong_img_bundle.image_path = str(rng.choice(images))
                    # Re-evaluate with wrong image
                    from src.coherence.coherence_engine import evaluate_coherence
                    eval_out = evaluate_coherence(
                        text=wrong_img_bundle.generated_text,
                        image_path=wrong_img_bundle.image_path,
                        audio_path=wrong_img_bundle.audio_path,
                    )
                    wrong_img_bundle.scores = eval_out.get("scores", {})
                
                results["wrong_image"].append({
                    "prompt": prompt,
                    "scores": wrong_img_bundle.scores,
                    "run_id": wrong_img_bundle.run_id,
                })
                
                # WRONG AUDIO
                print(f"[Run {run_idx+1}/{n_per_prompt}] Wrong audio...")
                wrong_aud_bundle = generate_and_evaluate(
                    prompt=prompt,
                    out_dir="runs/calibration",
                    use_ollama=True,
                    deterministic=True,
                    seed=run_seed + 20000,
                )
                # Replace with random audio
                audios = list(Path("data/wikimedia/audio").glob("*.wav"))
                if audios and baseline.audio_path != str(audios[0]):
                    wrong_aud_bundle.audio_path = str(rng.choice(audios))
                    # Re-evaluate with wrong audio
                    from src.coherence.coherence_engine import evaluate_coherence
                    eval_out = evaluate_coherence(
                        text=wrong_aud_bundle.generated_text,
                        image_path=wrong_aud_bundle.image_path,
                        audio_path=wrong_aud_bundle.audio_path,
                    )
                    wrong_aud_bundle.scores = eval_out.get("scores", {})
                
                results["wrong_audio"].append({
                    "prompt": prompt,
                    "scores": wrong_aud_bundle.scores,
                    "run_id": wrong_aud_bundle.run_id,
                })
                
            except Exception as e:
                print(f"\n✗ ERROR in run {run_idx+1} for prompt '{prompt[:50]}...': {e}")
                print(f"  Continuing with next run...")
                # Mark prompt as failed if all runs fail, but continue
                if run_idx == n_per_prompt - 1:
                    prompt_failed = True
        
        if prompt_failed:
            failed_prompts.append(prompt)
            print(f"\n⚠ Warning: Prompt '{prompt[:50]}...' had failures, but continuing...")
    
    if failed_prompts:
        print(f"\n⚠ {len(failed_prompts)} prompts had failures, but continuing with calibration...")
    
    # Add metadata about failed prompts
    results["_metadata"] = {
        "failed_prompts": failed_prompts,
        "total_prompts": len(prompts),
        "successful_prompts": len(prompts) - len(failed_prompts),
    }
    
    return results


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute distribution statistics."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q75": 0.0}
    
    sorted_vals = sorted(values)
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "q25": float(sorted_vals[len(sorted_vals) // 4]),
        "q75": float(sorted_vals[3 * len(sorted_vals) // 4]),
        "count": len(values),
    }


def compute_separation(
    baseline_values: List[float],
    wrong_values: List[float],
) -> Dict[str, float]:
    """Compute separation metrics between baseline and wrong distributions."""
    if not baseline_values or not wrong_values:
        return {"overlap": 1.0, "separation": 0.0, "threshold": 0.0}
    
    baseline_stats = compute_distribution_stats(baseline_values)
    wrong_stats = compute_distribution_stats(wrong_values)
    
    # Compute overlap (simplified: area under both curves in overlap region)
    baseline_mean = baseline_stats["mean"]
    baseline_std = baseline_stats["std"]
    wrong_mean = wrong_stats["mean"]
    wrong_std = wrong_stats["std"]
    
    # Separation: distance between means normalized by combined std
    combined_std = np.sqrt(baseline_std**2 + wrong_std**2)
    separation = (baseline_mean - wrong_mean) / max(combined_std, 1e-6)
    
    # Threshold: point that maximizes separation (simplified: midpoint)
    threshold = (baseline_mean + wrong_mean) / 2
    
    # Overlap estimate: how much distributions overlap
    overlap_region_min = max(baseline_stats["min"], wrong_stats["min"])
    overlap_region_max = min(baseline_stats["max"], wrong_stats["max"])
    overlap = max(0.0, (overlap_region_max - overlap_region_min) / 
                  max(baseline_stats["max"] - baseline_stats["min"], 1e-6))
    
    return {
        "separation": float(separation),
        "overlap": float(overlap),
        "threshold": float(threshold),
        "baseline_mean": baseline_mean,
        "wrong_mean": wrong_mean,
    }


def calibrate_metrics(perturbation_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute calibration parameters from perturbation results."""
    metrics = ["msci", "st_i", "st_a", "si_a"]
    
    calibration = {
        "normalization": {},
        "thresholds": {},
        "separation_analysis": {},
    }
    
    for metric in metrics:
        # Extract values
        baseline_values = [
            r["scores"].get(metric)
            for r in perturbation_results["baseline"]
            if r["scores"].get(metric) is not None
        ]
        
        wrong_image_values = [
            r["scores"].get(metric)
            for r in perturbation_results["wrong_image"]
            if r["scores"].get(metric) is not None
        ]
        
        wrong_audio_values = [
            r["scores"].get(metric)
            for r in perturbation_results["wrong_audio"]
            if r["scores"].get(metric) is not None
        ]
        
        # Compute normalization parameters from baseline
        baseline_stats = compute_distribution_stats(baseline_values)
        calibration["normalization"][metric] = {
            "mean": baseline_stats["mean"],
            "std": baseline_stats["std"],
            "min": baseline_stats["min"],
            "max": baseline_stats["max"],
        }
        
        # Compute separation for each wrong variant
        separations = {}
        if wrong_image_values:
            separations["wrong_image"] = compute_separation(baseline_values, wrong_image_values)
        if wrong_audio_values:
            separations["wrong_audio"] = compute_separation(baseline_values, wrong_audio_values)
        
        calibration["separation_analysis"][metric] = separations
        
        # Derive thresholds
        # Use worst-case separation (lowest separation = hardest to distinguish)
        all_separations = [s["separation"] for s in separations.values()]
        if all_separations:
            min_separation = min(all_separations)
            # Threshold: if separation is good (>1.0), use midpoint; otherwise be conservative
            if min_separation > 1.0:
                # Good separation: use midpoint between baseline and wrong
                worst_sep = min(separations.values(), key=lambda x: x["separation"])
                threshold = worst_sep["threshold"]
            else:
                # Poor separation: use conservative threshold (lower than baseline mean)
                threshold = baseline_stats["mean"] - baseline_stats["std"]
            
            calibration["thresholds"][metric] = {
                "low": float(threshold),
                "very_low": float(threshold - baseline_stats["std"]),
                "min_separation": float(min_separation),
            }
        else:
            # Fallback: use percentiles
            calibration["thresholds"][metric] = {
                "low": baseline_stats["q25"],
                "very_low": baseline_stats["min"],
                "min_separation": 0.0,
            }
    
    return calibration


def main():
    # Configuration
    CALIBRATION_PROMPTS = [
        "A quiet beach at night with gentle waves and distant wind",
        "A rainy neon-lit city street at night with reflections on wet pavement",
        "A calm forest at dawn with birdsong and soft mist",
        "A futuristic city skyline at sunset with flying vehicles",
    ]
    N_PER_PROMPT = 5
    SEED = 42
    
    print("=" * 60)
    print("Metric Calibration")
    print("=" * 60)
    print(f"Prompts: {len(CALIBRATION_PROMPTS)}")
    print(f"Runs per prompt: {N_PER_PROMPT}")
    print(f"Total runs: {len(CALIBRATION_PROMPTS) * N_PER_PROMPT * 3}")  # baseline + wrong_image + wrong_audio
    print()
    
    # Run perturbation batch
    perturbation_results = run_perturbation_batch(
        prompts=CALIBRATION_PROMPTS,
        n_per_prompt=N_PER_PROMPT,
        seed=SEED,
    )
    
    # Save raw perturbation results
    out_dir = Path("runs/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "perturbation_batch_results.json").open("w", encoding="utf-8") as f:
        json.dump(perturbation_results, f, indent=2)
    print(f"\n✓ Saved perturbation results to {out_dir / 'perturbation_batch_results.json'}")
    
    # Compute calibration
    calibration = calibrate_metrics(perturbation_results)
    
    # Save calibration config
    calibration_file = out_dir / "calibration_config.json"
    with calibration_file.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    print(f"✓ Saved calibration config to {calibration_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    
    print("\nNormalization Parameters (from baseline):")
    for metric, params in calibration["normalization"].items():
        print(f"  {metric}:")
        print(f"    mean: {params['mean']:.4f}")
        print(f"    std:  {params['std']:.4f}")
        print(f"    range: [{params['min']:.4f}, {params['max']:.4f}]")
    
    print("\nDerived Thresholds:")
    for metric, thresholds in calibration["thresholds"].items():
        print(f"  {metric}:")
        print(f"    low: {thresholds['low']:.4f}")
        print(f"    very_low: {thresholds['very_low']:.4f}")
        print(f"    separation: {thresholds['min_separation']:.4f}")
    
    print("\nSeparation Analysis:")
    for metric, separations in calibration["separation_analysis"].items():
        print(f"  {metric}:")
        for variant, sep_data in separations.items():
            print(f"    {variant}: separation={sep_data['separation']:.4f}, overlap={sep_data['overlap']:.2%}")
    
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review calibration_config.json")
    print("2. Update coherence engine to use normalized scores and calibrated thresholds")
    print("3. Re-run evaluation with calibrated metrics")


if __name__ == "__main__":
    main()
