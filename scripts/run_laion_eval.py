"""
Dataset-driven evaluation loop for LAION captions.

Phase 1: Operationalize dataset evaluation
- Load LAION captions (hard cap 500, seeded shuffle)
- For each caption: semantic plan → generate → evaluate → store
- Output: runs/laion_eval/raw_results.json + summary_msci.json
"""

import json
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

from src.data.laion_loader import LaionPromptLoader
from src.pipeline.generate_and_evaluate import generate_and_evaluate


def deterministic_shuffle(items: List[Any], seed: int = 42) -> List[Any]:
    """Shuffle deterministically using seed."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    return shuffled


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute MSCI summary statistics from results."""
    msci_values = [r["scores"]["msci"] for r in results if r["scores"].get("msci") is not None]
    st_i_values = [r["scores"]["st_i"] for r in results if r["scores"].get("st_i") is not None]
    st_a_values = [r["scores"]["st_a"] for r in results if r["scores"].get("st_a") is not None]
    si_a_values = [r["scores"]["si_a"] for r in results if r["scores"].get("si_a") is not None]

    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "count": len(values),
            "mean": round(mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
        }

    return {
        "msci": stats(msci_values),
        "st_i": stats(st_i_values),
        "st_a": stats(st_a_values),
        "si_a": stats(si_a_values),
        "total_runs": len(results),
        "successful_runs": len([r for r in results if r["scores"].get("msci") is not None]),
    }


def main():
    # Configuration
    MAX_SAMPLES = 500
    SEED = 42
    OUT_DIR = Path("runs/laion_eval")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LAION Dataset Evaluation")
    print("=" * 60)
    print(f"Max samples: {MAX_SAMPLES}")
    print(f"Seed: {SEED}")
    print(f"Output dir: {OUT_DIR}")
    print()

    # Load LAION captions
    loader = LaionPromptLoader(max_samples=None)  # Load all, we'll cap after shuffle
    all_samples = loader.load()
    print(f"Loaded {len(all_samples)} samples from LAION")

    # Deterministic shuffle and cap
    samples = deterministic_shuffle(all_samples, seed=SEED)[:MAX_SAMPLES]
    print(f"Selected {len(samples)} samples (deterministic shuffle, seed={SEED})")
    print()

    # Run evaluation loop
    all_results: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, start=1):
        caption = sample["caption"]
        sample_id = sample["id"]

        print(f"\n[{idx}/{len(samples)}] Processing: {sample_id}")
        print(f"  Caption: {caption[:80]}..." if len(caption) > 80 else f"  Caption: {caption}")

        try:
            # Generate and evaluate
            bundle = generate_and_evaluate(
                prompt=caption,
                out_dir=str(OUT_DIR),
                use_ollama=True,
                deterministic=True,
                seed=SEED + idx,  # Vary seed per run but deterministically
            )

            # Extract results
            record = {
                "sample_id": sample_id,
                "caption": caption,
                "run_id": bundle.run_id,
                "scores": bundle.scores,
                "classification": bundle.coherence.get("classification", {}),
                "semantic_drift": bundle.semantic_drift,
                "bundle_path": str(Path(bundle.meta["out_dir"]) / "bundle.json"),
            }

            all_results.append(record)

            # Print quick summary
            msci = bundle.scores.get("msci", "N/A")
            classification = bundle.coherence.get("classification", {}).get("label", "UNKNOWN")
            print(f"  ✓ MSCI: {msci}, Classification: {classification}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed_runs.append({
                "sample_id": sample_id,
                "caption": caption,
                "error": str(e),
            })

    # Save raw results
    raw_results_file = OUT_DIR / "raw_results.json"
    with raw_results_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved raw results to {raw_results_file}")

    # Save failed runs if any
    if failed_runs:
        failed_file = OUT_DIR / "failed_runs.json"
        with failed_file.open("w", encoding="utf-8") as f:
            json.dump(failed_runs, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(failed_runs)} failed runs to {failed_file}")

    # Compute and save summary statistics
    summary = compute_summary_stats(all_results)
    summary_file = OUT_DIR / "summary_msci.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total runs: {summary['total_runs']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Failed runs: {len(failed_runs)}")
    print()
    print("MSCI Statistics:")
    msci_stats = summary["msci"]
    print(f"  Mean: {msci_stats['mean']:.4f}")
    print(f"  Min:  {msci_stats['min']:.4f}")
    print(f"  Max:  {msci_stats['max']:.4f}")
    print(f"  Std:  {msci_stats['std']:.4f}")
    print()
    print("Similarity Statistics:")
    print(f"  st_i (text-image): mean={summary['st_i']['mean']:.4f}")
    print(f"  st_a (text-audio): mean={summary['st_a']['mean']:.4f}")
    print(f"  si_a (image-audio): mean={summary['si_a']['mean']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
