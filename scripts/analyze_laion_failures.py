"""
Phase 2: Automatic failure diagnostics for LAION evaluation results.

Analyzes raw_results.json and produces:
- diagnostics.json with failure mode taxonomy
- top-10 worst examples saved as bundles
- Failure mode counters and statistics
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Failure mode thresholds (will be calibrated in Phase 4)
THRESHOLDS = {
    "low_similarity": 0.3,  # Below this is considered low
    "very_low_similarity": 0.1,
    "high_variance": 0.15,  # Std dev threshold for high variance
    "missing_audio_elements": True,  # Flag if audio_elements empty in plan
    "missing_visual_elements": True,  # Flag if visual_attributes empty
}


def analyze_plan_completeness(plan: Dict[str, Any]) -> Dict[str, bool]:
    """Check if semantic plan has missing elements."""
    # Check actual plan structure (UnifiedPlan schema)
    audio_constraints = plan.get("audio_constraints", {})
    audio_intent = audio_constraints.get("audio_intent", []) if isinstance(audio_constraints, dict) else []
    audio_sources = audio_constraints.get("sound_sources", []) if isinstance(audio_constraints, dict) else []
    audio_elements = audio_intent + audio_sources
    
    style_controls = plan.get("style_controls", {})
    visual_style = style_controls.get("visual_style", []) if isinstance(style_controls, dict) else []
    image_constraints = plan.get("image_constraints", {})
    image_objects = image_constraints.get("objects", []) if isinstance(image_constraints, dict) else []
    visual_elements = visual_style + image_objects
    
    core_semantics = plan.get("core_semantics", {})
    main_subjects = core_semantics.get("main_subjects", []) if isinstance(core_semantics, dict) else []
    
    return {
        "planner_missing_audio_elements": len(audio_elements) == 0,
        "planner_missing_visual_elements": len(visual_elements) == 0,
        "planner_missing_primary_entities": len(main_subjects) == 0,
    }


def analyze_similarity_scores(scores: Dict[str, float]) -> Dict[str, bool]:
    """Analyze similarity scores for failure patterns."""
    st_i = scores.get("st_i")
    st_a = scores.get("st_a")
    si_a = scores.get("si_a")
    msci = scores.get("msci")

    flags = {}

    if st_i is not None:
        flags["text_image_similarity_low"] = st_i < THRESHOLDS["low_similarity"]
        flags["text_image_similarity_very_low"] = st_i < THRESHOLDS["very_low_similarity"]

    if st_a is not None:
        flags["text_audio_similarity_low"] = st_a < THRESHOLDS["low_similarity"]
        flags["text_audio_similarity_very_low"] = st_a < THRESHOLDS["very_low_similarity"]

    if si_a is not None:
        flags["image_audio_similarity_low"] = si_a < THRESHOLDS["low_similarity"]
        flags["image_audio_similarity_very_low"] = si_a < THRESHOLDS["very_low_similarity"]

    if msci is not None:
        flags["msci_low"] = msci < THRESHOLDS["low_similarity"]
        flags["msci_very_low"] = msci < THRESHOLDS["very_low_similarity"]
        flags["msci_negative"] = msci < 0

    return flags


def analyze_semantic_drift(drift: Dict[str, float]) -> Dict[str, bool]:
    """Analyze semantic drift from plan to outputs."""
    # High drift threshold (will be calibrated)
    DRIFT_THRESHOLD = 0.5

    flags = {}
    for modality, drift_value in drift.items():
        flags[f"{modality}_drift_high"] = drift_value > DRIFT_THRESHOLD

    return flags


def classify_failure_mode(
    plan_flags: Dict[str, bool],
    similarity_flags: Dict[str, bool],
    drift_flags: Dict[str, bool],
    classification: Dict[str, Any],
) -> str:
    """Classify the primary failure mode."""
    label = classification.get("label", "UNKNOWN")

    # Check for specific failure patterns
    if similarity_flags.get("text_audio_similarity_very_low") or similarity_flags.get("image_audio_similarity_very_low"):
        return "AUDIO_ALIGNMENT_FAILURE"
    
    if plan_flags.get("planner_missing_audio_elements"):
        return "PLANNER_AUDIO_UNDERSPECIFIED"
    
    if plan_flags.get("planner_missing_visual_elements"):
        return "PLANNER_VISUAL_UNDERSPECIFIED"
    
    if similarity_flags.get("text_image_similarity_very_low"):
        return "TEXT_IMAGE_MISMATCH"
    
    if drift_flags.get("audio_drift_high"):
        return "AUDIO_GENERATION_DRIFT"
    
    if drift_flags.get("image_drift_high"):
        return "IMAGE_GENERATION_DRIFT"
    
    if similarity_flags.get("msci_negative"):
        return "GLOBAL_COHERENCE_FAILURE"
    
    # Fall back to classification label
    if label in ["GLOBAL_FAILURE", "LOCAL_MODALITY_WEAKNESS"]:
        return label
    
    return "NO_CLEAR_FAILURE"


def analyze_results(raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze all results and produce diagnostics."""
    diagnostics = {
        "failure_mode_counts": {},
        "flag_counts": {},
        "worst_examples": [],
        "statistics": {},
    }

    all_flags = {
        "planner": {},
        "similarity": {},
        "drift": {},
    }

    for result in raw_results:
        # Extract plan from bundle if available
        plan = {}
        if "bundle_path" in result:
            bundle_path = Path(result["bundle_path"])
            if bundle_path.exists():
                try:
                    with bundle_path.open("r", encoding="utf-8") as f:
                        bundle = json.load(f)
                    plan = bundle.get("meta", {}).get("semantic_plan", {})
                except Exception:
                    pass

        # Analyze plan completeness
        plan_flags = analyze_plan_completeness(plan)
        for flag, value in plan_flags.items():
            all_flags["planner"][flag] = all_flags["planner"].get(flag, 0) + (1 if value else 0)

        # Analyze similarity scores
        similarity_flags = analyze_similarity_scores(result.get("scores", {}))
        for flag, value in similarity_flags.items():
            all_flags["similarity"][flag] = all_flags["similarity"].get(flag, 0) + (1 if value else 0)

        # Analyze semantic drift
        drift_flags = analyze_semantic_drift(result.get("semantic_drift", {}))
        for flag, value in drift_flags.items():
            all_flags["drift"][flag] = all_flags["drift"].get(flag, 0) + (1 if value else 0)

        # Classify failure mode
        failure_mode = classify_failure_mode(
            plan_flags,
            similarity_flags,
            drift_flags,
            result.get("classification", {}),
        )
        diagnostics["failure_mode_counts"][failure_mode] = (
            diagnostics["failure_mode_counts"].get(failure_mode, 0) + 1
        )

        # Collect worst examples (lowest MSCI)
        msci = result.get("scores", {}).get("msci")
        if msci is not None:
            diagnostics["worst_examples"].append({
                "sample_id": result.get("sample_id"),
                "caption": result.get("caption"),
                "msci": msci,
                "scores": result.get("scores", {}),
                "failure_mode": failure_mode,
                "flags": {
                    **plan_flags,
                    **similarity_flags,
                    **drift_flags,
                },
                "bundle_path": result.get("bundle_path"),
            })

    # Sort worst examples by MSCI
    diagnostics["worst_examples"].sort(key=lambda x: x["msci"])
    diagnostics["worst_examples"] = diagnostics["worst_examples"][:10]

    # Aggregate flag counts
    diagnostics["flag_counts"] = {
        "planner": all_flags["planner"],
        "similarity": all_flags["similarity"],
        "drift": all_flags["drift"],
    }

    # Compute statistics
    msci_values = [
        r.get("scores", {}).get("msci")
        for r in raw_results
        if r.get("scores", {}).get("msci") is not None
    ]
    if msci_values:
        diagnostics["statistics"] = {
            "total_runs": len(raw_results),
            "runs_with_msci": len(msci_values),
            "msci_mean": sum(msci_values) / len(msci_values),
            "msci_min": min(msci_values),
            "msci_max": max(msci_values),
            "negative_msci_count": sum(1 for m in msci_values if m < 0),
        }

    return diagnostics


def save_worst_examples(diagnostics: Dict[str, Any], output_dir: Path):
    """Copy worst examples to a dedicated directory."""
    worst_dir = output_dir / "worst_examples"
    worst_dir.mkdir(parents=True, exist_ok=True)

    for idx, example in enumerate(diagnostics["worst_examples"], start=1):
        bundle_path = Path(example["bundle_path"])
        if bundle_path.exists():
            # Copy entire run directory
            run_dir = bundle_path.parent
            dest_dir = worst_dir / f"rank_{idx:02d}_{example['sample_id']}"
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(run_dir, dest_dir)

            # Save metadata
            metadata = {
                "rank": idx,
                "sample_id": example["sample_id"],
                "caption": example["caption"],
                "msci": example["msci"],
                "scores": example["scores"],
                "failure_mode": example["failure_mode"],
                "flags": example["flags"],
            }
            with (dest_dir / "failure_analysis.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    results_dir = Path("runs/laion_eval")
    raw_results_file = results_dir / "raw_results.json"

    if not raw_results_file.exists():
        print(f"Error: {raw_results_file} not found. Run run_laion_eval.py first.")
        return

    print("=" * 60)
    print("Failure Diagnostics Analysis")
    print("=" * 60)

    # Load raw results
    with raw_results_file.open("r", encoding="utf-8") as f:
        raw_results = json.load(f)

    print(f"Loaded {len(raw_results)} results")
    print()

    # Analyze
    diagnostics = analyze_results(raw_results)

    # Save diagnostics
    diagnostics_file = results_dir / "diagnostics.json"
    with diagnostics_file.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved diagnostics to {diagnostics_file}")

    # Save worst examples
    save_worst_examples(diagnostics, results_dir)
    print(f"✓ Saved top-10 worst examples to {results_dir / 'worst_examples'}")

    # Print summary
    print("\n" + "=" * 60)
    print("FAILURE MODE SUMMARY")
    print("=" * 60)
    print("\nFailure Mode Counts:")
    for mode, count in sorted(
        diagnostics["failure_mode_counts"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {mode}: {count}")

    print("\nTop Planner Issues:")
    for flag, count in sorted(
        diagnostics["flag_counts"]["planner"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]:
        print(f"  {flag}: {count}")

    print("\nTop Similarity Issues:")
    for flag, count in sorted(
        diagnostics["flag_counts"]["similarity"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]:
        print(f"  {flag}: {count}")

    print("\nWorst 5 Examples (by MSCI):")
    for example in diagnostics["worst_examples"][:5]:
        print(f"  MSCI={example['msci']:.4f}: {example['caption'][:60]}...")
        print(f"    Failure mode: {example['failure_mode']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
