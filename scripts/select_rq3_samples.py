#!/usr/bin/env python3
"""
RQ3 Sample Selection — Stratified sampling for human evaluation.

Selects 30 bundles from RQ1 (baseline + perturbations) and RQ2 (direct mode)
spanning the full MSCI range. All evaluators rate the same 30 samples.

Usage:
    python scripts/select_rq3_samples.py
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_candidate_bundles():
    """Load candidate bundles from RQ1 and RQ2 results."""
    candidates = []

    # --- RQ1: baseline + perturbations (seed=42 only, to avoid duplicates) ---
    rq1_path = PROJECT_ROOT / "runs" / "rq1" / "rq1_results.json"
    if rq1_path.exists():
        with open(rq1_path) as f:
            rq1 = json.load(f)
        for r in rq1["results"]:
            if r["seed"] != 42:
                continue  # One seed per prompt to avoid near-duplicate bundles
            candidates.append({
                "source": "rq1",
                "prompt_id": r["prompt_id"],
                "prompt_text": r["prompt_text"],
                "domain": r["domain"],
                "condition": r["condition"],
                "mode": "direct",
                "seed": r["seed"],
                "msci": r["msci"],
                "st_i": r["st_i"],
                "st_a": r["st_a"],
                "image_path": str(PROJECT_ROOT / r["image_path"]),
                "audio_path": str(PROJECT_ROOT / r["audio_path"]),
            })

    # --- RQ2: direct mode only (seed=42), for variety ---
    rq2_path = PROJECT_ROOT / "runs" / "rq2" / "rq2_results.json"
    if rq2_path.exists():
        with open(rq2_path) as f:
            rq2 = json.load(f)
        for r in rq2["results"]:
            if r["seed"] != 42 or r["mode"] != "direct":
                continue
            # Skip if same prompt_id + baseline already in RQ1
            dup = any(
                c["prompt_id"] == r["prompt_id"] and c["condition"] == "baseline"
                and c["source"] == "rq1"
                for c in candidates
            )
            if not dup:
                candidates.append({
                    "source": "rq2",
                    "prompt_id": r["prompt_id"],
                    "prompt_text": r["prompt_text"],
                    "domain": r.get("domain", "unknown"),
                    "condition": "baseline",
                    "mode": "direct",
                    "seed": r["seed"],
                    "msci": r["msci"],
                    "st_i": r["st_i"],
                    "st_a": r["st_a"],
                    "image_path": str(PROJECT_ROOT / r["image_path"]),
                    "audio_path": str(PROJECT_ROOT / r["audio_path"]),
                })

    return candidates


def stratified_select(candidates, n_target=30, seed=42):
    """
    Select n_target samples stratified by MSCI score range and condition.

    Strategy:
    - 10 from baseline (matched) — spanning low/mid/high MSCI
    - 10 from wrong_image (image-mismatched)
    - 10 from wrong_audio (audio-mismatched)

    This ensures evaluators see the full range of alignment quality
    and all three perturbation conditions for RQ3 validity.
    """
    random.seed(seed)

    baselines = [c for c in candidates if c["condition"] == "baseline"]
    wrong_img = [c for c in candidates if c["condition"] == "wrong_image"]
    wrong_aud = [c for c in candidates if c["condition"] == "wrong_audio"]

    selected = []

    for group, n in [(baselines, 10), (wrong_img, 10), (wrong_aud, 10)]:
        if len(group) < n:
            print(f"Warning: only {len(group)} candidates in group, need {n}")
            n = len(group)

        # Sort by MSCI and pick evenly across the range
        group_sorted = sorted(group, key=lambda x: x["msci"])
        step = max(1, len(group_sorted) / n)
        indices = [int(i * step) for i in range(n)]
        # Clamp to valid range
        indices = [min(i, len(group_sorted) - 1) for i in indices]
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        # Fill if we lost any to dedup
        remaining = [i for i in range(len(group_sorted)) if i not in seen]
        random.shuffle(remaining)
        while len(unique_indices) < n and remaining:
            unique_indices.append(remaining.pop(0))

        for idx in unique_indices[:n]:
            selected.append(group_sorted[idx])

    # Assign sample IDs (blind, no condition info)
    random.shuffle(selected)
    for i, s in enumerate(selected):
        s["sample_id"] = f"S{i+1:03d}"

    return selected


def verify_files(samples):
    """Verify all referenced image/audio files exist."""
    missing = []
    for s in samples:
        if not Path(s["image_path"]).exists():
            missing.append(f"IMAGE: {s['image_path']}")
        if not Path(s["audio_path"]).exists():
            missing.append(f"AUDIO: {s['audio_path']}")
    return missing


def main():
    print("=" * 60)
    print("RQ3 Sample Selection")
    print("=" * 60)

    candidates = load_candidate_bundles()
    print(f"\nLoaded {len(candidates)} candidate bundles")

    # Count by condition
    from collections import Counter
    cond_counts = Counter(c["condition"] for c in candidates)
    for cond, count in sorted(cond_counts.items()):
        print(f"  {cond}: {count}")

    # Select 30 stratified samples
    samples = stratified_select(candidates, n_target=30)
    print(f"\nSelected {len(samples)} samples:")

    # Summary
    cond_counts = Counter(s["condition"] for s in samples)
    for cond, count in sorted(cond_counts.items()):
        print(f"  {cond}: {count}")

    msci_vals = [s["msci"] for s in samples]
    print(f"\nMSCI range: [{min(msci_vals):.4f}, {max(msci_vals):.4f}]")
    print(f"MSCI mean:  {sum(msci_vals)/len(msci_vals):.4f}")

    domain_counts = Counter(s["domain"] for s in samples)
    print(f"\nDomains: {dict(domain_counts)}")

    # Verify files exist
    missing = verify_files(samples)
    if missing:
        print(f"\nWARNING: {len(missing)} missing files:")
        for m in missing[:5]:
            print(f"  {m}")
    else:
        print("\nAll files verified.")

    # Save
    out_dir = PROJECT_ROOT / "runs" / "rq3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rq3_samples.json"

    output = {
        "experiment": "RQ3: Human Alignment Validation",
        "description": "30 stratified bundles for multi-rater human evaluation",
        "n_samples": len(samples),
        "conditions": dict(cond_counts),
        "msci_range": [min(msci_vals), max(msci_vals)],
        "samples": samples,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
    print("\nNext: run  python scripts/run_human_eval.py --evaluator <name>")


if __name__ == "__main__":
    main()
