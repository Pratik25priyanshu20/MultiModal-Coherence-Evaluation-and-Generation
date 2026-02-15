#!/usr/bin/env python3
"""
Regenerate RQ3 audio with AudioLDM — unique audio per sample.

Replaces the repeated retrieval audio with prompt-specific generated audio.
- Baseline / wrong_image samples: audio generated from the prompt text (matched)
- Wrong_audio samples: audio generated from a mismatched domain prompt (preserves perturbation)

Recomputes MSCI scores after regeneration.

Usage:
    python scripts/regenerate_rq3_audio.py
    python scripts/regenerate_rq3_audio.py --force-audioldm    # Try AudioLDM (needs ~2GB download)
    python scripts/regenerate_rq3_audio.py --device cuda        # GPU acceleration
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLES_PATH = PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json"
AUDIO_DIR = PROJECT_ROOT / "runs" / "rq3" / "audio"

# Mismatched prompts for wrong_audio condition — deliberately from wrong domains
MISMATCH_PROMPTS = {
    "nature": [
        "busy city intersection with car horns and police sirens",
        "crowded indoor shopping mall with chatter and music",
        "construction site with jackhammers and heavy machinery",
    ],
    "urban": [
        "quiet forest stream with birds singing at dawn",
        "ocean waves gently lapping on a sandy beach",
        "rain falling on leaves in a tropical jungle",
    ],
    "water": [
        "busy highway traffic with trucks and motorcycles",
        "children playing in a school playground",
        "city cafe with espresso machine and conversations",
    ],
    "mixed": [
        "industrial factory floor with metal grinding",
        "quiet library with pages turning and clock ticking",
        "farm animals in a barnyard at sunrise",
    ],
}


def get_mismatch_prompt(domain: str, seed: int) -> str:
    """Get a deliberately mismatched audio prompt for a domain."""
    rng = random.Random(seed)
    options = MISMATCH_PROMPTS.get(domain, MISMATCH_PROMPTS["mixed"])
    return rng.choice(options)


def main():
    parser = argparse.ArgumentParser(description="Regenerate RQ3 audio with AudioLDM")
    parser.add_argument("--force-audioldm", action="store_true",
                        help="Try to load AudioLDM model (downloads ~2GB on first run)")
    parser.add_argument("--device", default="cpu", help="Device for generation (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    args = parser.parse_args()

    # Load existing samples
    with SAMPLES_PATH.open() as f:
        data = json.load(f)
    samples = data["samples"]

    print("=" * 70)
    print("RQ3 Audio Regeneration")
    print("=" * 70)
    print(f"  Samples: {len(samples)}")
    print(f"  Backend: {'AudioLDM' if args.force_audioldm else 'Deterministic Ambient'}")
    print(f"  Device:  {args.device}")
    print(f"  Output:  {AUDIO_DIR}")
    print("=" * 70)

    # Count current unique audio files
    current_audio = set(s["audio_path"] for s in samples)
    print(f"\n  Current unique audio files: {len(current_audio)}")
    print(f"  Target unique audio files:  {len(samples)}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize audio generator
    from src.generators.audio.generator import AudioGenerator
    generator = AudioGenerator(device=args.device, force_audioldm=args.force_audioldm)

    backend_used = "audioldm" if generator._audioldm_pipe is not None else "fallback_ambient"
    print(f"  Audio backend loaded: {backend_used}")
    if generator._audioldm_error:
        print(f"  AudioLDM note: {generator._audioldm_error}")

    print("\nGenerating audio...")
    t_start = time.time()

    for i, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        condition = sample["condition"]
        prompt = sample["prompt_text"]
        domain = sample.get("domain", "mixed")
        seed = args.seed + i  # Unique seed per sample

        # Determine audio prompt
        if condition == "wrong_audio":
            # Generate audio from a MISMATCHED prompt (preserves the perturbation)
            audio_prompt = get_mismatch_prompt(domain, seed)
            note = f"wrong_audio: generated from mismatched prompt: '{audio_prompt}'"
        else:
            # Baseline and wrong_image: generate audio matching the text prompt
            audio_prompt = prompt
            note = f"matched audio for: '{prompt[:50]}...'"

        out_path = AUDIO_DIR / f"{sample_id}_{condition}.wav"

        t0 = time.time()
        result = generator.generate(
            prompt=audio_prompt,
            out_path=str(out_path),
            seed=seed,
        )
        elapsed = time.time() - t0

        # Update sample with new audio path
        sample["audio_path"] = str(out_path)
        sample["audio_backend"] = result.backend
        sample["audio_prompt"] = audio_prompt

        print(f"  [{i+1}/{len(samples)}] {sample_id} ({condition}) -> {result.backend} ({elapsed:.1f}s)")

    gen_time = time.time() - t_start
    print(f"\nAudio generation complete: {gen_time:.0f}s")

    # Verify all unique
    new_audio = set(s["audio_path"] for s in samples)
    print(f"  Unique audio files: {len(new_audio)} / {len(samples)}")

    # Recompute MSCI scores
    print("\nRecomputing MSCI scores...")
    from src.coherence.coherence_engine import evaluate_coherence

    for i, sample in enumerate(samples):
        t0 = time.time()
        try:
            eval_out = evaluate_coherence(
                text=sample["prompt_text"],
                image_path=sample["image_path"],
                audio_path=sample["audio_path"],
            )
            scores = eval_out.get("scores", {})
            old_msci = sample["msci"]
            sample["msci"] = scores.get("msci")
            sample["st_i"] = scores.get("st_i")
            sample["st_a"] = scores.get("st_a")
            sample["msci_old"] = old_msci  # Keep old value for comparison

            elapsed = time.time() - t0
            delta = sample["msci"] - old_msci if sample["msci"] is not None else 0
            print(
                f"  [{i+1}/{len(samples)}] {sample['sample_id']} "
                f"MSCI: {old_msci:.4f} -> {sample['msci']:.4f} "
                f"({delta:+.4f}) ({elapsed:.1f}s)"
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] {sample['sample_id']} ERROR: {e} ({elapsed:.1f}s)")

    # Update MSCI range
    msci_vals = [s["msci"] for s in samples if s.get("msci") is not None]
    data["msci_range"] = [min(msci_vals), max(msci_vals)] if msci_vals else [0, 0]
    data["audio_regenerated"] = True
    data["audio_backend"] = backend_used
    data["samples"] = samples

    # Save updated samples
    # Back up old file
    backup_path = SAMPLES_PATH.with_suffix(".json.bak")
    if SAMPLES_PATH.exists():
        backup_path.write_text(SAMPLES_PATH.read_text())
        print(f"\n  Old samples backed up to: {backup_path}")

    with SAMPLES_PATH.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"  Updated samples saved to: {SAMPLES_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Audio files generated: {len(samples)}")
    print(f"  All unique: {len(new_audio) == len(samples)}")
    print(f"  Backend: {backend_used}")
    if msci_vals:
        import numpy as np
        print(f"  MSCI range: [{min(msci_vals):.4f}, {max(msci_vals):.4f}]")
        print(f"  MSCI mean:  {np.mean(msci_vals):.4f}")

    # Per-condition summary
    from collections import Counter
    for cond in ["baseline", "wrong_image", "wrong_audio"]:
        cond_scores = [s["msci"] for s in samples if s["condition"] == cond and s.get("msci") is not None]
        if cond_scores:
            import numpy as np
            print(f"  {cond:<14} N={len(cond_scores)}  mean={np.mean(cond_scores):.4f}")

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    print("\nIMPORTANT: Delete old evaluation sessions before restarting human eval:")
    print(f"  rm -rf {PROJECT_ROOT / 'runs/rq3/sessions'}")
    print("  streamlit run app/human_eval_app.py")


if __name__ == "__main__":
    main()
