#!/usr/bin/env python3
"""
Generate matching images for 70 new AudioCaps-based evaluation samples.

Problem: Text-image coherence is only 20-30% for new samples because images
were retrieved via CLIP from a pool of just 57 generic photos. AudioCaps
captions describe sounds ("cat meows", "water pouring") that don't match
generic landscape/city images.

Fix: Generate images with Stable Diffusion 1.5 from the caption text.
Only generates for baseline (24) and wrong_audio (23) samples where
the image SHOULD match the text. Skips wrong_image samples (mismatched
image is intentional).

Usage:
    python scripts/generate_eval_images.py --device cuda --seed 42
    python scripts/generate_eval_images.py --dry-run          # preview only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import RQ3_SAMPLES_EXTENDED_PATH, MSCI_WEIGHTS

logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "generated" / "eval_images"


def load_samples(path: Path) -> dict:
    """Load extended samples JSON."""
    with open(path) as f:
        return json.load(f)


def select_samples(data: dict) -> list[dict]:
    """Select new samples (S031+) where condition != wrong_image."""
    targets = []
    for sample in data["samples"]:
        sid = int(sample["sample_id"][1:])
        if sid >= 31 and sample["condition"] != "wrong_image":
            targets.append(sample)
    return targets


def generate_images(
    targets: list[dict],
    device: str = "cpu",
    seed: int = 42,
) -> list[dict]:
    """Generate images with SD 1.5 and recompute embeddings."""
    from src.generators.image.generator_hybrid import HybridImageGenerator
    from src.embeddings.aligned_embeddings import AlignedEmbedder
    from src.embeddings.similarity import cosine_similarity

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Init SD 1.5 pipeline
    logger.info("Loading Stable Diffusion 1.5 on %s...", device)
    generator = HybridImageGenerator(
        force_sd=True,
        sd_model="sd1.5",
        device=device,
    )

    if generator._sd_pipe is None:
        logger.error("Failed to load SD pipeline. Aborting.")
        return targets

    # Init embedder for recomputing st_i
    logger.info("Loading CLIP embedder...")
    embedder = AlignedEmbedder(enable_cache=False)

    w_ti = MSCI_WEIGHTS["st_i"]
    w_ta = MSCI_WEIGHTS["st_a"]

    for i, sample in enumerate(targets):
        sid = sample["sample_id"]
        out_path = OUTPUT_DIR / f"{sid}_{sample['condition']}.png"

        logger.info(
            "[%d/%d] %s: generating image for '%s'",
            i + 1, len(targets), sid, sample["prompt_text"][:60],
        )

        # Generate image
        result = generator.generate(
            prompt=sample["prompt_text"],
            out_path=str(out_path),
            seed=seed,
        )

        old_st_i = sample.get("st_i", 0)
        old_image = sample.get("image_path", "")

        # Recompute text-image similarity
        text_emb = embedder.embed_text(sample["prompt_text"])
        image_emb = embedder.embed_image(str(out_path))
        new_st_i = cosine_similarity(text_emb, image_emb)

        # Update sample
        sample["image_path"] = str(out_path)
        sample["st_i"] = round(new_st_i, 4)
        sample["msci"] = round(w_ti * new_st_i + w_ta * sample["st_a"], 4)

        logger.info(
            "  %s: st_i %.3f -> %.3f | image: %s -> %s",
            sid, old_st_i, new_st_i,
            Path(old_image).name if old_image else "none",
            out_path.name,
        )

    # Free GPU memory
    generator.unload()
    logger.info("SD pipeline unloaded, GPU memory freed.")

    return targets


def update_json(data: dict, targets: list[dict], path: Path) -> None:
    """Write updated targets back into the full data and save."""
    target_map = {s["sample_id"]: s for s in targets}
    for i, sample in enumerate(data["samples"]):
        if sample["sample_id"] in target_map:
            data["samples"][i] = target_map[sample["sample_id"]]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved updated samples to %s", path)


def print_summary(targets: list[dict], data: dict) -> None:
    """Print before/after statistics."""
    # Compute mean st_i for all new baseline samples
    baseline = [s for s in targets if s["condition"] == "baseline"]
    wrong_audio = [s for s in targets if s["condition"] == "wrong_audio"]

    if baseline:
        mean_sti = sum(s["st_i"] for s in baseline) / len(baseline)
        print(f"\n  Baseline samples ({len(baseline)}): mean st_i = {mean_sti:.3f}")
    if wrong_audio:
        mean_sti = sum(s["st_i"] for s in wrong_audio) / len(wrong_audio)
        print(f"  Wrong-audio samples ({len(wrong_audio)}): mean st_i = {mean_sti:.3f}")

    # Compare with original 30 baseline
    orig_baseline = [
        s for s in data["samples"]
        if int(s["sample_id"][1:]) < 31 and s["condition"] == "baseline"
    ]
    if orig_baseline:
        mean_orig = sum(s["st_i"] for s in orig_baseline) / len(orig_baseline)
        print(f"  Original baseline ({len(orig_baseline)}): mean st_i = {mean_orig:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate matching images for new evaluation samples"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cuda", "mps", "cpu"],
        help="Device for SD inference (default: cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview selected samples without generating",
    )
    parser.add_argument(
        "--samples-path", type=str, default=None,
        help="Override path to rq3_samples_extended.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    samples_path = Path(args.samples_path) if args.samples_path else RQ3_SAMPLES_EXTENDED_PATH

    if not samples_path.exists():
        logger.error("Samples file not found: %s", samples_path)
        sys.exit(1)

    data = load_samples(samples_path)
    targets = select_samples(data)

    print(f"Selected {len(targets)} samples for image generation:")
    print(f"  - baseline: {sum(1 for s in targets if s['condition'] == 'baseline')}")
    print(f"  - wrong_audio: {sum(1 for s in targets if s['condition'] == 'wrong_audio')}")
    print(f"  - skipped wrong_image: {sum(1 for s in data['samples'] if int(s['sample_id'][1:]) >= 31 and s['condition'] == 'wrong_image')}")
    print(f"  Output dir: {OUTPUT_DIR}")

    if args.dry_run:
        print("\n[DRY RUN] Samples that would be generated:")
        for s in targets:
            print(f"  {s['sample_id']} ({s['condition']}): {s['prompt_text'][:70]}")
        return

    targets = generate_images(targets, device=args.device, seed=args.seed)
    update_json(data, targets, samples_path)
    print_summary(targets, data)

    # Verify all images exist
    missing = [
        s["sample_id"] for s in targets
        if not Path(s["image_path"]).exists()
    ]
    if missing:
        logger.warning("Missing images for: %s", missing)
    else:
        print(f"\nAll {len(targets)} images generated successfully in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
