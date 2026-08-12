#!/usr/bin/env python3
"""
Generate New Evaluation Samples from AudioCaps for Human Rating.

Takes downloaded AudioCaps samples (text + audio), retrieves matching images
via CLIP from the image index, and creates evaluation bundles in the same
format as the existing RQ3 samples (S001-S030).

New samples get IDs S031, S032, ... and are ADDED to the existing data.
Nothing from the original 30 samples is modified.

Conditions created per sample:
  - baseline: correct text + correct image + correct audio
  - wrong_image: correct text + MISMATCHED image + correct audio
  - wrong_audio: correct text + correct image + MISMATCHED audio

Output:
  - runs/rq3/rq3_samples_extended.json  (original 30 + new samples)
  - runs/rq3/eval_bundles/              (HTML bundles for human raters)

Usage:
    python scripts/generate_new_eval_samples.py --n-new 50
    python scripts/generate_new_eval_samples.py --n-new 50 --domains nature,urban,water
    python scripts/generate_new_eval_samples.py --n-new 30 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from src.config.settings import (
    RQ3_SAMPLES_PATH, RQ3_SAMPLES_EXTENDED_PATH, RQ3_HUMAN_SCORES_PATH,
    IMAGE_INDEX_PATH,
)

SAMPLES_PATH = RQ3_SAMPLES_PATH
HUMAN_SCORES_PATH = RQ3_HUMAN_SCORES_PATH
MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmarks" / "audiocaps" / "manifest.json"
OUTPUT_SAMPLES_PATH = RQ3_SAMPLES_EXTENDED_PATH
BUNDLES_DIR = PROJECT_ROOT / "runs" / "rq3" / "eval_bundles"


# ---------------------------------------------------------------------------
# Domain classification for AudioCaps captions
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "nature": [
        "bird", "rain", "thunder", "wind", "forest", "river", "stream",
        "insect", "cricket", "frog", "animal", "dog", "cat", "rooster",
        "ocean", "wave", "waterfall", "storm", "leaves", "grass",
    ],
    "urban": [
        "car", "truck", "bus", "train", "engine", "horn", "siren",
        "traffic", "construction", "drill", "hammer", "machine",
        "airplane", "helicopter", "motorcycle", "vehicle", "door",
        "bell", "clock", "alarm", "phone", "keyboard",
    ],
    "water": [
        "water", "splash", "drip", "pour", "fountain", "pool",
        "shower", "faucet", "tap", "bubble", "boat", "ship",
        "fishing", "harbor", "sea", "lake",
    ],
}


def classify_domain(caption: str) -> str:
    """Classify a caption into a domain based on keywords."""
    caption_lower = caption.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in caption_lower)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "mixed"
    return best


# ---------------------------------------------------------------------------
# Image retrieval via CLIP
# ---------------------------------------------------------------------------

def load_image_index() -> Tuple[np.ndarray, List[str]]:
    """Load the image embedding index."""
    if not IMAGE_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Image index not found: {IMAGE_INDEX_PATH}\n"
            "Run: python scripts/build_embedding_indexes.py"
        )
    data = np.load(IMAGE_INDEX_PATH, allow_pickle=True)
    # Keys are 'ids' (paths), 'embs' (embeddings), 'domains'
    embeddings = data["embs"]
    paths = list(data["ids"])
    logger.info("Image index: %d images, %d-d embeddings", len(paths), embeddings.shape[1])
    return embeddings, paths


def retrieve_best_image(
    text: str,
    image_embeddings: np.ndarray,
    image_paths: List[str],
    embedder,
    exclude_paths: Optional[set] = None,
) -> Tuple[str, float]:
    """Retrieve the best matching image for a text query via CLIP."""
    text_emb = embedder.embed_text(text)
    # Cosine similarity
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)
    img_norms = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = img_norms @ text_norm

    # Sort by similarity descending
    order = np.argsort(-similarities)
    for idx in order:
        path = image_paths[idx]
        if exclude_paths and path in exclude_paths:
            continue
        return path, float(similarities[idx])

    # Fallback: return best even if excluded
    best_idx = order[0]
    return image_paths[best_idx], float(similarities[best_idx])


def retrieve_mismatched_image(
    text: str,
    matched_path: str,
    image_embeddings: np.ndarray,
    image_paths: List[str],
    embedder,
    domain: str,
) -> str:
    """Retrieve a mismatched image (low similarity to text, different domain)."""
    text_emb = embedder.embed_text(text)
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)
    img_norms = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = img_norms @ text_norm

    # Pick from the bottom quartile (low similarity), excluding matched
    order = np.argsort(similarities)  # ascending
    for idx in order:
        path = image_paths[idx]
        if path == matched_path:
            continue
        return path

    return image_paths[order[0]]


# ---------------------------------------------------------------------------
# Audio mismatch selection
# ---------------------------------------------------------------------------

def select_mismatched_audio(
    matched_audio: str,
    all_audio_paths: List[str],
    matched_domain: str,
    rng: np.random.Generator,
) -> str:
    """Select a mismatched audio file (different from matched, ideally different domain)."""
    # Try to find audio from a different domain
    candidates = [p for p in all_audio_paths if p != matched_audio]
    if not candidates:
        return matched_audio  # fallback

    # Prefer audio from different domain
    diff_domain = [p for p in candidates if matched_domain not in Path(p).stem.lower()]
    if diff_domain:
        return rng.choice(diff_domain)
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Sample selection from AudioCaps
# ---------------------------------------------------------------------------

def select_diverse_samples(
    entries: List[Dict],
    n_samples: int,
    seed: int = 2024,
) -> List[Dict]:
    """Select diverse AudioCaps samples across domains."""
    rng = np.random.default_rng(seed)

    # Classify domains
    for e in entries:
        e["domain"] = classify_domain(e["caption"])

    # Group by domain
    by_domain = {}
    for e in entries:
        by_domain.setdefault(e["domain"], []).append(e)

    logger.info("AudioCaps domain distribution:")
    for domain, items in sorted(by_domain.items()):
        logger.info("  %s: %d", domain, len(items))

    # Stratified sampling: equal from each domain
    domains = ["nature", "urban", "water", "mixed"]
    per_domain = max(1, n_samples // len(domains))
    selected = []

    for domain in domains:
        pool = by_domain.get(domain, [])
        if not pool:
            continue
        n_pick = min(per_domain, len(pool))
        picked = rng.choice(pool, n_pick, replace=False).tolist()
        selected.extend(picked)

    # Fill remaining from largest pools
    remaining = n_samples - len(selected)
    if remaining > 0:
        used_ids = {e["audiocap_id"] for e in selected}
        pool = [e for e in entries if e["audiocap_id"] not in used_ids]
        if pool:
            extra = rng.choice(pool, min(remaining, len(pool)), replace=False).tolist()
            selected.extend(extra)

    rng.shuffle(selected)
    return selected[:n_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate new evaluation samples")
    parser.add_argument("--n-new", type=int, default=50,
                        help="Number of new samples to generate (default: 50)")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what would be created, don't write files")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("GENERATE NEW EVALUATION SAMPLES")
    print("=" * 70)

    # 1. Load existing samples
    print("\n--- Loading existing data ---")
    if SAMPLES_PATH.exists():
        with open(SAMPLES_PATH) as f:
            existing_data = json.load(f)
        existing_samples = existing_data["samples"]
        existing_ids = {s["sample_id"] for s in existing_samples}
        print(f"  Existing samples: {len(existing_samples)} ({min(existing_ids)}-{max(existing_ids)})")
    else:
        existing_data = {"samples": []}
        existing_samples = []
        existing_ids = set()
        print("  No existing samples found")

    # Find next sample ID
    max_id = 0
    for sid in existing_ids:
        num = int(sid.replace("S", ""))
        max_id = max(max_id, num)
    next_id = max_id + 1
    print(f"  Next sample ID: S{next_id:03d}")

    # 2. Load AudioCaps manifest
    print("\n--- Loading AudioCaps manifest ---")
    if not MANIFEST_PATH.exists():
        print(f"  ERROR: {MANIFEST_PATH} not found!")
        print("  Run: python scripts/download_benchmarks.py --audiocaps --huggingface --max-samples 1000")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    entries = manifest.get("entries", [])

    # Filter to entries whose audio actually exists
    valid_entries = [e for e in entries if Path(e["audio_path"]).exists()]
    print(f"  AudioCaps entries: {len(valid_entries)} (with audio on disk)")

    if len(valid_entries) < args.n_new:
        print(f"  WARNING: Only {len(valid_entries)} entries available, reducing n_new")
        args.n_new = len(valid_entries)

    # 3. Select diverse samples
    print(f"\n--- Selecting {args.n_new} diverse samples ---")
    selected = select_diverse_samples(valid_entries, args.n_new, args.seed)
    print(f"  Selected {len(selected)} samples")

    # Domain distribution
    domain_counts = {}
    for s in selected:
        domain_counts[s["domain"]] = domain_counts.get(s["domain"], 0) + 1
    for d, c in sorted(domain_counts.items()):
        print(f"    {d}: {c}")

    if args.dry_run:
        print("\n--- DRY RUN: Would create these samples ---")
        for i, entry in enumerate(selected[:10]):
            sid = f"S{next_id + i:03d}"
            print(f"  {sid}: [{entry['domain']}] {entry['caption'][:60]}...")
        if len(selected) > 10:
            print(f"  ... and {len(selected) - 10} more")
        print("\nRe-run without --dry-run to generate files.")
        return

    # 4. Load image index and embedder
    print("\n--- Loading image index + embedder ---")
    t0 = time.time()
    image_embeddings, image_paths = load_image_index()

    from src.embeddings.aligned_embeddings import AlignedEmbedder
    embedder = AlignedEmbedder(target_dim=512)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Collect all audio paths for mismatch selection
    all_audio_paths = [e["audio_path"] for e in valid_entries]

    # 5. Generate samples with conditions
    print(f"\n--- Generating {args.n_new} samples (3 conditions each) ---")
    new_samples = []
    conditions = ["baseline", "wrong_image", "wrong_audio"]

    # We'll create n_new/3 of each condition (balanced)
    n_per_condition = args.n_new // 3
    n_remainder = args.n_new - 3 * n_per_condition

    # Assign conditions
    condition_assignments = (
        ["baseline"] * n_per_condition +
        ["wrong_image"] * n_per_condition +
        ["wrong_audio"] * n_per_condition +
        ["baseline"] * n_remainder  # extra go to baseline
    )
    rng.shuffle(condition_assignments)

    used_images = set()

    for i, (entry, condition) in enumerate(zip(selected, condition_assignments)):
        sid = f"S{next_id + i:03d}"
        caption = entry["caption"]
        audio_path = entry["audio_path"]
        domain = entry["domain"]

        # Retrieve matching image via CLIP
        matched_image, sim = retrieve_best_image(
            caption, image_embeddings, image_paths, embedder, exclude_paths=used_images
        )
        used_images.add(matched_image)

        # Build sample based on condition
        if condition == "baseline":
            image_path = matched_image
            audio_final = audio_path
        elif condition == "wrong_image":
            image_path = retrieve_mismatched_image(
                caption, matched_image, image_embeddings, image_paths, embedder, domain
            )
            audio_final = audio_path
        elif condition == "wrong_audio":
            image_path = matched_image
            audio_final = select_mismatched_audio(audio_path, all_audio_paths, domain, rng)

        # Compute MSCI components
        try:
            text_emb_clip = embedder.embed_text(caption)
            img_emb = embedder.embed_image(image_path)
            text_emb_clap = embedder.embed_text_for_audio(caption)
            aud_emb = embedder.embed_audio(audio_final)

            from src.embeddings.similarity import cosine_similarity
            st_i = float(cosine_similarity(text_emb_clip, img_emb))
            st_a = float(cosine_similarity(text_emb_clap, aud_emb))
            msci = 0.45 * st_i + 0.45 * st_a
        except Exception as e:
            logger.warning("Error computing MSCI for %s: %s", sid, e)
            st_i = 0.0
            st_a = 0.0
            msci = 0.0

        sample = {
            "source": "audiocaps",
            "prompt_id": f"ac_{entry['audiocap_id']}",
            "prompt_text": caption,
            "domain": domain,
            "condition": condition,
            "mode": "direct",
            "seed": 42,
            "msci": round(msci, 4),
            "st_i": round(st_i, 4),
            "st_a": round(st_a, 4),
            "image_path": str(image_path),
            "audio_path": str(audio_final),
            "sample_id": sid,
        }
        new_samples.append(sample)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_new}] {sid} [{condition}] [{domain}] "
                  f"MSCI={msci:.3f} st_i={st_i:.3f} st_a={st_a:.3f}")

    # 6. Summary statistics
    print(f"\n--- Summary ---")
    new_conditions = {}
    new_domains = {}
    msci_vals = []
    for s in new_samples:
        new_conditions[s["condition"]] = new_conditions.get(s["condition"], 0) + 1
        new_domains[s["domain"]] = new_domains.get(s["domain"], 0) + 1
        msci_vals.append(s["msci"])

    print(f"  New samples: {len(new_samples)}")
    print(f"  Conditions: {dict(sorted(new_conditions.items()))}")
    print(f"  Domains: {dict(sorted(new_domains.items()))}")
    print(f"  MSCI range: [{min(msci_vals):.3f}, {max(msci_vals):.3f}]")
    print(f"  MSCI mean: {np.mean(msci_vals):.3f}")

    # 7. Save extended samples
    print(f"\n--- Saving ---")
    all_samples = existing_samples + new_samples
    extended_data = {
        "experiment": "RQ3: Human Alignment Validation (Extended)",
        "description": f"Original {len(existing_samples)} + {len(new_samples)} new AudioCaps samples",
        "n_samples": len(all_samples),
        "n_original": len(existing_samples),
        "n_new": len(new_samples),
        "conditions": {},
        "samples": all_samples,
    }

    # Count conditions
    for s in all_samples:
        c = s["condition"]
        extended_data["conditions"][c] = extended_data["conditions"].get(c, 0) + 1

    OUTPUT_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SAMPLES_PATH, "w") as f:
        json.dump(extended_data, f, indent=2)
    print(f"  Saved: {OUTPUT_SAMPLES_PATH}")
    print(f"  Total samples: {len(all_samples)} ({len(existing_samples)} original + {len(new_samples)} new)")

    # 8. Create evaluation bundles for human raters
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

    # Save a simple CSV for raters
    csv_path = BUNDLES_DIR / "new_samples_for_rating.csv"
    with open(csv_path, "w") as f:
        f.write("sample_id,prompt_text,image_path,audio_path,domain,condition\n")
        for s in new_samples:
            caption_escaped = s["prompt_text"].replace('"', '""')
            f.write(f'{s["sample_id"]},"{caption_escaped}",{s["image_path"]},{s["audio_path"]},'
                    f'{s["domain"]},{s["condition"]}\n')
    print(f"  Rating CSV: {csv_path}")

    # Save a JSON version for programmatic access
    rating_json = BUNDLES_DIR / "new_samples_for_rating.json"
    with open(rating_json, "w") as f:
        json.dump({
            "instructions": (
                "Rate each sample on a 1-5 scale for:\n"
                "  text_image_coherence: How well does the image match the text?\n"
                "  text_audio_coherence: How well does the audio match the text?\n"
                "  image_audio_coherence: How well do the image and audio go together?\n"
                "  overall_coherence: Overall, how coherent is this multimodal bundle?\n"
                "\n"
                "Scale: 1=Completely Unrelated, 2=Vague Connection, 3=Partial Match, "
                "4=Mostly Aligned, 5=Strong Alignment"
            ),
            "samples": [
                {
                    "sample_id": s["sample_id"],
                    "prompt_text": s["prompt_text"],
                    "image_path": s["image_path"],
                    "audio_path": s["audio_path"],
                    "domain": s["domain"],
                }
                for s in new_samples
            ],
        }, f, indent=2)
    print(f"  Rating JSON: {rating_json}")

    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print(f"  1. Have 3+ raters evaluate the {len(new_samples)} new samples")
    print(f"     Use: {rating_json}")
    print(f"  2. Save ratings as session files in runs/rq3/sessions/")
    print(f"  3. Re-run: python scripts/optimize_cmsci.py")
    print(f"     (will automatically use all {len(all_samples)} samples)")


if __name__ == "__main__":
    main()
