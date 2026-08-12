#!/usr/bin/env python3
"""
Prepare training data for cross-space bridge and probabilistic adapters.

Downloads and preprocesses paired image-audio data from multiple sources:
  1. OmniBench (1.1K triples): HuggingFace dataset with image + audio + text
  2. VGGSound (210K): Pre-extracted CLIP+CLAP features from ClipClap-GZSL
  3. Domain-matched (~1050 pairs): From existing embedding indexes
  4. AudioCaps triples: Caption->CLIP text + audio->CLAP from manifest
  5. VGGSound triples: Label->CLIP text + audio->CLAP from manifest

Supports embedding augmentation (Gaussian noise, dropout, mixup) and
dataset statistics reporting.

Outputs a combined npz file ready for bridge training.

Usage:
    python scripts/prepare_bridge_data.py --help
    python scripts/prepare_bridge_data.py --omnibench-only
    python scripts/prepare_bridge_data.py --vggsound-features-path /path/to/vggsound_features/
    python scripts/prepare_bridge_data.py --audiocaps-triples --vggsound-triples --report-stats
    python scripts/prepare_bridge_data.py --all --augment --augment-factor 3 --report-stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "bridge_training"
DOMAIN_MATCHED_NPZ = OUTPUT_DIR / "domain_matched.npz"
OMNIBENCH_NPZ = OUTPUT_DIR / "omnibench.npz"
VGGSOUND_NPZ = OUTPUT_DIR / "vggsound.npz"
COMBINED_NPZ = OUTPUT_DIR / "combined_training.npz"


def prepare_domain_matched(
    image_index_path: str = "data/embeddings/image_index.npz",
    audio_index_path: str = "data/embeddings/audio_index.npz",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build domain-matched pairs from existing embedding indexes.

    Cross-products within each domain (nature, urban, water).
    Expected ~1,050 pairs.
    """
    logger.info("Building domain-matched pairs from embedding indexes...")
    img_path = str(PROJECT_ROOT / image_index_path)
    aud_path = str(PROJECT_ROOT / audio_index_path)

    if not Path(img_path).exists() or not Path(aud_path).exists():
        logger.warning("Embedding indexes not found. Run: python scripts/build_embedding_indexes.py")
        return np.array([]), np.array([]), 0

    img_index = np.load(img_path, allow_pickle=True)
    aud_index = np.load(aud_path, allow_pickle=True)

    # Our indexes use keys: ids, embs, domains
    img_embeddings = img_index["embs"]
    img_domains = list(img_index["domains"])
    aud_embeddings = aud_index["embs"]
    aud_domains = list(aud_index["domains"])

    # Build cross-product pairs within each domain
    image_embs_list = []
    audio_embs_list = []
    pair_counts = {}

    for domain in ["nature", "urban", "water"]:
        img_idx = [i for i, d in enumerate(img_domains) if d == domain]
        aud_idx = [i for i, d in enumerate(aud_domains) if d == domain]

        n_pairs = len(img_idx) * len(aud_idx)
        pair_counts[domain] = n_pairs

        for ii in img_idx:
            for ai in aud_idx:
                image_embs_list.append(img_embeddings[ii])
                audio_embs_list.append(aud_embeddings[ai])

    total = sum(pair_counts.values())
    logger.info("Domain-matched pairs: %s = %d total",
                ", ".join(f"{d}={n}" for d, n in pair_counts.items()), total)

    if total == 0:
        logger.error("No domain-matched pairs found!")
        return np.array([]), np.array([]), 0

    image_embs = np.array(image_embs_list, dtype=np.float32)
    audio_embs = np.array(audio_embs_list, dtype=np.float32)

    logger.info("Domain-matched: %d pairs, image shape %s, audio shape %s",
                len(image_embs), image_embs.shape, audio_embs.shape)

    np.savez_compressed(
        DOMAIN_MATCHED_NPZ,
        image_embeddings=image_embs,
        audio_embeddings=audio_embs,
        source="domain_matched",
    )
    logger.info("Saved: %s", DOMAIN_MATCHED_NPZ)

    return image_embs, audio_embs, len(image_embs)


def prepare_omnibench() -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Download OmniBench dataset and extract CLIP/CLAP embeddings.

    OmniBench contains ~1.1K image-audio-text triples.
    We embed with CLIP (image) and CLAP (audio) to get paired embeddings.

    Uses raw bytes from the dataset to avoid torchcodec/ffmpeg dependency.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install datasets: pip install datasets")
        return np.array([]), np.array([]), 0

    logger.info("Loading OmniBench dataset from HuggingFace...")
    try:
        ds = load_dataset("m-a-p/OmniBench", split="train")
    except Exception as e:
        logger.error("Failed to load OmniBench: %s", e)
        return np.array([]), np.array([]), 0

    logger.info("OmniBench loaded: %d samples", len(ds))

    from src.embeddings.aligned_embeddings import AlignedEmbedder
    import tempfile
    import io
    from PIL import Image as PILImage

    embedder = AlignedEmbedder(target_dim=512)

    # Access raw arrow table to get bytes without audio decoding
    table = ds.data
    # Combine all chunks into a single contiguous array
    audio_col = table.column("audio").combine_chunks()
    image_col = table.column("image").combine_chunks()

    image_embs = []
    audio_embs = []
    errors = 0

    for i in range(len(ds)):
        try:
            # Get raw bytes from arrow column (avoids torchcodec)
            img_data = image_col[i].as_py()
            aud_data = audio_col[i].as_py()

            img_bytes = img_data.get("bytes") if img_data else None
            aud_bytes = aud_data.get("bytes") if aud_data else None
            aud_path = aud_data.get("path", "") if aud_data else ""

            if img_bytes is None or aud_bytes is None:
                errors += 1
                continue

            # Embed image from bytes via temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_img:
                tmp_img.write(img_bytes)
                tmp_img.flush()
                img_emb = embedder.embed_image(tmp_img.name)

            # Embed audio from bytes via temp file
            # Determine suffix from path
            suffix = ".wav"
            if aud_path.endswith(".mp3"):
                suffix = ".mp3"
            elif aud_path.endswith(".flac"):
                suffix = ".flac"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp_aud:
                tmp_aud.write(aud_bytes)
                tmp_aud.flush()
                aud_emb = embedder.embed_audio(tmp_aud.name)

            image_embs.append(img_emb)
            audio_embs.append(aud_emb)

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("Error on sample %d: %s", i, e)

        if (i + 1) % 100 == 0:
            logger.info("  [%d/%d] embedded, %d errors so far", i + 1, len(ds), errors)

    if not image_embs:
        logger.error("No OmniBench samples embedded successfully")
        return np.array([]), np.array([]), 0

    image_arr = np.array(image_embs, dtype=np.float32)
    audio_arr = np.array(audio_embs, dtype=np.float32)

    logger.info("OmniBench: %d pairs embedded (%d errors), shapes: %s, %s",
                len(image_arr), errors, image_arr.shape, audio_arr.shape)

    np.savez_compressed(
        OMNIBENCH_NPZ,
        image_embeddings=image_arr,
        audio_embeddings=audio_arr,
        source="omnibench",
    )
    logger.info("Saved: %s", OMNIBENCH_NPZ)

    return image_arr, audio_arr, len(image_arr)


def prepare_vggsound(features_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load pre-extracted VGGSound CLIP+CLAP features.

    Expects pre-extracted features from ClipClap-GZSL or similar:
      features_path/
        clip_image_features.npy   (N, 512) — CLIP image embeddings
        clap_audio_features.npy   (N, 512) — CLAP audio embeddings

    Or alternatively:
      features_path/
        visual_features.npz       — with key 'features' (N, 512)
        audio_features.npz        — with key 'features' (N, 512)
    """
    feat_dir = Path(features_path)
    if not feat_dir.exists():
        logger.error("VGGSound features directory not found: %s", features_path)
        return np.array([]), np.array([]), 0

    logger.info("Loading VGGSound features from %s...", features_path)

    image_embs = None
    audio_embs = None

    # Try .npy format first
    clip_npy = feat_dir / "clip_image_features.npy"
    clap_npy = feat_dir / "clap_audio_features.npy"
    if clip_npy.exists() and clap_npy.exists():
        image_embs = np.load(clip_npy)
        audio_embs = np.load(clap_npy)
    else:
        # Try .npz format
        vis_npz = feat_dir / "visual_features.npz"
        aud_npz = feat_dir / "audio_features.npz"
        if vis_npz.exists() and aud_npz.exists():
            image_embs = np.load(vis_npz)["features"]
            audio_embs = np.load(aud_npz)["features"]
        else:
            # Try generic naming
            for f in feat_dir.glob("*.npy"):
                if "clip" in f.stem.lower() or "image" in f.stem.lower() or "visual" in f.stem.lower():
                    image_embs = np.load(f)
                elif "clap" in f.stem.lower() or "audio" in f.stem.lower():
                    audio_embs = np.load(f)

    if image_embs is None or audio_embs is None:
        logger.error("Could not find CLIP image and CLAP audio features in %s", features_path)
        logger.error("Expected: clip_image_features.npy + clap_audio_features.npy")
        return np.array([]), np.array([]), 0

    # Validate dimensions
    if image_embs.shape[1] != 512:
        logger.error("CLIP features must be 512-d, got %d", image_embs.shape[1])
        return np.array([]), np.array([]), 0
    if audio_embs.shape[1] != 512:
        logger.error("CLAP features must be 512-d, got %d", audio_embs.shape[1])
        return np.array([]), np.array([]), 0

    # Match lengths
    n = min(len(image_embs), len(audio_embs))
    image_embs = image_embs[:n].astype(np.float32)
    audio_embs = audio_embs[:n].astype(np.float32)

    logger.info("VGGSound: %d pairs, shapes: %s, %s",
                n, image_embs.shape, audio_embs.shape)

    np.savez_compressed(
        VGGSOUND_NPZ,
        image_embeddings=image_embs,
        audio_embeddings=audio_embs,
        source="vggsound",
    )
    logger.info("Saved: %s", VGGSOUND_NPZ)

    return image_embs, audio_embs, n


def prepare_audiocaps_triples(
    manifest_path: Optional[str] = None,
    max_samples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prepare AudioCaps triples: embed captions via CLIP text encoder and audio via CLAP.

    Loads a manifest JSON with "entries" list, each containing "caption" and "audio_path".
    Returns (text_clip_embs, audio_embs, count).
    """
    if manifest_path is None:
        manifest_path = str(PROJECT_ROOT / "data" / "benchmarks" / "audiocaps" / "manifest.json")

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        logger.error("AudioCaps manifest not found: %s", manifest_path)
        return np.array([]), np.array([]), 0

    logger.info("Loading AudioCaps manifest from %s...", manifest_path)
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    if not entries:
        logger.error("AudioCaps manifest has no entries")
        return np.array([]), np.array([]), 0

    entries = entries[:max_samples]
    logger.info("AudioCaps: processing up to %d entries", len(entries))

    from src.embeddings.aligned_embeddings import AlignedEmbedder

    embedder = AlignedEmbedder(target_dim=512)

    text_clip_embs = []
    audio_embs = []
    errors = 0

    for i, entry in enumerate(entries):
        try:
            caption = entry.get("caption", "")
            audio_path = entry.get("audio_path", "")

            if not caption or not audio_path:
                errors += 1
                continue

            # Resolve audio path relative to manifest directory
            audio_file = Path(audio_path)
            if not audio_file.is_absolute():
                audio_file = manifest_file.parent / audio_path

            if not audio_file.exists():
                errors += 1
                continue

            # Embed caption via CLIP text encoder (image-space proxy)
            text_emb = embedder.embed_text(caption)
            # Embed audio via CLAP
            aud_emb = embedder.embed_audio(str(audio_file))

            text_clip_embs.append(text_emb)
            audio_embs.append(aud_emb)

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("AudioCaps error on entry %d: %s", i, e)

        if (i + 1) % 100 == 0:
            logger.info("  AudioCaps [%d/%d] embedded, %d errors so far",
                        i + 1, len(entries), errors)

    if not text_clip_embs:
        logger.error("No AudioCaps samples embedded successfully")
        return np.array([]), np.array([]), 0

    text_arr = np.array(text_clip_embs, dtype=np.float32)
    audio_arr = np.array(audio_embs, dtype=np.float32)

    logger.info("AudioCaps: %d pairs embedded (%d errors), shapes: %s, %s",
                len(text_arr), errors, text_arr.shape, audio_arr.shape)

    output_path = OUTPUT_DIR / "audiocaps_triples.npz"
    np.savez_compressed(
        output_path,
        text_clip_embeddings=text_arr,
        audio_embeddings=audio_arr,
        source="audiocaps",
    )
    logger.info("Saved: %s", output_path)

    return text_arr, audio_arr, len(text_arr)


def prepare_vggsound_triples(
    manifest_path: Optional[str] = None,
    max_samples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prepare VGGSound triples: embed labels/captions via CLIP text encoder and audio via CLAP.

    Loads a manifest JSON with "entries" list, each containing "label" or "caption"
    and "audio_path". Returns (text_clip_embs, audio_embs, count).
    """
    if manifest_path is None:
        manifest_path = str(PROJECT_ROOT / "data" / "benchmarks" / "vggsound" / "manifest.json")

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        logger.error("VGGSound manifest not found: %s", manifest_path)
        return np.array([]), np.array([]), 0

    logger.info("Loading VGGSound manifest from %s...", manifest_path)
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    if not entries:
        logger.error("VGGSound manifest has no entries")
        return np.array([]), np.array([]), 0

    entries = entries[:max_samples]
    logger.info("VGGSound triples: processing up to %d entries", len(entries))

    from src.embeddings.aligned_embeddings import AlignedEmbedder

    embedder = AlignedEmbedder(target_dim=512)

    text_clip_embs = []
    audio_embs = []
    errors = 0

    for i, entry in enumerate(entries):
        try:
            # Use "label" or "caption" field as text
            text = entry.get("label", "") or entry.get("caption", "")
            audio_path = entry.get("audio_path", "")

            if not text or not audio_path:
                errors += 1
                continue

            # Resolve audio path relative to manifest directory
            audio_file = Path(audio_path)
            if not audio_file.is_absolute():
                audio_file = manifest_file.parent / audio_path

            if not audio_file.exists():
                errors += 1
                continue

            # Embed text via CLIP text encoder (image-space proxy)
            text_emb = embedder.embed_text(text)
            # Embed audio via CLAP
            aud_emb = embedder.embed_audio(str(audio_file))

            text_clip_embs.append(text_emb)
            audio_embs.append(aud_emb)

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("VGGSound triples error on entry %d: %s", i, e)

        if (i + 1) % 100 == 0:
            logger.info("  VGGSound triples [%d/%d] embedded, %d errors so far",
                        i + 1, len(entries), errors)

    if not text_clip_embs:
        logger.error("No VGGSound triple samples embedded successfully")
        return np.array([]), np.array([]), 0

    text_arr = np.array(text_clip_embs, dtype=np.float32)
    audio_arr = np.array(audio_embs, dtype=np.float32)

    logger.info("VGGSound triples: %d pairs embedded (%d errors), shapes: %s, %s",
                len(text_arr), errors, text_arr.shape, audio_arr.shape)

    output_path = OUTPUT_DIR / "vggsound_triples.npz"
    np.savez_compressed(
        output_path,
        text_clip_embeddings=text_arr,
        audio_embeddings=audio_arr,
        source="vggsound_triples",
    )
    logger.info("Saved: %s", output_path)

    return text_arr, audio_arr, len(text_arr)


def augment_audio_embeddings(
    image_embs: np.ndarray,
    audio_embs: np.ndarray,
    n_augmented: int = 3,
    noise_std: float = 0.02,
    dropout_rate: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embedding-space augmentation for data diversity.

    For each pair, creates n_augmented variants:
      1. Gaussian noise: add N(0, noise_std) then L2-normalize
      2. Dropout: zero out random dimensions with probability dropout_rate, then L2-normalize
      3. Mixup (if n_augmented >= 3): blend with a random other pair using alpha ~ Beta(0.4, 0.4)

    Returns (augmented_images, augmented_audio) concatenated with originals.
    """
    if len(image_embs) == 0 or len(audio_embs) == 0:
        logger.warning("No embeddings to augment")
        return image_embs, audio_embs

    rng = np.random.default_rng(seed)
    n_pairs = len(image_embs)
    dim = image_embs.shape[1]

    logger.info("Augmenting %d pairs with %d variants each (noise_std=%.3f, dropout=%.2f)",
                n_pairs, n_augmented, noise_std, dropout_rate)

    aug_images = [image_embs.copy()]
    aug_audio = [audio_embs.copy()]

    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return x / norms

    # Variant 1: Gaussian noise
    noise_img = image_embs + rng.normal(0, noise_std, size=image_embs.shape).astype(np.float32)
    noise_aud = audio_embs + rng.normal(0, noise_std, size=audio_embs.shape).astype(np.float32)
    aug_images.append(_l2_normalize(noise_img))
    aug_audio.append(_l2_normalize(noise_aud))

    # Variant 2: Dropout
    if n_augmented >= 2:
        drop_mask_img = rng.random((n_pairs, dim)) > dropout_rate
        drop_mask_aud = rng.random((n_pairs, dim)) > dropout_rate
        dropped_img = image_embs * drop_mask_img.astype(np.float32)
        dropped_aud = audio_embs * drop_mask_aud.astype(np.float32)
        aug_images.append(_l2_normalize(dropped_img))
        aug_audio.append(_l2_normalize(dropped_aud))

    # Variant 3: Mixup
    if n_augmented >= 3:
        mix_indices = rng.integers(0, n_pairs, size=n_pairs)
        alphas = rng.beta(0.4, 0.4, size=(n_pairs, 1)).astype(np.float32)
        mixed_img = alphas * image_embs + (1 - alphas) * image_embs[mix_indices]
        mixed_aud = alphas * audio_embs + (1 - alphas) * audio_embs[mix_indices]
        aug_images.append(_l2_normalize(mixed_img))
        aug_audio.append(_l2_normalize(mixed_aud))

    # Additional noise variants if n_augmented > 3
    for v in range(3, n_augmented):
        scale = noise_std * (1 + 0.5 * (v - 2))
        extra_img = image_embs + rng.normal(0, scale, size=image_embs.shape).astype(np.float32)
        extra_aud = audio_embs + rng.normal(0, scale, size=audio_embs.shape).astype(np.float32)
        aug_images.append(_l2_normalize(extra_img))
        aug_audio.append(_l2_normalize(extra_aud))

    combined_images = np.concatenate(aug_images, axis=0)
    combined_audio = np.concatenate(aug_audio, axis=0)

    logger.info("Augmentation complete: %d original -> %d total pairs (%.1fx)",
                n_pairs, len(combined_images), len(combined_images) / n_pairs)

    return combined_images, combined_audio


def report_dataset_statistics(
    datasets: List[Tuple[str, np.ndarray, np.ndarray]],
) -> None:
    """
    Print formatted table of dataset statistics.

    Input: list of (name, image_embs, audio_embs) tuples.
    Shows per-source counts, shapes, mean/std cosine similarity, and totals.
    """
    if not datasets:
        logger.info("No datasets to report on.")
        return

    header = f"{'Source':<25} {'Count':>8} {'Img Shape':>16} {'Aud Shape':>16} {'Cos Sim Mean':>14} {'Cos Sim Std':>13}"
    separator = "-" * len(header)

    logger.info("")
    logger.info("DATASET STATISTICS")
    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    total_count = 0

    for name, img_embs, aud_embs in datasets:
        count = len(img_embs)
        if count == 0:
            logger.info(f"{name:<25} {'0':>8} {'N/A':>16} {'N/A':>16} {'N/A':>14} {'N/A':>13}")
            continue

        total_count += count
        img_shape = str(img_embs.shape)
        aud_shape = str(aud_embs.shape)

        # Compute cosine similarity between paired image and audio embeddings
        img_norm = img_embs / np.maximum(np.linalg.norm(img_embs, axis=1, keepdims=True), 1e-8)
        aud_norm = aud_embs / np.maximum(np.linalg.norm(aud_embs, axis=1, keepdims=True), 1e-8)
        cos_sims = np.sum(img_norm * aud_norm, axis=1)

        mean_sim = float(np.mean(cos_sims))
        std_sim = float(np.std(cos_sims))

        logger.info(f"{name:<25} {count:>8d} {img_shape:>16} {aud_shape:>16} {mean_sim:>14.4f} {std_sim:>13.4f}")

    logger.info(separator)
    logger.info(f"{'TOTAL':<25} {total_count:>8d}")
    logger.info(separator)
    logger.info("")


def combine_datasets(
    datasets: List[Tuple[np.ndarray, np.ndarray, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine multiple (image_embs, audio_embs) datasets into one."""
    all_images = []
    all_audio = []

    for img, aud, name in datasets:
        if len(img) > 0:
            all_images.append(img)
            all_audio.append(aud)
            logger.info("  %s: %d pairs", name, len(img))

    if not all_images:
        logger.error("No datasets to combine!")
        return np.array([]), np.array([])

    combined_images = np.concatenate(all_images, axis=0)
    combined_audio = np.concatenate(all_audio, axis=0)

    # Shuffle together
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(combined_images))
    combined_images = combined_images[perm]
    combined_audio = combined_audio[perm]

    logger.info("Combined: %d total pairs", len(combined_images))

    np.savez_compressed(
        COMBINED_NPZ,
        image_embeddings=combined_images,
        audio_embeddings=combined_audio,
        source="combined",
    )
    logger.info("Saved: %s", COMBINED_NPZ)

    return combined_images, combined_audio


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for cross-space bridge and prob adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Domain-matched only (CPU, fast)
  python scripts/prepare_bridge_data.py --domain-matched-only

  # OmniBench + domain-matched (CPU, slower — embeds ~1K samples)
  python scripts/prepare_bridge_data.py --omnibench-only

  # Full dataset with pre-downloaded VGGSound features
  python scripts/prepare_bridge_data.py --vggsound-features-path /data/vggsound_features/

  # AudioCaps + VGGSound triples with statistics
  python scripts/prepare_bridge_data.py --audiocaps-triples --vggsound-triples --report-stats

  # Everything with augmentation
  python scripts/prepare_bridge_data.py --all --augment --augment-factor 3 --report-stats

  # Everything
  python scripts/prepare_bridge_data.py --all --vggsound-features-path /data/vggsound_features/
        """,
    )
    parser.add_argument("--domain-matched-only", action="store_true",
                        help="Only build domain-matched pairs from embedding indexes")
    parser.add_argument("--omnibench-only", action="store_true",
                        help="Build domain-matched + OmniBench (CPU feasible)")
    parser.add_argument("--vggsound-features-path", type=str, default=None,
                        help="Path to pre-downloaded VGGSound CLIP+CLAP features")
    parser.add_argument("--all", action="store_true",
                        help="Build all available datasets and combine")
    parser.add_argument("--audiocaps-triples", action="store_true",
                        help="Prepare AudioCaps triples (caption->CLIP text + audio->CLAP)")
    parser.add_argument("--vggsound-triples", action="store_true",
                        help="Prepare VGGSound triples (label->CLIP text + audio->CLAP)")
    parser.add_argument("--augment", action="store_true",
                        help="Enable embedding augmentation (requires at least one dataset)")
    parser.add_argument("--augment-factor", type=int, default=3,
                        help="Number of augmented variants per pair (default: 3)")
    parser.add_argument("--report-stats", action="store_true",
                        help="Print dataset statistics")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    datasets = []

    # Always include domain-matched
    logger.info("=" * 60)
    logger.info("BRIDGE TRAINING DATA PREPARATION")
    logger.info("=" * 60)

    img_dm, aud_dm, n_dm = prepare_domain_matched()
    if n_dm > 0:
        datasets.append((img_dm, aud_dm, "domain_matched"))

    if args.domain_matched_only:
        elapsed = time.time() - t_start
        logger.info("Done in %.1fs. Domain-matched: %d pairs", elapsed, n_dm)
        return 0

    # OmniBench
    if args.omnibench_only or args.all:
        img_ob, aud_ob, n_ob = prepare_omnibench()
        if n_ob > 0:
            datasets.append((img_ob, aud_ob, "omnibench"))

    if args.omnibench_only and not args.all:
        if len(datasets) > 1:
            combine_datasets(datasets)
        elapsed = time.time() - t_start
        logger.info("Done in %.1fs", elapsed)
        return 0

    # VGGSound (pre-extracted features)
    if args.vggsound_features_path or args.all:
        if args.vggsound_features_path:
            img_vg, aud_vg, n_vg = prepare_vggsound(args.vggsound_features_path)
            if n_vg > 0:
                datasets.append((img_vg, aud_vg, "vggsound"))
        elif args.all:
            logger.info("VGGSound: skipped (no --vggsound-features-path provided)")

    # AudioCaps triples (caption -> CLIP text + audio -> CLAP)
    has_new_sources = False
    datasets_for_stats = None
    if args.audiocaps_triples or args.all:
        txt_ac, aud_ac, n_ac = prepare_audiocaps_triples()
        if n_ac > 0:
            datasets.append((txt_ac, aud_ac, "audiocaps_triples"))
            has_new_sources = True

    # VGGSound triples (label -> CLIP text + audio -> CLAP)
    if args.vggsound_triples or args.all:
        txt_vt, aud_vt, n_vt = prepare_vggsound_triples()
        if n_vt > 0:
            datasets.append((txt_vt, aud_vt, "vggsound_triples"))
            has_new_sources = True

    # Embedding augmentation
    if args.augment and datasets:
        logger.info("Running embedding augmentation (factor=%d)...", args.augment_factor)
        # Combine all pre-augmentation data for augmentation
        pre_aug_images = np.concatenate([d[0] for d in datasets if len(d[0]) > 0], axis=0)
        pre_aug_audio = np.concatenate([d[1] for d in datasets if len(d[1]) > 0], axis=0)
        aug_images, aug_audio = augment_audio_embeddings(
            pre_aug_images, pre_aug_audio,
            n_augmented=args.augment_factor,
        )
        # Replace datasets with single augmented dataset
        # Keep original source names for statistics but use augmented combined for output
        datasets_for_stats = [(name, img, aud) for img, aud, name in datasets]
        datasets = [(aug_images, aug_audio, "augmented_combined")]
        has_new_sources = True

    # Report dataset statistics
    if args.report_stats:
        if args.augment and datasets_for_stats is not None:
            # Show pre-augmentation stats plus augmented total
            stats_list = datasets_for_stats + [
                ("augmented_combined", datasets[0][0], datasets[0][1])
            ]
        else:
            stats_list = [(name, img, aud) for img, aud, name in datasets]
        report_dataset_statistics(stats_list)

    # Combine all
    if len(datasets) > 1:
        combine_datasets(datasets)
        # Save v3 if new sources were included
        if has_new_sources:
            combined_images = np.concatenate([d[0] for d in datasets if len(d[0]) > 0], axis=0)
            combined_audio = np.concatenate([d[1] for d in datasets if len(d[1]) > 0], axis=0)
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(combined_images))
            combined_images = combined_images[perm]
            combined_audio = combined_audio[perm]
            v3_path = OUTPUT_DIR / "combined_training_v3.npz"
            np.savez_compressed(
                v3_path,
                image_embeddings=combined_images,
                audio_embeddings=combined_audio,
                source="combined_v3",
            )
            logger.info("Saved v3 combined: %s (%d pairs)", v3_path, len(combined_images))
    elif len(datasets) == 1 and has_new_sources:
        # Single augmented dataset — save as v3
        v3_path = OUTPUT_DIR / "combined_training_v3.npz"
        np.savez_compressed(
            v3_path,
            image_embeddings=datasets[0][0],
            audio_embeddings=datasets[0][1],
            source="combined_v3",
        )
        logger.info("Saved v3 combined: %s (%d pairs)", v3_path, len(datasets[0][0]))

    elapsed = time.time() - t_start
    total = sum(len(d[0]) for d in datasets)
    logger.info("=" * 60)
    logger.info("COMPLETE: %d total pairs from %d sources in %.1fs",
                total, len(datasets), elapsed)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
