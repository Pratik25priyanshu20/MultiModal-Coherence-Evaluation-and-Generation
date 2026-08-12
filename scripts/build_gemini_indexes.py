#!/usr/bin/env python3
"""
Build Gemini Embedding Indexes for Image and Audio Retrieval.

Re-embeds all images and audio files using Gemini Embedding 2 to create
unified-space indexes for the cMSCI v2 negative bank.

Reuses file discovery logic from build_embedding_indexes.py.

Usage:
    python scripts/build_gemini_indexes.py
    python scripts/build_gemini_indexes.py --images-only
    python scripts/build_gemini_indexes.py --audio-only

API cost estimate: ~161 API calls, ~$0.78
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_embedding_indexes import (
    collect_audio,
    collect_images,
)
from src.config.settings import (
    GEMINI_AUDIO_INDEX_PATH,
    GEMINI_IMAGE_INDEX_PATH,
)

OUT_DIR = GEMINI_IMAGE_INDEX_PATH.parent


def build_gemini_image_index() -> None:
    """Build image index using Gemini embeddings."""
    from src.embeddings.gemini_embedder import GeminiEmbedder

    embedder = GeminiEmbedder(enable_cache=True)
    images = collect_images()

    if not images:
        print("No images found.")
        return

    print(f"Embedding {len(images)} images via Gemini API...")
    ids = []
    embs = []
    domains = []
    skipped = 0
    t0 = time.time()

    for idx, (path, domain) in enumerate(images):
        try:
            emb = embedder.embed_image(str(path))
            ids.append(str(path))
            embs.append(emb)
            domains.append(domain)
            print(f"  [{idx+1}/{len(images)}] {path.name} ({domain})", end="\r")
        except Exception as e:
            print(f"  Skipped {path.name}: {e}")
            skipped += 1

    elapsed = time.time() - t0
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(GEMINI_IMAGE_INDEX_PATH),
        ids=np.array(ids),
        embs=np.stack(embs),
        domains=np.array(domains),
    )

    print(f"Gemini image index built: {len(ids)} images ({skipped} skipped) in {elapsed:.1f}s")
    for domain in sorted(set(domains)):
        count = domains.count(domain)
        print(f"  {domain}: {count}")


def build_gemini_audio_index() -> None:
    """Build audio index using Gemini embeddings."""
    from src.embeddings.gemini_embedder import GeminiEmbedder

    embedder = GeminiEmbedder(enable_cache=True)
    audios = collect_audio()

    if not audios:
        print("No audio found.")
        return

    print(f"Embedding {len(audios)} audio files via Gemini API...")
    ids = []
    embs = []
    domains = []
    skipped = 0
    t0 = time.time()

    for idx, (path, domain) in enumerate(audios):
        try:
            emb = embedder.embed_audio(str(path))
            ids.append(str(path))
            embs.append(emb)
            domains.append(domain)
            print(f"  [{idx+1}/{len(audios)}] {path.name} ({domain})", end="\r")
        except Exception as e:
            print(f"  Skipped {path.name}: {e}")
            skipped += 1

    elapsed = time.time() - t0
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(GEMINI_AUDIO_INDEX_PATH),
        ids=np.array(ids),
        embs=np.stack(embs),
        domains=np.array(domains),
    )

    print(f"Gemini audio index built: {len(ids)} audio files ({skipped} skipped) in {elapsed:.1f}s")
    for domain in sorted(set(domains)):
        count = domains.count(domain)
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Gemini embedding indexes")
    parser.add_argument("--images-only", action="store_true", help="Only build image index")
    parser.add_argument("--audio-only", action="store_true", help="Only build audio index")
    args = parser.parse_args()

    print("=" * 60)
    print("Building Gemini Embedding Indexes")
    print("=" * 60)

    if not args.audio_only:
        print("\n--- Image Index ---")
        build_gemini_image_index()

    if not args.images_only:
        print("\n--- Audio Index ---")
        build_gemini_audio_index()

    print("\nDone.")
