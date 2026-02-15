"""
Build embedding indexes for image and audio retrieval.

Scans ALL available data directories (not just processed/) and builds
a single unified index per modality with domain tags.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.aligned_embeddings import AlignedEmbedder

IMG_DIRS = [
    Path("data/processed/images"),
    Path("data/wikimedia/images"),
]
AUD_DIRS = [
    Path("data/processed/audio"),
    Path("data/freesound/audio"),
]
OUT_DIR = Path("data/embeddings")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# Exclude video formats that might be in wikimedia
EXCLUDE_EXTS = {".webm", ".svg", ".gif"}
AUD_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# Domain tags derived from filename patterns
DOMAIN_RULES = {
    "nature": ["forest", "tree", "mountain", "jungle", "garden", "park", "field",
               "meadow", "countryside", "rural", "fog", "dawn", "sunrise"],
    "urban": ["city", "street", "neon", "urban", "downtown", "skyscraper",
              "building", "traffic", "people", "cobblestone"],
    "water": ["beach", "ocean", "wave", "sea", "shore", "coast", "lake",
              "river", "rain", "water"],
}


def infer_domain(filepath: str, caption: str = "") -> str:
    """Infer domain from filename and optional caption."""
    text = (Path(filepath).stem + " " + caption).lower()
    for domain, keywords in DOMAIN_RULES.items():
        if any(kw in text for kw in keywords):
            return domain
    return "other"


def collect_images() -> list[tuple[Path, str]]:
    """Collect all images with domain tags. Returns (path, domain) pairs."""
    seen = set()
    results = []
    for d in IMG_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            if f.suffix.lower() in EXCLUDE_EXTS:
                continue
            # Deduplicate by filename (processed/ and wikimedia/ share some files)
            if f.name in seen:
                continue
            seen.add(f.name)
            domain = infer_domain(str(f))
            results.append((f, domain))
    return results


def collect_audio() -> list[tuple[Path, str]]:
    """Collect all audio with domain tags. Returns (path, domain) pairs."""
    seen = set()
    results = []

    # Load freesound captions for better domain tagging
    captions = {}
    for samples_file in [Path("data/freesound/samples.json")]:
        if samples_file.exists():
            with samples_file.open() as f:
                for entry in json.load(f):
                    audio_path = entry.get("audio", "")
                    caption = entry.get("caption", "")
                    captions[Path(audio_path).name] = caption

    for d in AUD_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in AUD_EXTS:
                continue
            if f.name in seen:
                continue
            seen.add(f.name)
            caption = captions.get(f.name, "")
            domain = infer_domain(str(f), caption)
            results.append((f, domain))
    return results


def build_image_index() -> None:
    """Build unified image index from all sources."""
    embedder = AlignedEmbedder(target_dim=512, enable_cache=False)
    images = collect_images()

    if not images:
        print("No images found in any source directory.")
        return

    ids = []
    embs = []
    domains = []
    skipped = 0

    for path, domain in images:
        try:
            emb = embedder.embed_image(str(path))
            ids.append(str(path))
            embs.append(emb)
            domains.append(domain)
        except Exception as e:
            print(f"  Skipped {path.name}: {e}")
            skipped += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "image_index.npz",
        ids=np.array(ids),
        embs=np.stack(embs),
        domains=np.array(domains),
    )

    print(f"Image index built: {len(ids)} images ({skipped} skipped)")
    for domain in sorted(set(domains)):
        count = domains.count(domain)
        print(f"  {domain}: {count}")


def build_audio_index() -> None:
    """Build unified audio index from all sources."""
    embedder = AlignedEmbedder(target_dim=512, enable_cache=False)
    audios = collect_audio()

    if not audios:
        print("No audio found in any source directory.")
        return

    ids = []
    embs = []
    domains = []
    skipped = 0

    for path, domain in audios:
        try:
            emb = embedder.embed_audio(str(path))
            ids.append(str(path))
            embs.append(emb)
            domains.append(domain)
        except Exception as e:
            print(f"  Skipped {path.name}: {e}")
            skipped += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "audio_index.npz",
        ids=np.array(ids),
        embs=np.stack(embs),
        domains=np.array(domains),
    )

    print(f"Audio index built: {len(ids)} audio files ({skipped} skipped)")
    for domain in sorted(set(domains)):
        count = domains.count(domain)
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    build_image_index()
    build_audio_index()
