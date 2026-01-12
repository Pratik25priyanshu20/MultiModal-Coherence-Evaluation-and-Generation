from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings.aligned_embeddings import AlignedEmbedder

IMG_DIR = Path("data/processed/images")
AUD_DIR = Path("data/processed/audio")
OUT_DIR = Path("data/embeddings")


def build_image_index() -> None:
    embedder = AlignedEmbedder(target_dim=512)
    paths = sorted(
        path
        for path in IMG_DIR.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not paths:
        raise RuntimeError(
            f"No images found in {IMG_DIR}. "
            "Add images before building the index."
        )

    ids = []
    embs = []
    for path in paths:
        emb = embedder.embed_image(str(path))
        ids.append(str(path))
        embs.append(emb)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "image_index.npz",
        ids=np.array(ids),
        embs=np.stack(embs),
    )

    print(f"✅ Image index built with {len(ids)} items")


def build_audio_index() -> None:
    embedder = AlignedEmbedder(target_dim=512)
    paths = sorted(
        path
        for path in AUD_DIR.glob("*")
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
    )

    if not paths:
        raise RuntimeError(
            f"No audio files found in {AUD_DIR}. "
            "Add audio files before building the index."
        )

    ids = []
    embs = []
    for path in paths:
        emb = embedder.embed_audio(str(path))
        ids.append(str(path))
        embs.append(emb)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "audio_index.npz",
        ids=np.array(ids),
        embs=np.stack(embs),
    )

    print(f"✅ Audio index built with {len(ids)} items")


if __name__ == "__main__":
    if IMG_DIR.exists():
        build_image_index()
    else:
        print(f"❌ Missing directory: {IMG_DIR}")

    if AUD_DIR.exists():
        build_audio_index()
    else:
        print(f"❌ Missing directory: {AUD_DIR}")
