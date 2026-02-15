from __future__ import annotations

from typing import Optional

import numpy as np

from src.embeddings.audio_embedder import AudioEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.embeddings.projection import ProjectionHead
from src.embeddings.text_embedder import TextEmbedder
from src.utils.cache import EmbeddingCache


class AlignedEmbedder:
    """
    Cross-modal embedding with correct space alignment.

    Two pre-trained shared spaces are used:
    - CLIP: text ↔ image (both from openai/clip-vit-base-patch32, 512-d)
    - CLAP: text ↔ audio (both from laion/clap-htsat-unfused, 512-d)

    For text-image similarity: use embed_text() and embed_image()
    For text-audio similarity: use embed_text_for_audio() and embed_audio()
    For image-audio similarity: these are cross-space (CLIP vs CLAP) —
        no meaningful direct comparison without a trained bridge.

    ProjectionHead is identity when in_dim == out_dim (preserving pre-trained
    alignment). Only applies a linear transformation when dimensions differ.
    """

    def __init__(
        self,
        target_dim: int = 512,
        enable_cache: bool = True,
        cache_dir: str = ".cache/embeddings",
    ):
        self.text = TextEmbedder()       # CLIP text encoder
        self.image = ImageEmbedder()     # CLIP image encoder
        self.audio = AudioEmbedder()     # CLAP audio encoder (also has text)

        # Identity projections when dims match (512 → 512)
        self.text_proj = ProjectionHead(512, target_dim)
        self.image_proj = ProjectionHead(512, target_dim)
        self.audio_proj = ProjectionHead(512, target_dim)

        self.cache: Optional[EmbeddingCache] = None
        if enable_cache:
            self.cache = EmbeddingCache(cache_dir=cache_dir)

    def embed_text(self, text: str) -> np.ndarray:
        """CLIP text embedding — use for text-image comparison."""
        if self.cache:
            cached = self.cache.get(text, "text")
            if cached is not None:
                return cached

        emb = self.text.embed(text)
        projected = self.text_proj.project(emb)

        if self.cache:
            self.cache.set(text, "text", projected)

        return projected

    def embed_text_for_audio(self, text: str) -> np.ndarray:
        """CLAP text embedding — use for text-audio comparison."""
        if self.cache:
            cached = self.cache.get(text, "text_clap")
            if cached is not None:
                return cached

        emb = self.audio.embed_text(text)
        projected = self.audio_proj.project(emb)

        if self.cache:
            self.cache.set(text, "text_clap", projected)

        return projected

    def embed_image(self, path: str) -> np.ndarray:
        """CLIP image embedding — use for text-image comparison."""
        if self.cache:
            cached = self.cache.get(path, "image")
            if cached is not None:
                return cached

        emb = self.image.embed(path)
        projected = self.image_proj.project(emb)

        if self.cache:
            self.cache.set(path, "image", projected)

        return projected

    def embed_audio(self, path: str) -> np.ndarray:
        """CLAP audio embedding — use for text-audio comparison."""
        if self.cache:
            cached = self.cache.get(path, "audio")
            if cached is not None:
                return cached

        emb = self.audio.embed(path)
        projected = self.audio_proj.project(emb)

        if self.cache:
            self.cache.set(path, "audio", projected)

        return projected

    @staticmethod
    def shared(
        target_dim: int = 512,
        enable_cache: bool = True,
        cache_dir: str = ".cache/embeddings",
    ) -> "AlignedEmbedder":
        """
        Get the thread-safe shared singleton instance.

        Use this in parallel experiments to avoid loading CLIP+CLAP per thread.
        """
        from src.embeddings.shared_embedder import get_shared_embedder
        return get_shared_embedder(target_dim, enable_cache, cache_dir)
