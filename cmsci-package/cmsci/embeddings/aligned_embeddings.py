from __future__ import annotations

from typing import Optional

import numpy as np

from cmsci.embeddings.audio_embedder import AudioEmbedder
from cmsci.embeddings.image_embedder import ImageEmbedder
from cmsci.embeddings.projection import ProjectionHead
from cmsci.embeddings.text_embedder import TextEmbedder
from cmsci.utils.cache import EmbeddingCache
from cmsci.config.settings import (
    AUDIO_USE_WINDOWED,
    AUDIO_WINDOW_SEC,
    AUDIO_HOP_SEC,
    AUDIO_AGGREGATION,
)


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
        use_windowed_audio: bool = AUDIO_USE_WINDOWED,
        audio_aggregation: str = AUDIO_AGGREGATION,
    ):
        self.text = TextEmbedder()       # CLIP text encoder
        self.image = ImageEmbedder()     # CLIP image encoder
        self.audio = AudioEmbedder()     # CLAP audio encoder (also has text)

        # Identity projections when dims match (512 → 512)
        self.text_proj = ProjectionHead(512, target_dim)
        self.image_proj = ProjectionHead(512, target_dim)
        self.audio_proj = ProjectionHead(512, target_dim)

        # Windowed CLAP settings
        self._use_windowed = use_windowed_audio
        self._audio_aggregation = audio_aggregation
        self._audio_analyzer = None
        if self._use_windowed:
            from cmsci.embeddings.audio_analysis import AudioAnalyzer
            self._audio_analyzer = AudioAnalyzer(
                window_sec=AUDIO_WINDOW_SEC,
                hop_sec=AUDIO_HOP_SEC,
            )

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
        """CLAP audio embedding — use for text-audio comparison.

        When windowed mode is enabled, splits audio into overlapping windows,
        embeds each with CLAP, and aggregates (max or mean). Uses a distinct
        cache key to avoid collisions with single-clip embeddings.
        """
        cache_key = f"audio_windowed_{self._audio_aggregation}" if self._use_windowed else "audio"

        if self.cache:
            cached = self.cache.get(path, cache_key)
            if cached is not None:
                return cached

        if self._use_windowed and self._audio_analyzer is not None:
            emb = self._audio_analyzer.embed_windowed(
                path,
                embedder=self.audio,
                aggregation=self._audio_aggregation,
            )
        else:
            emb = self.audio.embed(path)

        projected = self.audio_proj.project(emb)

        if self.cache:
            self.cache.set(path, cache_key, projected)

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
        from cmsci.embeddings.shared_embedder import get_shared_embedder
        return get_shared_embedder(target_dim, enable_cache, cache_dir)
