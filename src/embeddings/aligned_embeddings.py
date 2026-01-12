from __future__ import annotations

import numpy as np

from src.embeddings.audio_embedder import AudioEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.embeddings.projection import ProjectionHead
from src.embeddings.text_embedder import TextEmbedder


class AlignedEmbedder:
    """
    Ensures all modalities live in the same embedding space.
    """

    def __init__(self, target_dim: int = 512):
        self.text = TextEmbedder()
        self.image = ImageEmbedder()
        self.audio = AudioEmbedder()

        self.text_proj = ProjectionHead(512, target_dim)
        self.image_proj = ProjectionHead(512, target_dim)
        self.audio_proj = ProjectionHead(512, target_dim)

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.text.embed(text)
        return self.text_proj.project(emb)

    def embed_image(self, path: str) -> np.ndarray:
        emb = self.image.embed(path)
        return self.image_proj.project(emb)

    def embed_audio(self, path: str) -> np.ndarray:
        emb = self.audio.embed(path)
        return self.audio_proj.project(emb)
