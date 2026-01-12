from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity


class ImageRetrievalGenerator:
    """
    V1 image generator via retrieval.
    """

    def __init__(self, index_path: str = "data/embeddings/image_index.npz"):
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise RuntimeError(
                f"[ImageRetrievalGenerator] Missing image index at {self.index_path}. "
                "Run scripts/build_embedding_indexes.py first."
            )

        data = np.load(self.index_path, allow_pickle=True)
        self.ids = data["ids"].tolist()
        self.embs = data["embs"].astype("float32")

        if len(self.ids) == 0:
            raise RuntimeError(
                "[ImageRetrievalGenerator] Image index is empty. "
                "Add images to data/processed/images/ and rebuild the index."
            )

        self.embedder = AlignedEmbedder(target_dim=512)

    def retrieve_top_k(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        query_emb = self.embedder.embed_text(query_text)
        scored = [
            (path, cosine_similarity(query_emb, emb))
            for path, emb in zip(self.ids, self.embs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def generate_image(
    prompt: str,
    out_dir: str,
    index_path: str = "data/embeddings/image_index.npz",
) -> str:
    generator = ImageRetrievalGenerator(index_path=index_path)
    results = generator.retrieve_top_k(prompt, k=1)
    if not results:
        raise RuntimeError("No images available for retrieval.")
    return results[0][0]
