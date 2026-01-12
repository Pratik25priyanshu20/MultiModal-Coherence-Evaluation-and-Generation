"""
Improved image generator with better retrieval and filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity


class ImprovedImageRetrievalGenerator:
    """
    Improved image retrieval with:
    - Better similarity thresholding
    - Multiple candidates with filtering
    - Fallback handling
    """

    def __init__(
        self,
        index_path: str = "data/embeddings/image_index.npz",
        min_similarity: float = 0.15,  # Minimum similarity threshold
        top_k: int = 5,  # Retrieve top K, then filter
    ):
        self.index_path = Path(index_path)
        self.min_similarity = min_similarity
        self.top_k = top_k

        if not self.index_path.exists():
            raise RuntimeError(
                f"[ImprovedImageRetrievalGenerator] Missing image index at {self.index_path}. "
                "Run scripts/build_embedding_indexes.py first."
            )

        data = np.load(self.index_path, allow_pickle=True)
        self.ids = data["ids"].tolist()
        self.embs = data["embs"].astype("float32")

        if len(self.ids) == 0:
            raise RuntimeError(
                "[ImprovedImageRetrievalGenerator] Image index is empty. "
                "Add images to data/processed/images/ and rebuild the index."
            )

        self.embedder = AlignedEmbedder(target_dim=512)

    def retrieve_top_k(
        self,
        query_text: str,
        k: int = 1,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top K images with similarity filtering.
        
        Returns list of (path, similarity_score) tuples, filtered by min_similarity.
        """
        if min_similarity is None:
            min_similarity = self.min_similarity

        query_emb = self.embedder.embed_text(query_text)
        scored = [
            (path, cosine_similarity(query_emb, emb))
            for path, emb in zip(self.ids, self.embs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum similarity
        filtered = [(path, score) for path, score in scored[:self.top_k] if score >= min_similarity]
        
        # If nothing meets threshold, return best match anyway (with warning)
        if not filtered and scored:
            return [(scored[0][0], scored[0][1])]  # Return best match even if below threshold
        
        return filtered[:k]


def generate_image_improved(
    prompt: str,
    out_dir: str,
    index_path: str = "data/embeddings/image_index.npz",
    min_similarity: float = 0.15,
) -> str:
    """
    Improved image generation with better filtering.
    """
    generator = ImprovedImageRetrievalGenerator(
        index_path=index_path,
        min_similarity=min_similarity,
    )
    results = generator.retrieve_top_k(prompt, k=1, min_similarity=min_similarity)
    if not results:
        raise RuntimeError("No images available for retrieval.")
    
    path, score = results[0]
    
    # Log if similarity is low (for debugging)
    if score < 0.2:
        import warnings
        warnings.warn(
            f"Low image similarity ({score:.3f}) for prompt: {prompt[:50]}... "
            f"Image: {Path(path).name}"
        )
    
    return path
