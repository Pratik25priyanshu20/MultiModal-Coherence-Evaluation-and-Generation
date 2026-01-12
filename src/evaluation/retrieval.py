from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.embeddings.similarity import cosine_similarity


def retrieve_top_k(
    query_emb: np.ndarray,
    candidates: List[Tuple[str, np.ndarray]],
    k: int = 5,
):
    scored = [(item_id, cosine_similarity(query_emb, emb)) for item_id, emb in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
