"""
Audio retrieval generator — retrieves best-matching audio from indexed pool.

Mirrors the image retrieval approach:
- Uses CLAP text encoder to embed query
- Compares against CLAP audio embeddings in the audio index
- Returns best matching real audio file

Falls back to synthetic ambient if no audio index is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity
from src.exceptions import IndexError_

logger = logging.getLogger(__name__)


@dataclass
class AudioRetrievalResult:
    """Result of audio retrieval with metadata for experiment bundles."""
    audio_path: str
    similarity: float
    retrieval_failed: bool
    candidates_considered: int
    candidates_above_threshold: int
    top_5: List[Tuple[str, float]]


class AudioRetrievalGenerator:
    """
    Audio retrieval using CLAP shared space.

    Query: CLAP text embedding of the prompt
    Index: CLAP audio embeddings of real audio files

    This ensures st_a (text-audio similarity) is meaningful because
    both query and candidates live in CLAP's shared space.
    """

    def __init__(
        self,
        index_path: str = "data/embeddings/audio_index.npz",
        min_similarity: float = 0.10,
        top_k: int = 5,
    ):
        self.index_path = Path(index_path)
        self.min_similarity = min_similarity
        self.top_k = top_k

        if not self.index_path.exists():
            raise IndexError_(
                f"Missing audio index at {self.index_path}. "
                "Run: python scripts/build_embedding_indexes.py",
                index_path=str(self.index_path),
            )

        data = np.load(self.index_path, allow_pickle=True)
        self.ids = data["ids"].tolist()
        self.embs = data["embs"].astype("float32")

        if len(self.ids) == 0:
            raise IndexError_(
                "Audio index is empty. "
                "Add audio files and run: python scripts/build_embedding_indexes.py",
                index_path=str(self.index_path),
            )

        self.embedder = AlignedEmbedder(target_dim=512)

    def retrieve(
        self,
        query_text: str,
        min_similarity: Optional[float] = None,
    ) -> AudioRetrievalResult:
        """
        Retrieve best matching audio for a text query.

        Uses CLAP text encoder for the query (same space as indexed audio embeddings).
        """
        if min_similarity is None:
            min_similarity = self.min_similarity

        # Use CLAP text encoder (not CLIP) for audio retrieval
        query_emb = self.embedder.embed_text_for_audio(query_text)

        scored = []
        for audio_path, audio_emb in zip(self.ids, self.embs):
            sim = cosine_similarity(query_emb, audio_emb)
            scored.append((audio_path, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_5 = [(Path(p).name, s) for p, s in scored[:5]]

        above_threshold = [(p, s) for p, s in scored if s >= min_similarity]

        if above_threshold:
            best_path, best_sim = above_threshold[0]
            logger.debug("Audio retrieval: %s (sim=%.4f, %d/%d above threshold)",
                         Path(best_path).name, best_sim, len(above_threshold), len(scored))
            return AudioRetrievalResult(
                audio_path=best_path,
                similarity=best_sim,
                retrieval_failed=False,
                candidates_considered=len(scored),
                candidates_above_threshold=len(above_threshold),
                top_5=top_5,
            )

        # Fallback: return best candidate even if below threshold
        best_path, best_sim = scored[0]
        logger.warning("Audio retrieval below threshold: %s (sim=%.4f < %.2f)",
                       Path(best_path).name, best_sim, min_similarity)
        return AudioRetrievalResult(
            audio_path=best_path,
            similarity=best_sim,
            retrieval_failed=best_sim < min_similarity,
            candidates_considered=len(scored),
            candidates_above_threshold=0,
            top_5=top_5,
        )


def retrieve_audio_with_metadata(
    prompt: str,
    index_path: str = "data/embeddings/audio_index.npz",
    min_similarity: float = 0.10,
) -> AudioRetrievalResult:
    """
    Retrieve audio for a prompt and return full metadata.

    Use this in experiment pipelines where audio quality matters.
    """
    generator = AudioRetrievalGenerator(
        index_path=index_path,
        min_similarity=min_similarity,
    )
    return generator.retrieve(prompt, min_similarity=min_similarity)
