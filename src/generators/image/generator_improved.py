"""
Improved image generator with domain gating, similarity thresholding,
and explicit retrieval failure reporting.

Phase 1B+1C: Addresses retrieval reliability for controlled experiments.
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


# Domain keywords for gating — reject obvious mismatches
DOMAIN_KEYWORDS = {
    "nature": {"forest", "tree", "mountain", "jungle", "garden", "park", "field",
               "meadow", "countryside", "rural", "fog", "dawn", "sunrise", "hill",
               "valley", "woodland", "grove", "leaf", "green", "wildlife"},
    "urban": {"city", "street", "neon", "urban", "downtown", "skyscraper",
              "building", "traffic", "night", "cobblestone", "road", "car",
              "sign", "shop", "window", "concrete", "sidewalk"},
    "water": {"beach", "ocean", "wave", "sea", "shore", "coast", "lake",
              "river", "water", "sand", "surf", "tide", "tropical", "island"},
}

# Domains that should NOT co-occur in prompt+image
INCOMPATIBLE_DOMAINS = {
    "nature": {"urban"},
    "urban": {"nature", "water"},
    "water": {"urban"},
}


@dataclass
class ImageRetrievalResult:
    """Result of image retrieval with metadata for experiment bundles."""
    image_path: str
    similarity: float
    domain: str
    retrieval_failed: bool
    candidates_considered: int
    candidates_above_threshold: int
    top_5: List[Tuple[str, float]]


def _detect_prompt_domain(prompt: str) -> Optional[str]:
    """Detect the primary domain of a prompt from keywords."""
    prompt_lower = prompt.lower()
    prompt_words = set(prompt_lower.split())

    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        overlap = len(prompt_words & keywords)
        # Also check substring matches for compound words
        substring_hits = sum(1 for kw in keywords if kw in prompt_lower)
        scores[domain] = overlap + substring_hits

    if not scores or max(scores.values()) == 0:
        return None

    best_domain = max(scores, key=scores.get)
    return best_domain


def _is_domain_compatible(prompt_domain: Optional[str], image_domain: str) -> bool:
    """Check if image domain is compatible with prompt domain."""
    if prompt_domain is None:
        return True  # No domain detected, allow everything
    if image_domain == "other":
        return True  # Unknown domain, don't reject
    incompatible = INCOMPATIBLE_DOMAINS.get(prompt_domain, set())
    return image_domain not in incompatible


class ImprovedImageRetrievalGenerator:
    """
    Image retrieval with:
    - Domain gating: rejects obvious domain mismatches (forest prompt → no city images)
    - Raised similarity floor: min_similarity=0.20 (was 0.15)
    - Explicit retrieval failure: returns retrieval_failed=True instead of silent nonsense
    - Full diagnostic metadata for experiment bundles
    """

    def __init__(
        self,
        index_path: str = "data/embeddings/image_index.npz",
        min_similarity: float = 0.20,
        top_k: int = 5,
    ):
        self.index_path = Path(index_path)
        self.min_similarity = min_similarity
        self.top_k = top_k

        if not self.index_path.exists():
            raise IndexError_(
                f"Missing image index at {self.index_path}. "
                "Run: python scripts/build_embedding_indexes.py",
                index_path=str(self.index_path),
            )

        data = np.load(self.index_path, allow_pickle=True)
        self.ids = data["ids"].tolist()
        self.embs = data["embs"].astype("float32")

        # Load domain tags if available (from rebuilt index)
        if "domains" in data:
            self.domains = data["domains"].tolist()
        else:
            # Infer from filenames for old indexes
            self.domains = [self._infer_domain(p) for p in self.ids]

        if len(self.ids) == 0:
            raise IndexError_(
                "Image index is empty. "
                "Add images and run: python scripts/build_embedding_indexes.py",
                index_path=str(self.index_path),
            )

        self.embedder = AlignedEmbedder(target_dim=512)

    @staticmethod
    def _infer_domain(filepath: str) -> str:
        """Infer domain from filename."""
        name = Path(filepath).stem.lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                return domain
        return "other"

    def retrieve(
        self,
        query_text: str,
        min_similarity: Optional[float] = None,
    ) -> ImageRetrievalResult:
        """
        Retrieve best matching image with domain gating and quality checks.

        Returns ImageRetrievalResult with full metadata including retrieval_failed flag.
        """
        if min_similarity is None:
            min_similarity = self.min_similarity

        prompt_domain = _detect_prompt_domain(query_text)
        query_emb = self.embedder.embed_text(query_text)

        # Score all candidates
        scored = []
        for img_path, img_emb, img_domain in zip(self.ids, self.embs, self.domains):
            sim = cosine_similarity(query_emb, img_emb)
            scored.append((img_path, sim, img_domain))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_5 = [(Path(p).name, s) for p, s, _ in scored[:5]]

        # Phase 1: Domain gating — filter out incompatible domains
        domain_filtered = [
            (p, s, d) for p, s, d in scored
            if _is_domain_compatible(prompt_domain, d)
        ]

        # Phase 2: Similarity thresholding
        candidates = domain_filtered if domain_filtered else scored
        above_threshold = [(p, s, d) for p, s, d in candidates if s >= min_similarity]

        if above_threshold:
            # Best candidate passes both domain and similarity checks
            best_path, best_sim, best_domain = above_threshold[0]
            return ImageRetrievalResult(
                image_path=best_path,
                similarity=best_sim,
                domain=best_domain,
                retrieval_failed=False,
                candidates_considered=len(scored),
                candidates_above_threshold=len(above_threshold),
                top_5=top_5,
            )

        # Phase 3: Fallback — nothing passed threshold
        # Return best domain-compatible candidate (even if below threshold)
        if domain_filtered:
            best_path, best_sim, best_domain = domain_filtered[0]
        else:
            best_path, best_sim, best_domain = scored[0]

        return ImageRetrievalResult(
            image_path=best_path,
            similarity=best_sim,
            domain=best_domain,
            retrieval_failed=best_sim < min_similarity,
            candidates_considered=len(scored),
            candidates_above_threshold=0,
            top_5=top_5,
        )

    # Backward-compatible method
    def retrieve_top_k(
        self,
        query_text: str,
        k: int = 1,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Backward-compatible interface. Returns list of (path, score) tuples."""
        result = self.retrieve(query_text, min_similarity)
        return [(result.image_path, result.similarity)]


def generate_image_improved(
    prompt: str,
    out_dir: str,
    index_path: str = "data/embeddings/image_index.npz",
    min_similarity: float = 0.20,
) -> str:
    """
    Generate (retrieve) an image for a prompt.

    Returns the image path. Warns on low similarity or retrieval failure.
    """
    generator = ImprovedImageRetrievalGenerator(
        index_path=index_path,
        min_similarity=min_similarity,
    )
    result = generator.retrieve(prompt, min_similarity=min_similarity)

    if result.retrieval_failed:
        logger.warning(
            "Image retrieval failed: no image above threshold (%.2f) "
            "for prompt: \"%s...\" — best: %s (sim=%.4f, domain=%s)",
            min_similarity, prompt[:60], Path(result.image_path).name,
            result.similarity, result.domain,
        )
    elif result.similarity < 0.25:
        logger.warning(
            "Low image similarity: %.4f for \"%s...\" -> %s",
            result.similarity, prompt[:60], Path(result.image_path).name,
        )

    return result.image_path


def generate_image_with_metadata(
    prompt: str,
    index_path: str = "data/embeddings/image_index.npz",
    min_similarity: float = 0.20,
) -> ImageRetrievalResult:
    """
    Generate (retrieve) an image and return full metadata.

    Use this in experiment pipelines where retrieval quality matters.
    """
    generator = ImprovedImageRetrievalGenerator(
        index_path=index_path,
        min_similarity=min_similarity,
    )
    return generator.retrieve(prompt, min_similarity=min_similarity)
