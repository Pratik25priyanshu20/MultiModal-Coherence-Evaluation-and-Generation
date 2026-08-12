"""
Gemini Cosine Baselines.

Raw cosine similarity baselines using Gemini Embedding 2's unified space.
These are the Gemini equivalents of the CLIP+CLAP cosine baselines, but
with the key advantage that image-audio similarity is now meaningful.

Baselines:
    - 2-channel: (cos(t,i) + cos(t,a)) / 2  (same channels as CLIP+CLAP)
    - 3-channel: (cos(t,i) + cos(t,a) + cos(i,a)) / 3  (novel!)
    - gram_3d: Gramian 3D coherence (geometric, no calibration)

Usage:
    baseline = GeminiCosineBaseline()
    scores = baseline.evaluate_all(samples, human_scores)
    print(scores["3ch"]["rho"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from src.coherence.gram_volume import (
    gram_volume_2d,
    gram_volume_3d,
    normalized_gram_coherence,
)
from src.embeddings.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class GeminiCosineBaseline:
    """
    Raw cosine and geometric baselines using Gemini unified embeddings.
    """

    def __init__(self):
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from src.embeddings.gemini_embedder import GeminiEmbedder
            self._embedder = GeminiEmbedder.shared()
        return self._embedder

    def score_2ch(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        """
        2-channel cosine: (cos(t,i) + cos(t,a)) / 2.
        Same channels as CLIP+CLAP baseline for fair comparison.
        """
        embedder = self._get_embedder()
        emb_t = embedder.embed_text(text)

        scores = []
        if image_path:
            emb_i = embedder.embed_image(image_path)
            scores.append(cosine_similarity(emb_t, emb_i))
        if audio_path:
            emb_a = embedder.embed_audio(audio_path)
            scores.append(cosine_similarity(emb_t, emb_a))

        return float(np.mean(scores)) if scores else None

    def score_3ch(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        """
        3-channel cosine: (cos(t,i) + cos(t,a) + cos(i,a)) / 3.
        Novel: image-audio channel was impossible with CLIP+CLAP.
        """
        embedder = self._get_embedder()
        emb_t = embedder.embed_text(text)
        emb_i = embedder.embed_image(image_path) if image_path else None
        emb_a = embedder.embed_audio(audio_path) if audio_path else None

        scores = []
        if emb_i is not None:
            scores.append(cosine_similarity(emb_t, emb_i))
        if emb_a is not None:
            scores.append(cosine_similarity(emb_t, emb_a))
        if emb_i is not None and emb_a is not None:
            scores.append(cosine_similarity(emb_i, emb_a))

        return float(np.mean(scores)) if scores else None

    def score_gram3d(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        """
        Gramian 3D coherence (geometric, no calibration).
        Uses exact 3D Gramian when all three modalities are present.
        """
        embedder = self._get_embedder()
        emb_t = embedder.embed_text(text)
        emb_i = embedder.embed_image(image_path) if image_path else None
        emb_a = embedder.embed_audio(audio_path) if audio_path else None

        if emb_i is not None and emb_a is not None:
            vol = gram_volume_3d(emb_t, emb_i, emb_a)
            return float(normalized_gram_coherence(vol, n_vectors=3))

        # Fallback to 2D average
        coherences = []
        if emb_i is not None:
            coherences.append(normalized_gram_coherence(gram_volume_2d(emb_t, emb_i)))
        if emb_a is not None:
            coherences.append(normalized_gram_coherence(gram_volume_2d(emb_t, emb_a)))

        return float(np.mean(coherences)) if coherences else None

    def evaluate_all(
        self,
        samples: List[Dict[str, Any]],
        human_scores: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all Gemini baselines and compute Spearman correlations.

        Args:
            samples: List of sample dicts with text, image_path, audio_path.
            human_scores: Dict mapping sample_id -> {"weighted_score": {"mean": float}}.

        Returns:
            Dict with keys "2ch", "3ch", "gram3d", each containing:
                rho, p, sig, scores list.
        """
        results = {}

        for method_name, score_fn in [
            ("2ch", self.score_2ch),
            ("3ch", self.score_3ch),
            ("gram3d", self.score_gram3d),
        ]:
            method_scores = []
            human_vals = []

            for s in samples:
                sid = s.get("sample_id", "")
                if sid not in human_scores:
                    continue

                score = score_fn(
                    text=s.get("prompt_text", s.get("text", "")),
                    image_path=s.get("image_path"),
                    audio_path=s.get("audio_path"),
                )
                if score is not None:
                    method_scores.append(score)
                    human_vals.append(human_scores[sid]["weighted_score"]["mean"])

            if len(method_scores) >= 5:
                rho, p = sp_stats.spearmanr(method_scores, human_vals)
                results[method_name] = {
                    "rho": float(rho),
                    "p": float(p),
                    "sig": p < 0.05,
                    "n": len(method_scores),
                    "scores": method_scores,
                }
            else:
                results[method_name] = {
                    "rho": None, "p": None, "sig": False,
                    "n": len(method_scores), "scores": method_scores,
                }

        return results
