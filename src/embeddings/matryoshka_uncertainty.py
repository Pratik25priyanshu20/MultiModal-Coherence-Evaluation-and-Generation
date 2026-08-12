"""
Matryoshka Scale Consistency — Training-Free Uncertainty Estimation.

Novel contribution: replaces ProbVLM (which requires trained adapters) by
exploiting the Matryoshka Representation Learning (MRL) property of Gemini
Embedding 2. Coherence is measured at multiple truncation dimensions
(768, 1536, 3072) and stability across scales indicates confidence.

Intuition:
    - A truly coherent triple (text, image, audio) will show HIGH coherence
      at ALL scales — the relationship is captured even in low dimensions.
    - An ambiguous or weakly-related triple will show VARIABLE coherence
      across scales — lower dimensions lose the fragile alignment.

    consistency = 1 - std(coherences) / (mean(coherences) + eps)

This is training-free, requires no adapter weights, and works with any
Matryoshka-compatible embedding model.

Usage:
    from src.embeddings.matryoshka_uncertainty import MatryoshkaUncertainty
    mu = MatryoshkaUncertainty()
    result = mu.compute_scale_consistency(text_3072, img_3072, aud_3072)
    print(result["consistency"])  # 0.0-1.0, higher = more confident
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.coherence.gram_volume import (
    gram_volume_2d,
    gram_volume_3d,
    normalized_gram_coherence,
)
from src.config.settings import MATRYOSHKA_DIMS

logger = logging.getLogger(__name__)


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector."""
    v = v.astype(np.float64)
    norm = np.linalg.norm(v) + eps
    return v / norm


def _truncate_and_normalize(emb: np.ndarray, dim: int) -> np.ndarray:
    """Truncate to dim and L2-renormalize (Matryoshka prefix property)."""
    return _l2_normalize(emb[:dim])


class MatryoshkaUncertainty:
    """
    Training-free uncertainty estimation via Matryoshka scale consistency.

    Measures how stable coherence scores are across MRL truncation
    dimensions. High consistency = high confidence, low consistency =
    uncertain alignment that depends on fine-grained features.
    """

    def __init__(self, dims: Optional[List[int]] = None):
        """
        Args:
            dims: Matryoshka truncation dimensions. Default: [768, 1536, 3072].
        """
        self.dims = sorted(dims or MATRYOSHKA_DIMS)

    def compute_scale_consistency(
        self,
        text_emb: np.ndarray,
        image_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute overall scale consistency across all available channels.

        Uses 3D Gramian volume when all three modalities are present,
        otherwise falls back to average of 2D volumes.

        Args:
            text_emb: Full-dimension text embedding (3072-d).
            image_emb: Full-dimension image embedding (3072-d), optional.
            audio_emb: Full-dimension audio embedding (3072-d), optional.

        Returns:
            Dict with:
                consistency: float in [0, 1] — stability measure
                coherences: list of coherence at each scale
                std: standard deviation of coherences
                mean: mean coherence across scales
        """
        coherences = []

        for dim in self.dims:
            t = _truncate_and_normalize(text_emb, dim)
            i = _truncate_and_normalize(image_emb, dim) if image_emb is not None else None
            a = _truncate_and_normalize(audio_emb, dim) if audio_emb is not None else None

            if i is not None and a is not None:
                # 3D Gramian: exact tri-modal coherence
                vol = gram_volume_3d(t, i, a)
                coh = normalized_gram_coherence(vol, n_vectors=3)
            elif i is not None:
                vol = gram_volume_2d(t, i)
                coh = normalized_gram_coherence(vol)
            elif a is not None:
                vol = gram_volume_2d(t, a)
                coh = normalized_gram_coherence(vol)
            else:
                continue

            coherences.append(coh)

        if len(coherences) < 2:
            return {
                "consistency": 1.0,
                "coherences": coherences,
                "std": 0.0,
                "mean": coherences[0] if coherences else 0.0,
            }

        mean_coh = float(np.mean(coherences))
        std_coh = float(np.std(coherences))
        eps = 1e-8
        consistency = float(1.0 - std_coh / (mean_coh + eps))
        consistency = max(0.0, min(1.0, consistency))

        return {
            "consistency": consistency,
            "coherences": coherences,
            "std": std_coh,
            "mean": mean_coh,
        }

    def compute_per_channel_consistency(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> Dict:
        """
        Compute scale consistency for a single pairwise channel.

        Args:
            emb1: First embedding (full dimension).
            emb2: Second embedding (full dimension).

        Returns:
            Dict with consistency, coherences list, std.
        """
        coherences = []
        for dim in self.dims:
            v1 = _truncate_and_normalize(emb1, dim)
            v2 = _truncate_and_normalize(emb2, dim)
            vol = gram_volume_2d(v1, v2)
            coherences.append(normalized_gram_coherence(vol))

        if len(coherences) < 2:
            return {
                "consistency": 1.0,
                "coherences": coherences,
                "std": 0.0,
            }

        mean_coh = float(np.mean(coherences))
        std_coh = float(np.std(coherences))
        eps = 1e-8
        consistency = float(1.0 - std_coh / (mean_coh + eps))
        consistency = max(0.0, min(1.0, consistency))

        return {
            "consistency": consistency,
            "coherences": coherences,
            "std": std_coh,
        }

    def compute_adaptive_weights(
        self,
        text_emb: np.ndarray,
        image_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute adaptive channel weights from per-channel scale consistency.

        More consistent channels get higher weights, analogous to ProbVLM's
        inverse-uncertainty weighting but computed without any trained models.

        w_ti = c_ti / (c_ti + c_ta + eps)
        w_ta = c_ta / (c_ti + c_ta + eps)

        Args:
            text_emb: Full-dimension text embedding.
            image_emb: Full-dimension image embedding (optional).
            audio_emb: Full-dimension audio embedding (optional).

        Returns:
            Dict with:
                w_ti: text-image weight
                w_ta: text-audio weight
                w_ia: image-audio weight (if both present)
                c_ti: text-image consistency
                c_ta: text-audio consistency
                c_ia: image-audio consistency (if both present)
        """
        c_ti = 0.0
        c_ta = 0.0
        c_ia = 0.0

        if image_emb is not None:
            result_ti = self.compute_per_channel_consistency(text_emb, image_emb)
            c_ti = result_ti["consistency"]

        if audio_emb is not None:
            result_ta = self.compute_per_channel_consistency(text_emb, audio_emb)
            c_ta = result_ta["consistency"]

        if image_emb is not None and audio_emb is not None:
            result_ia = self.compute_per_channel_consistency(image_emb, audio_emb)
            c_ia = result_ia["consistency"]

        eps = 1e-8

        # 2-channel weights (text-image vs text-audio)
        total_2ch = c_ti + c_ta + eps
        w_ti = c_ti / total_2ch
        w_ta = c_ta / total_2ch

        # 3-channel weights (for complementarity mixing)
        total_3ch = c_ti + c_ta + c_ia + eps
        w_ti_3 = c_ti / total_3ch
        w_ta_3 = c_ta / total_3ch
        w_ia_3 = c_ia / total_3ch

        return {
            "w_ti": float(w_ti),
            "w_ta": float(w_ta),
            "w_ia": float(w_ia_3) if image_emb is not None and audio_emb is not None else 0.0,
            "w_ti_3ch": float(w_ti_3),
            "w_ta_3ch": float(w_ta_3),
            "w_ia_3ch": float(w_ia_3),
            "c_ti": float(c_ti),
            "c_ta": float(c_ta),
            "c_ia": float(c_ia),
        }
