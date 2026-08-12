"""
Multi-Scale Gramian Fusion for Matryoshka Embeddings.

Exploits the Matryoshka Representation Learning (MRL) property of Gemini
embeddings: truncating to smaller dimensions (768, 1536) captures different
semantic granularity levels.

By computing Gramian coherences at multiple scales and fusing them, we get a
more robust coherence signal than any single scale alone.

The key insight: coarse scales (768-d) capture broad topic agreement, while
fine scales (3072-d) capture nuanced semantic alignment. Fusing across scales
is analogous to multi-resolution analysis in signal processing.

Usage:
    from src.coherence.multiscale_gramian import MultiscaleGramian

    ms = MultiscaleGramian()
    result = ms.compute(text_3072, img_3072, aud_3072)
    # result["fused_ti"], result["fused_ta"], etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.coherence.gram_volume import (
    gram_volume_2d,
    gram_volume_3d,
    normalized_gram_coherence,
)
from src.config.settings import MATRYOSHKA_DIMS


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector."""
    v = v.astype(np.float64)
    norm = np.linalg.norm(v) + eps
    return v / norm


class MultiscaleGramian:
    """
    Compute Gramian coherences at multiple Matryoshka truncation scales
    and fuse them into a single coherence signal per channel.
    """

    def __init__(self, dims: Optional[List[int]] = None):
        self._dims = sorted(dims or MATRYOSHKA_DIMS)

    def _truncate_and_normalize(self, v: np.ndarray, dim: int) -> np.ndarray:
        """Truncate to `dim` dimensions and re-normalize."""
        return _l2_normalize(v[:dim])

    def compute(
        self,
        text_full: np.ndarray,
        image_full: Optional[np.ndarray] = None,
        audio_full: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Compute Gramian coherences at each MRL scale.

        Args:
            text_full: Full-dimension (3072-d) text embedding.
            image_full: Full-dimension image embedding (optional).
            audio_full: Full-dimension audio embedding (optional).

        Returns:
            Dict with per-scale and fused coherence values:
                - per_scale: {dim: {ti, ta, ia, tia}} coherences
                - fused_ti, fused_ta, fused_ia, fused_tia: equal-weight fused
                - scale_stds: {ti, ta, ia, tia} cross-scale standard deviations
        """
        per_scale = {}

        for dim in self._dims:
            t = self._truncate_and_normalize(text_full, dim)
            i = self._truncate_and_normalize(image_full, dim) if image_full is not None else None
            a = self._truncate_and_normalize(audio_full, dim) if audio_full is not None else None

            scale_result = {}

            if i is not None:
                vol_ti = gram_volume_2d(t, i)
                scale_result["ti"] = normalized_gram_coherence(vol_ti)
            if a is not None:
                vol_ta = gram_volume_2d(t, a)
                scale_result["ta"] = normalized_gram_coherence(vol_ta)
            if i is not None and a is not None:
                vol_ia = gram_volume_2d(i, a)
                scale_result["ia"] = normalized_gram_coherence(vol_ia)
                vol_tia = gram_volume_3d(t, i, a)
                scale_result["tia"] = normalized_gram_coherence(vol_tia, n_vectors=3)

            per_scale[dim] = scale_result

        # Fuse across scales (equal-weight average)
        fused = {}
        scale_stds = {}
        for channel in ["ti", "ta", "ia", "tia"]:
            values = [per_scale[d][channel] for d in self._dims if channel in per_scale[d]]
            if values:
                fused[channel] = float(np.mean(values))
                scale_stds[channel] = float(np.std(values))
            else:
                fused[channel] = None
                scale_stds[channel] = None

        return {
            "per_scale": per_scale,
            "fused_ti": fused.get("ti"),
            "fused_ta": fused.get("ta"),
            "fused_ia": fused.get("ia"),
            "fused_tia": fused.get("tia"),
            "scale_stds": scale_stds,
        }

    def fused_coherence(
        self,
        text_full: np.ndarray,
        image_full: Optional[np.ndarray] = None,
        audio_full: Optional[np.ndarray] = None,
        weights: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Compute weighted fusion of coherences across scales.

        Args:
            text_full: Full-dim text embedding.
            image_full: Full-dim image embedding.
            audio_full: Full-dim audio embedding.
            weights: Optional {dim: weight} mapping. Default: equal weights.

        Returns:
            Dict with fused coherence per channel (ti, ta, ia, tia).
        """
        result = self.compute(text_full, image_full, audio_full)
        per_scale = result["per_scale"]

        if weights is None:
            weights = {d: 1.0 / len(self._dims) for d in self._dims}

        fused = {}
        for channel in ["ti", "ta", "ia", "tia"]:
            num = 0.0
            denom = 0.0
            for dim in self._dims:
                if channel in per_scale[dim]:
                    w = weights.get(dim, 0.0)
                    num += w * per_scale[dim][channel]
                    denom += w
            fused[channel] = float(num / denom) if denom > 0 else None

        return fused
