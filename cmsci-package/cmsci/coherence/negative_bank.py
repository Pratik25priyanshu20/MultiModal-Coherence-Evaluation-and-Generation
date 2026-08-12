"""
Contrastive Negative Bank for cMSCI Calibration.

Computes contrastive margins by comparing a matched (text, image, audio)
triple against hard-negative alternatives from the embedding indexes.

A positive contrastive margin means the matched triple has tighter
geometric coherence than mismatched alternatives — the defining
property of a well-calibrated metric.

Per-channel contrastive margins (computed within each embedding space):
    m_ti = E[V_ti^neg] - V_ti^match    (CLIP space)
    m_ta = E[V_ta^neg] - V_ta^match    (CLAP space)
    m = w_ti * m_ti + (1-w_ti) * m_ta   (combined by engine)

    > 0 → matched pair is more coherent than negatives (good)
    ≤ 0 → metric cannot distinguish matched from mismatched (bad)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cmsci.coherence.gram_volume import gram_volume_2d, gram_volume_3d, normalized_gram_coherence
from cmsci.embeddings.similarity import l2_normalize

logger = logging.getLogger(__name__)


class NegativeBank:
    """
    Loads pre-computed embedding indexes and provides hard negatives.

    Hard negatives are embeddings with high individual similarity to the
    query but from a different domain — the most challenging cases for
    the coherence metric.
    """

    def __init__(
        self,
        image_index_path: Optional[str] = None,
        audio_index_path: Optional[str] = None,
    ):
        if image_index_path is None or audio_index_path is None:
            from cmsci.assets import resolve_assets
            _a = resolve_assets()
            image_index_path = image_index_path or (str(_a.image_index) if _a.image_index else "data/embeddings/image_index.npz")
            audio_index_path = audio_index_path or (str(_a.audio_index) if _a.audio_index else "data/embeddings/audio_index.npz")
        self._image_ids: Optional[np.ndarray] = None
        self._image_embs: Optional[np.ndarray] = None
        self._image_domains: Optional[np.ndarray] = None
        self._audio_ids: Optional[np.ndarray] = None
        self._audio_embs: Optional[np.ndarray] = None
        self._audio_domains: Optional[np.ndarray] = None

        self._load_index(image_index_path, "image")
        self._load_index(audio_index_path, "audio")

    def _load_index(self, path: str, modality: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("Index not found: %s — %s negatives disabled", path, modality)
            return

        data = np.load(path, allow_pickle=True)
        ids = data["ids"] if "ids" in data else data.get("paths", np.array([]))
        embs = data["embs"] if "embs" in data else data.get("embeddings", np.array([]))
        domains = data["domains"] if "domains" in data else np.array(["other"] * len(ids))

        if modality == "image":
            self._image_ids = ids
            self._image_embs = embs.astype(np.float32)
            self._image_domains = domains
            logger.info("Loaded image index: %d entries", len(ids))
        else:
            self._audio_ids = ids
            self._audio_embs = embs.astype(np.float32)
            self._audio_domains = domains
            logger.info("Loaded audio index: %d entries", len(ids))

    @property
    def has_images(self) -> bool:
        return self._image_embs is not None and len(self._image_embs) > 0

    @property
    def has_audio(self) -> bool:
        return self._audio_embs is not None and len(self._audio_embs) > 0

    def get_hard_negative_images(
        self,
        text_emb: np.ndarray,
        exclude_domain: str = "",
        k: int = 5,
    ) -> List[np.ndarray]:
        """
        Get top-k hardest negative images (high text similarity but wrong domain).

        Args:
            text_emb: CLIP text embedding for the query.
            exclude_domain: Domain to exclude (the correct domain).
            k: Number of negatives to return.

        Returns:
            List of image embeddings (hard negatives).
        """
        if not self.has_images:
            return []

        text_n = l2_normalize(text_emb.squeeze())
        sims = self._image_embs @ text_n

        # Filter by domain: exclude the matched domain
        if exclude_domain:
            mask = np.array([d != exclude_domain for d in self._image_domains])
        else:
            mask = np.ones(len(sims), dtype=bool)

        sims_masked = np.where(mask, sims, -np.inf)
        top_k_idx = np.argsort(sims_masked)[-k:][::-1]

        return [self._image_embs[i] for i in top_k_idx if sims_masked[i] > -np.inf]

    def get_hard_negative_audio(
        self,
        text_emb: np.ndarray,
        exclude_domain: str = "",
        k: int = 5,
    ) -> List[np.ndarray]:
        """
        Get top-k hardest negative audio (high text similarity but wrong domain).

        Args:
            text_emb: CLAP text embedding for the query.
            exclude_domain: Domain to exclude.
            k: Number of negatives to return.

        Returns:
            List of audio embeddings (hard negatives).
        """
        if not self.has_audio:
            return []

        text_n = l2_normalize(text_emb.squeeze())
        sims = self._audio_embs @ text_n

        if exclude_domain:
            mask = np.array([d != exclude_domain for d in self._audio_domains])
        else:
            mask = np.ones(len(sims), dtype=bool)

        sims_masked = np.where(mask, sims, -np.inf)
        top_k_idx = np.argsort(sims_masked)[-k:][::-1]

        return [self._audio_embs[i] for i in top_k_idx if sims_masked[i] > -np.inf]

    def compute_contrastive_margin(
        self,
        gram_ti: Optional[float],
        gram_ta: Optional[float],
        text_clip_emb: np.ndarray,
        image_emb: Optional[np.ndarray] = None,
        text_clap_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
        domain: str = "",
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute per-channel contrastive margins against hard negatives.

        Margins are computed within each embedding space separately to
        avoid mixing heterogeneous distributions:

            m_ti = E[V_ti^neg] - V_ti^match    (CLIP space)
            m_ta = E[V_ta^neg] - V_ta^match    (CLAP space)

        The caller (engine) combines: m = w_ti * m_ti + (1-w_ti) * m_ta.

        A positive per-channel margin means the matched pair has tighter
        geometric coherence than hard negatives in that space.

        Args:
            gram_ti: Matched text-image Gramian volume (CLIP space).
            gram_ta: Matched text-audio Gramian volume (CLAP space).
            text_clip_emb: CLIP text embedding (for finding negative images).
            image_emb: CLIP image embedding of the matched image.
            text_clap_emb: CLAP text embedding (for finding negative audio).
            audio_emb: CLAP audio embedding of the matched audio.
            domain: Domain of the matched prompt (excluded from negatives).
            k: Number of hard negatives per channel.

        Returns:
            Dict with per-channel margins (margin_ti, margin_ta),
            per-channel statistics, and total negative count.
        """
        margin_ti = 0.0
        margin_ta = 0.0
        mean_neg_ti = gram_ti if gram_ti is not None else 0.0
        mean_neg_ta = gram_ta if gram_ta is not None else 0.0
        n_neg_ti = 0
        n_neg_ta = 0

        # Text-Image margin (CLIP space only)
        if gram_ti is not None and image_emb is not None:
            neg_images = self.get_hard_negative_images(text_clip_emb, domain, k)
            if neg_images:
                neg_vols_ti = [gram_volume_2d(text_clip_emb, neg_img)
                               for neg_img in neg_images]
                mean_neg_ti = float(np.mean(neg_vols_ti))
                margin_ti = mean_neg_ti - gram_ti
                n_neg_ti = len(neg_vols_ti)

        # Text-Audio margin (CLAP space only)
        if gram_ta is not None and text_clap_emb is not None:
            neg_audios = self.get_hard_negative_audio(text_clap_emb, domain, k)
            if neg_audios:
                neg_vols_ta = [gram_volume_2d(text_clap_emb, neg_aud)
                               for neg_aud in neg_audios]
                mean_neg_ta = float(np.mean(neg_vols_ta))
                margin_ta = mean_neg_ta - gram_ta
                n_neg_ta = len(neg_vols_ta)

        return {
            "margin_ti": float(margin_ti),
            "margin_ta": float(margin_ta),
            "mean_neg_volume_ti": float(mean_neg_ti),
            "mean_neg_volume_ta": float(mean_neg_ta),
            "n_negatives_ti": n_neg_ti,
            "n_negatives_ta": n_neg_ta,
            "n_negatives": n_neg_ti + n_neg_ta,
        }
