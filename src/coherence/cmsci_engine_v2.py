"""
Calibrated Multimodal Semantic Coherence Index v2 (cMSCI v2) Engine.

Gemini-powered variant using a single unified embedding space (3072-d).
Proves cMSCI is embedding-agnostic by running the same calibration pipeline
on a fundamentally different backbone.

Key differences from v1 (CLIP+CLAP):
    - ONE text encoder (not two)
    - Image-audio channel is directly meaningful (no bridge/ExMCR needed)
    - Exact 3D Gramian volume (not an approximation via cross-space projection)
    - Matryoshka uncertainty replaces ProbVLM (training-free)
    - 3-channel contrastive margins (ti, ta, ia — all in same space)

Variant progression:
    A: Raw cosine average (3 channels: ti, ta, ia)
    B: GRAM-only (geometric, 2D avg or exact 3D)
    C: GRAM + z-norm (calibrated geometric)
    D: GRAM + z-norm + 3-channel contrastive margins
    E: + complementarity + Matryoshka adaptive weighting (full cMSCI v2)

Safety guarantee: gamma_mrl=0, w_compl=0 recovers Variant C exactly.

Usage:
    engine = CalibratedCoherenceEngineV2()
    result = engine.evaluate("A beach at sunset", "beach.jpg", "waves.wav")
    print(result["cmsci_v2"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.coherence.gram_volume import (
    gram_volume_2d,
    gram_volume_3d,
    normalized_gram_coherence,
)
from src.config.settings import (
    CMSCI_V2_ALPHA,
    CMSCI_V2_CAL_MODE,
    CMSCI_V2_GAMMA_MRL,
    CMSCI_V2_NEGATIVE_K,
    CMSCI_V2_USE_MULTISCALE,
    CMSCI_V2_W_COMPL,
    CMSCI_V2_W_IA,
    CMSCI_V2_W_TI,
    GEMINI_AUDIO_INDEX_PATH,
    GEMINI_CALIBRATION_PATH,
    GEMINI_IMAGE_INDEX_PATH,
)
from src.embeddings.similarity import cosine_similarity, l2_normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Negative Bank V2 — Unified Space
# ---------------------------------------------------------------------------

class NegativeBankV2:
    """
    Contrastive negative bank for Gemini unified embedding space.

    Unlike v1 which has separate CLIP/CLAP banks, v2 has a single bank
    per modality — all embeddings live in the same 3072-d space.
    This enables 3-channel contrastive margins (ti, ta, AND ia).
    """

    def __init__(
        self,
        image_index_path: str = "",
        audio_index_path: str = "",
    ):
        image_index_path = image_index_path or str(GEMINI_IMAGE_INDEX_PATH)
        audio_index_path = audio_index_path or str(GEMINI_AUDIO_INDEX_PATH)

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
            logger.warning("Gemini index not found: %s — %s negatives disabled", path, modality)
            return

        data = np.load(path, allow_pickle=True)
        ids = data["ids"] if "ids" in data else np.array([])
        embs = data["embs"] if "embs" in data else np.array([])
        domains = data["domains"] if "domains" in data else np.array(["other"] * len(ids))

        if modality == "image":
            self._image_ids = ids
            self._image_embs = embs.astype(np.float32)
            self._image_domains = domains
            logger.info("Loaded Gemini image index: %d entries", len(ids))
        else:
            self._audio_ids = ids
            self._audio_embs = embs.astype(np.float32)
            self._audio_domains = domains
            logger.info("Loaded Gemini audio index: %d entries", len(ids))

    @property
    def has_images(self) -> bool:
        return self._image_embs is not None and len(self._image_embs) > 0

    @property
    def has_audio(self) -> bool:
        return self._audio_embs is not None and len(self._audio_embs) > 0

    def _get_hard_negatives(
        self,
        query_emb: np.ndarray,
        bank_embs: np.ndarray,
        bank_domains: np.ndarray,
        exclude_domain: str = "",
        k: int = 5,
    ) -> List[np.ndarray]:
        """Get top-k hard negatives from a bank (high similarity, wrong domain)."""
        query_n = l2_normalize(query_emb.squeeze())
        sims = bank_embs @ query_n

        if exclude_domain:
            mask = np.array([d != exclude_domain for d in bank_domains])
        else:
            mask = np.ones(len(sims), dtype=bool)

        sims_masked = np.where(mask, sims, -np.inf)
        top_k_idx = np.argsort(sims_masked)[-k:][::-1]

        return [bank_embs[i] for i in top_k_idx if sims_masked[i] > -np.inf]

    def compute_contrastive_margin(
        self,
        gram_ti: Optional[float],
        gram_ta: Optional[float],
        gram_ia: Optional[float],
        text_emb: np.ndarray,
        image_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
        domain: str = "",
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute 3-channel contrastive margins in unified space.

        All three margins (ti, ta, ia) are computed in the SAME space,
        unlike v1 where ti was in CLIP space and ta was in CLAP space.

        margin = E[V^neg] - V^match  (positive = matched is more coherent)
        """
        margin_ti = 0.0
        margin_ta = 0.0
        margin_ia = 0.0
        n_neg_ti = 0
        n_neg_ta = 0
        n_neg_ia = 0

        # Text-Image margin: replace matched image with hard negative images
        if gram_ti is not None and image_emb is not None and self.has_images:
            negs = self._get_hard_negatives(
                text_emb, self._image_embs, self._image_domains, domain, k
            )
            if negs:
                neg_vols = [gram_volume_2d(text_emb, neg) for neg in negs]
                margin_ti = float(np.mean(neg_vols)) - gram_ti
                n_neg_ti = len(negs)

        # Text-Audio margin: replace matched audio with hard negative audio
        if gram_ta is not None and audio_emb is not None and self.has_audio:
            negs = self._get_hard_negatives(
                text_emb, self._audio_embs, self._audio_domains, domain, k
            )
            if negs:
                neg_vols = [gram_volume_2d(text_emb, neg) for neg in negs]
                margin_ta = float(np.mean(neg_vols)) - gram_ta
                n_neg_ta = len(negs)

        # Image-Audio margin (NEW in v2 — impossible in v1 without bridge)
        if gram_ia is not None and image_emb is not None and self.has_audio:
            negs = self._get_hard_negatives(
                image_emb, self._audio_embs, self._audio_domains, domain, k
            )
            if negs:
                neg_vols = [gram_volume_2d(image_emb, neg) for neg in negs]
                margin_ia = float(np.mean(neg_vols)) - gram_ia
                n_neg_ia = len(negs)

        return {
            "margin_ti": float(margin_ti),
            "margin_ta": float(margin_ta),
            "margin_ia": float(margin_ia),
            "n_negatives_ti": n_neg_ti,
            "n_negatives_ta": n_neg_ta,
            "n_negatives_ia": n_neg_ia,
            "n_negatives": n_neg_ti + n_neg_ta + n_neg_ia,
        }


# ---------------------------------------------------------------------------
# cMSCI v2 Engine
# ---------------------------------------------------------------------------

class CalibratedCoherenceEngineV2:
    """
    Gemini-powered calibrated multimodal coherence engine.

    Uses Gemini Embedding 2's unified 3072-d space for all three modalities.
    Computes cMSCI v2 with 5 variants (A-E) plus Matryoshka uncertainty.

    Usage:
        engine = CalibratedCoherenceEngineV2()
        result = engine.evaluate("text", "image.jpg", "audio.wav")
    """

    def __init__(
        self,
        calibration_path: Optional[str] = None,
        negative_bank_enabled: bool = True,
        image_index_path: str = "",
        audio_index_path: str = "",
    ):
        # Lazy import to avoid requiring google-genai at import time
        self._embedder = None

        # Calibration store
        self._calibration = None
        cal_path = calibration_path or str(GEMINI_CALIBRATION_PATH)
        if Path(cal_path).exists():
            from src.coherence.calibration import CalibrationStore
            self._calibration = CalibrationStore.load(cal_path)
            logger.info("v2 calibration loaded from %s", cal_path)

        # Negative bank
        self._negative_bank = None
        if negative_bank_enabled:
            try:
                self._negative_bank = NegativeBankV2(
                    image_index_path=image_index_path,
                    audio_index_path=audio_index_path,
                )
            except Exception as e:
                logger.warning("v2 negative bank disabled: %s", e)

        # Matryoshka uncertainty estimator
        self._matryoshka = None
        try:
            from src.embeddings.matryoshka_uncertainty import MatryoshkaUncertainty
            self._matryoshka = MatryoshkaUncertainty()
        except Exception as e:
            logger.warning("Matryoshka uncertainty disabled: %s", e)

        # Multi-scale Gramian fusion
        self._multiscale = None
        if CMSCI_V2_USE_MULTISCALE:
            try:
                from src.coherence.multiscale_gramian import MultiscaleGramian
                self._multiscale = MultiscaleGramian()
                logger.info("Multi-scale Gramian fusion enabled")
            except Exception as e:
                logger.warning("Multi-scale Gramian disabled: %s", e)

    def _get_embedder(self):
        """Lazy-initialize the Gemini embedder."""
        if self._embedder is None:
            from src.embeddings.gemini_embedder import GeminiEmbedder
            self._embedder = GeminiEmbedder.shared()
        return self._embedder

    def evaluate(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        domain: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate multimodal coherence with the full cMSCI v2 pipeline.

        Args:
            text: Text prompt.
            image_path: Path to image file.
            audio_path: Path to audio file.
            domain: Domain hint for negative bank.

        Returns:
            Dict with cmsci_v2, variant scores, intermediates, and uncertainty.
        """
        embedder = self._get_embedder()

        # ── Embed ──────────────────────────────────────────────
        emb_text = embedder.embed_text(text)
        emb_image = embedder.embed_image(image_path) if image_path else None
        emb_audio = embedder.embed_audio(audio_path) if audio_path else None

        # ── Raw Cosine Similarities (Variant A) ───────────────
        cos_ti = None
        cos_ta = None
        cos_ia = None

        if emb_text is not None and emb_image is not None:
            cos_ti = float(round(cosine_similarity(emb_text, emb_image), 4))
        if emb_text is not None and emb_audio is not None:
            cos_ta = float(round(cosine_similarity(emb_text, emb_audio), 4))
        if emb_image is not None and emb_audio is not None:
            cos_ia = float(round(cosine_similarity(emb_image, emb_audio), 4))

        available = {}
        if cos_ti is not None:
            available["ti"] = cos_ti
        if cos_ta is not None:
            available["ta"] = cos_ta
        if cos_ia is not None:
            available["ia"] = cos_ia

        variant_a = float(np.mean(list(available.values()))) if available else None

        # ── Gramian Volume (Variant B) ─────────────────────────
        gram_ti = None
        gram_ta = None
        gram_ia = None
        gram_tia = None

        if emb_text is not None and emb_image is not None:
            gram_ti = gram_volume_2d(emb_text, emb_image)
        if emb_text is not None and emb_audio is not None:
            gram_ta = gram_volume_2d(emb_text, emb_audio)
        if emb_image is not None and emb_audio is not None:
            gram_ia = gram_volume_2d(emb_image, emb_audio)
        if emb_text is not None and emb_image is not None and emb_audio is not None:
            gram_tia = gram_volume_3d(emb_text, emb_image, emb_audio)

        # 2-way coherence (average of pairwise gram coherences)
        gram_coherences_2d = []
        if gram_ti is not None:
            gram_coherences_2d.append(normalized_gram_coherence(gram_ti))
        if gram_ta is not None:
            gram_coherences_2d.append(normalized_gram_coherence(gram_ta))
        if gram_ia is not None:
            gram_coherences_2d.append(normalized_gram_coherence(gram_ia))

        gram_coherence_2d_avg = float(np.mean(gram_coherences_2d)) if gram_coherences_2d else None

        # 3D Gramian coherence (exact tri-modal)
        gram_coherence_3d = normalized_gram_coherence(gram_tia, n_vectors=3) if gram_tia is not None else None

        # Multi-scale Gramian fusion (if enabled)
        multiscale_result = None
        if self._multiscale is not None and emb_text is not None:
            try:
                multiscale_result = self._multiscale.compute(emb_text, emb_image, emb_audio)
                # Override single-scale gram coherences with fused versions
                if multiscale_result.get("fused_ti") is not None:
                    gram_coherences_2d = []
                    if multiscale_result["fused_ti"] is not None:
                        gram_coherences_2d.append(multiscale_result["fused_ti"])
                    if multiscale_result["fused_ta"] is not None:
                        gram_coherences_2d.append(multiscale_result["fused_ta"])
                    if multiscale_result["fused_ia"] is not None:
                        gram_coherences_2d.append(multiscale_result["fused_ia"])
                    gram_coherence_2d_avg = float(np.mean(gram_coherences_2d)) if gram_coherences_2d else None
                    if multiscale_result["fused_tia"] is not None:
                        gram_coherence_3d = multiscale_result["fused_tia"]
            except Exception as e:
                logger.warning("Multi-scale computation failed: %s", e)

        # Select B score based on config
        cal_mode = CMSCI_V2_CAL_MODE
        if cal_mode == "gram_3d" and gram_coherence_3d is not None:
            variant_b = gram_coherence_3d
        else:
            variant_b = gram_coherence_2d_avg

        # ── Z-Score Normalization (Variant C) ──────────────────
        w_ti = CMSCI_V2_W_TI
        z_gram_ti = None
        z_gram_ta = None
        z_gram_ia = None
        z_gram_tia = None
        variant_c = variant_b  # default to B if no calibration

        if self._calibration is not None:
            if gram_ti is not None:
                coh_ti = normalized_gram_coherence(gram_ti)
                z_gram_ti = self._calibration.normalize("gram_coh_ti_gemini", coh_ti)
            if gram_ta is not None:
                coh_ta = normalized_gram_coherence(gram_ta)
                z_gram_ta = self._calibration.normalize("gram_coh_ta_gemini", coh_ta)
            if gram_ia is not None:
                coh_ia = normalized_gram_coherence(gram_ia)
                z_gram_ia = self._calibration.normalize("gram_coh_ia_gemini", coh_ia)
            if gram_tia is not None:
                coh_tia = normalized_gram_coherence(gram_tia, n_vectors=3)
                z_gram_tia = self._calibration.normalize("gram_coh_tia_gemini", coh_tia)

            # Compute z_mean based on calibration mode
            if cal_mode == "gram_3d" and z_gram_tia is not None:
                z_mean = z_gram_tia
            else:
                # Weighted average of 2D z-scores
                z_parts = []
                z_weights = []
                if z_gram_ti is not None:
                    z_parts.append(z_gram_ti)
                    z_weights.append(w_ti)
                if z_gram_ta is not None:
                    z_parts.append(z_gram_ta)
                    z_weights.append(1.0 - w_ti)
                if z_gram_ia is not None and CMSCI_V2_W_IA > 0:
                    z_parts.append(z_gram_ia)
                    z_weights.append(CMSCI_V2_W_IA)

                if z_parts:
                    total_w = sum(z_weights)
                    z_mean = sum(z * wt for z, wt in zip(z_parts, z_weights)) / total_w
                else:
                    z_mean = None

            if z_mean is not None:
                variant_c = float(1.0 / (1.0 + np.exp(-z_mean)))

        # ── Contrastive Margin (Variant D) ─────────────────────
        contrastive_result = None
        variant_d = variant_c
        margin_alpha = CMSCI_V2_ALPHA
        margin = 0.0

        if self._negative_bank is not None:
            contrastive_result = self._negative_bank.compute_contrastive_margin(
                gram_ti=gram_ti,
                gram_ta=gram_ta,
                gram_ia=gram_ia,
                text_emb=emb_text,
                image_emb=emb_image,
                audio_emb=emb_audio,
                domain=domain,
                k=CMSCI_V2_NEGATIVE_K,
            )

            if contrastive_result["n_negatives"] > 0:
                # Weight margins by channel weights
                margin_parts = []
                margin_weights = []
                if contrastive_result["n_negatives_ti"] > 0:
                    margin_parts.append(contrastive_result["margin_ti"])
                    margin_weights.append(w_ti)
                if contrastive_result["n_negatives_ta"] > 0:
                    margin_parts.append(contrastive_result["margin_ta"])
                    margin_weights.append(1.0 - w_ti)
                if contrastive_result["n_negatives_ia"] > 0 and CMSCI_V2_W_IA > 0:
                    margin_parts.append(contrastive_result["margin_ia"])
                    margin_weights.append(CMSCI_V2_W_IA)

                if margin_parts:
                    total_mw = sum(margin_weights)
                    margin = sum(m * mw for m, mw in zip(margin_parts, margin_weights)) / max(total_mw, 1e-6)
                    contrastive_result["margin"] = float(margin)

                # Recompute with margin
                if cal_mode == "gram_3d" and z_gram_tia is not None:
                    z_d = z_gram_tia
                else:
                    z_parts_d = []
                    z_weights_d = []
                    if z_gram_ti is not None:
                        z_parts_d.append(z_gram_ti)
                        z_weights_d.append(w_ti)
                    if z_gram_ta is not None:
                        z_parts_d.append(z_gram_ta)
                        z_weights_d.append(1.0 - w_ti)
                    if z_gram_ia is not None and CMSCI_V2_W_IA > 0:
                        z_parts_d.append(z_gram_ia)
                        z_weights_d.append(CMSCI_V2_W_IA)
                    if z_parts_d:
                        total_wd = sum(z_weights_d)
                        z_d = sum(z * wt for z, wt in zip(z_parts_d, z_weights_d)) / total_wd
                    else:
                        z_d = None

                if z_d is not None:
                    variant_d = float(1.0 / (1.0 + np.exp(-(z_d + margin_alpha * margin))))

        # ── Complementarity + Matryoshka (Variant E) ───────────
        # In v2, complementarity = exact 3D Gramian (no ExMCR approximation)
        # Matryoshka scale consistency replaces ProbVLM adaptive weighting
        variant_e = variant_d
        w_compl = CMSCI_V2_W_COMPL
        gamma_mrl = CMSCI_V2_GAMMA_MRL
        z_compl = None
        matryoshka_result = None
        adaptive_w_ti = None

        # Complementarity: how much does adding the image-audio cross-channel help?
        # z_compl = z-normalized (1 - gram_3d_volume) — the 3D coherence tells us
        # whether all three modalities agree, which 2D channels can't capture.
        if gram_tia is not None and self._calibration is not None:
            coh_tia = normalized_gram_coherence(gram_tia, n_vectors=3)
            z_compl_raw = self._calibration.normalize("gram_coh_tia_gemini", coh_tia)
            z_compl = z_compl_raw  # positive = more coherent tri-modal alignment

        # Matryoshka adaptive weighting
        mrl_raw_w_ti = None  # raw Matryoshka weight before gamma mixing
        if self._matryoshka is not None and emb_text is not None:
            try:
                matryoshka_result = self._matryoshka.compute_scale_consistency(
                    emb_text, emb_image, emb_audio,
                )
                adaptive = self._matryoshka.compute_adaptive_weights(
                    emb_text, emb_image, emb_audio,
                )
                mrl_raw_w_ti = adaptive["w_ti"]  # always store raw for optimizer
                if gamma_mrl > 0:
                    adaptive_w_ti = float(
                        (1.0 - gamma_mrl) * w_ti + gamma_mrl * mrl_raw_w_ti
                    )
            except Exception as e:
                logger.warning("Matryoshka computation failed: %s", e)

        # Compose E logit
        # Reconstruct base z_2d (same as D's pre-margin component)
        z_2d_base = None
        w_for_z = adaptive_w_ti if adaptive_w_ti is not None else w_ti

        if cal_mode == "gram_3d" and z_gram_tia is not None:
            z_2d_base = z_gram_tia
        else:
            z_parts_e = []
            z_weights_e = []
            if z_gram_ti is not None:
                z_parts_e.append(z_gram_ti)
                z_weights_e.append(w_for_z)
            if z_gram_ta is not None:
                z_parts_e.append(z_gram_ta)
                z_weights_e.append(1.0 - w_for_z)
            if z_gram_ia is not None and CMSCI_V2_W_IA > 0:
                z_parts_e.append(z_gram_ia)
                z_weights_e.append(CMSCI_V2_W_IA)
            if z_parts_e:
                total_we = sum(z_weights_e)
                z_2d_base = sum(z * wt for z, wt in zip(z_parts_e, z_weights_e)) / total_we

        if z_2d_base is not None:
            logit_e = z_2d_base + margin_alpha * margin
            if z_compl is not None and w_compl > 0:
                logit_e += w_compl * z_compl
            variant_e = float(1.0 / (1.0 + np.exp(-logit_e)))

        # ── Assemble cMSCI v2 ──────────────────────────────────
        cmsci_v2 = variant_e
        active_variant = "E"

        if variant_e == variant_d:
            active_variant = "D"
        if variant_d == variant_c:
            active_variant = "C"
        if variant_c == variant_b:
            active_variant = "B" if variant_b is not None else "A"

        if cmsci_v2 is None:
            cmsci_v2 = variant_a
            active_variant = "A"

        logger.info(
            "cMSCI_v2 = %.4f (variant %s) | cosine_avg = %s",
            cmsci_v2 if cmsci_v2 is not None else 0.0,
            active_variant,
            variant_a,
        )

        return {
            "cmsci_v2": round(cmsci_v2, 4) if cmsci_v2 is not None else None,
            "active_variant": active_variant,
            "cosine": {
                "ti": cos_ti,
                "ta": cos_ta,
                "ia": cos_ia,
                "mean": round(variant_a, 4) if variant_a is not None else None,
            },
            "gram": {
                "ti": round(gram_ti, 4) if gram_ti is not None else None,
                "ta": round(gram_ta, 4) if gram_ta is not None else None,
                "ia": round(gram_ia, 4) if gram_ia is not None else None,
                "tia": round(gram_tia, 4) if gram_tia is not None else None,
                "coherence_2d_avg": round(gram_coherence_2d_avg, 4) if gram_coherence_2d_avg is not None else None,
                "coherence_3d": round(gram_coherence_3d, 4) if gram_coherence_3d is not None else None,
            },
            "calibration": {
                "z_gram_ti": round(z_gram_ti, 4) if z_gram_ti is not None else None,
                "z_gram_ta": round(z_gram_ta, 4) if z_gram_ta is not None else None,
                "z_gram_ia": round(z_gram_ia, 4) if z_gram_ia is not None else None,
                "z_gram_tia": round(z_gram_tia, 4) if z_gram_tia is not None else None,
                "z_compl": round(z_compl, 4) if z_compl is not None else None,
                "cal_mode": cal_mode if self._calibration is not None else None,
                "w_ti": w_ti,
                "w_compl": w_compl,
                "gamma_mrl": gamma_mrl,
                "margin_alpha": margin_alpha,
                "adaptive_w_ti": round(adaptive_w_ti, 4) if adaptive_w_ti is not None else None,
                "mrl_raw_w_ti": round(mrl_raw_w_ti, 4) if mrl_raw_w_ti is not None else None,
            },
            "contrastive": contrastive_result,
            "matryoshka": {
                "consistency": round(matryoshka_result["consistency"], 4) if matryoshka_result else None,
                "coherences": matryoshka_result["coherences"] if matryoshka_result else None,
                "std": round(matryoshka_result["std"], 4) if matryoshka_result else None,
            } if matryoshka_result else None,
            "variant_scores": {
                "A_cosine_avg": round(variant_a, 4) if variant_a is not None else None,
                "B_gram": round(variant_b, 4) if variant_b is not None else None,
                "C_gram_znorm": round(variant_c, 4) if variant_c is not None else None,
                "D_gram_znorm_contrastive": round(variant_d, 4) if variant_d is not None else None,
                "E_full_cmsci_v2": round(variant_e, 4) if variant_e is not None else None,
            },
            "multiscale": {
                "fused_ti": round(multiscale_result["fused_ti"], 4) if multiscale_result and multiscale_result.get("fused_ti") is not None else None,
                "fused_ta": round(multiscale_result["fused_ta"], 4) if multiscale_result and multiscale_result.get("fused_ta") is not None else None,
                "fused_ia": round(multiscale_result["fused_ia"], 4) if multiscale_result and multiscale_result.get("fused_ia") is not None else None,
                "fused_tia": round(multiscale_result["fused_tia"], 4) if multiscale_result and multiscale_result.get("fused_tia") is not None else None,
                "scale_stds": multiscale_result["scale_stds"] if multiscale_result else None,
            } if multiscale_result else None,
        }

    def evaluate_batch(
        self,
        items: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of (text, image_path, audio_path) triples."""
        results = []
        for item in items:
            result = self.evaluate(
                text=item.get("text", ""),
                image_path=item.get("image_path"),
                audio_path=item.get("audio_path"),
                domain=item.get("domain", ""),
            )
            results.append(result)
        return results
