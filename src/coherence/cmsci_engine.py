"""
Calibrated Multimodal Semantic Coherence Index (cMSCI) Engine.

Replaces fixed weighted averaging (MSCI) with a principled pipeline:
    1. Gramian Volume: geometric coherence of embedding vectors
    2. Distribution Normalization: z-score calibration per channel
    3. Contrastive Margin: comparison against hard negatives
    4. Cross-Space Alignment: Ex-MCR projects CLAP→CLIP for 3-way GRAM
    5. Probabilistic Uncertainty: MC sampling for confidence intervals

The CalibratedCoherenceEngine runs alongside CoherenceEngine (not replacing
it) and returns both legacy MSCI and new cMSCI scores for comparison.

Variant progression:
    A: MSCI (baseline, weighted cosine average)
    B: GRAM-only (geometric, no calibration)
    C: GRAM + z-norm (normalized geometric)
    D: GRAM + z-norm + contrastive (calibrated geometric)
    E: GRAM + z-norm + contrastive + Ex-MCR (3-way calibrated)
    F: Full cMSCI (probabilistic + calibrated + 3-way)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.coherence.gram_volume import (
    gram_volume_2d,
    gram_volume_3d,
    gram_volume_nd,
    normalized_gram_coherence,
)
from src.config.settings import (
    CMSCI_MARGIN_ALPHA,
    CMSCI_CHANNEL_WEIGHT_TI,
    CMSCI_CALIBRATION_MODE,
    CMSCI_W_3D,
    CMSCI_GAMMA,
    AUDIO_FILTER_SILENCE,
    AUDIO_SILENCE_RMS_DB,
    AUDIO_SILENCE_FRACTION_THRESHOLD,
    AUDIO_LOG_QUALITY_WARNINGS,
)
from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class CalibratedCoherenceEngine:
    """
    Uncertainty-aware, geometrically-grounded tri-modal coherence engine.

    Computes cMSCI alongside legacy MSCI for comparison.

    Usage:
        engine = CalibratedCoherenceEngine()
        result = engine.evaluate("A beach at sunset", "beach.jpg", "waves.wav")
        print(result["cmsci"])       # Calibrated score
        print(result["msci"])        # Legacy score (for comparison)
        print(result["variant_scores"])  # Scores for each variant A-F
    """

    def __init__(
        self,
        target_dim: int = 512,
        calibration_path: Optional[str] = None,
        exmcr_weights_path: Optional[str] = None,
        bridge_path: Optional[str] = None,
        prob_clip_adapter_path: Optional[str] = None,
        prob_clap_adapter_path: Optional[str] = None,
        negative_bank_enabled: bool = True,
    ):
        self.embedder = AlignedEmbedder(target_dim=target_dim)

        # Calibration store (Phase 2)
        self._calibration = None
        if calibration_path and Path(calibration_path).exists():
            from src.coherence.calibration import CalibrationStore
            self._calibration = CalibrationStore.load(calibration_path)
            logger.info("Calibration loaded from %s", calibration_path)

        # Negative bank (Phase 2)
        self._negative_bank = None
        if negative_bank_enabled:
            try:
                from src.coherence.negative_bank import NegativeBank
                self._negative_bank = NegativeBank()
            except Exception as e:
                logger.warning("Negative bank disabled: %s", e)

        # Ex-MCR projector (Phase 3 — projects CLAP into CLIP space)
        self._exmcr = None
        if exmcr_weights_path:
            from src.embeddings.space_alignment import ExMCRProjector
            self._exmcr = ExMCRProjector(weights_path=exmcr_weights_path)
            if self._exmcr.is_identity:
                logger.info("Ex-MCR in identity mode (no weights)")
            else:
                logger.info("Ex-MCR projector active")

        # Cross-Space Bridge (projects CLIP image + CLAP audio → shared 256-d)
        self._bridge = None
        if bridge_path and Path(bridge_path).exists():
            from src.embeddings.cross_space_bridge import CrossSpaceBridge
            self._bridge = CrossSpaceBridge.load(bridge_path)
            logger.info("CrossSpaceBridge loaded from %s", bridge_path)

        # Audio quality analyzer (silence detection)
        self._audio_analyzer = None
        if AUDIO_FILTER_SILENCE:
            from src.embeddings.audio_analysis import AudioAnalyzer
            self._audio_analyzer = AudioAnalyzer(
                silence_rms_db=AUDIO_SILENCE_RMS_DB,
                silence_fraction_threshold=AUDIO_SILENCE_FRACTION_THRESHOLD,
            )
            logger.info("Audio silence detection enabled (threshold=%.0f dB)", AUDIO_SILENCE_RMS_DB)

        # Probabilistic adapters (Phase 4)
        self._prob_clip = None
        self._prob_clap = None
        if prob_clip_adapter_path and Path(prob_clip_adapter_path).exists():
            from src.embeddings.probabilistic_adapter import ProbabilisticAdapter
            self._prob_clip = ProbabilisticAdapter.load(prob_clip_adapter_path)
            logger.info("CLIP probabilistic adapter loaded")
        if prob_clap_adapter_path and Path(prob_clap_adapter_path).exists():
            from src.embeddings.probabilistic_adapter import ProbabilisticAdapter
            self._prob_clap = ProbabilisticAdapter.load(prob_clap_adapter_path)
            logger.info("CLAP probabilistic adapter loaded")

    def evaluate(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        domain: str = "",
        n_mc_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate multimodal coherence with full cMSCI pipeline.

        Returns both legacy MSCI and cMSCI scores along with all
        intermediate computations for ablation analysis.

        Args:
            text: Text prompt.
            image_path: Path to image file.
            audio_path: Path to audio file.
            domain: Domain hint for negative bank (e.g., "nature").
            n_mc_samples: Number of MC samples for uncertainty.

        Returns:
            Dict with keys:
                msci: Legacy MSCI score (weighted cosine average)
                cmsci: Calibrated cMSCI score
                scores: Raw pairwise scores (st_i, st_a, si_a)
                gram: Gramian volume scores
                calibration: Z-normalized scores
                contrastive: Contrastive margin results
                uncertainty: MC sampling uncertainty (if adapters loaded)
                variant_scores: Scores for each variant A-F
        """
        # ── Audio Quality Check ────────────────────────────────
        audio_quality = None
        if audio_path and self._audio_analyzer is not None:
            try:
                from pathlib import Path as _Path
                if _Path(audio_path).exists():
                    report = self._audio_analyzer.analyze(audio_path)
                    audio_quality = report.to_dict()
                    if AUDIO_LOG_QUALITY_WARNINGS and report.issues:
                        for issue in report.issues:
                            logger.warning("Audio quality [%s]: %s", audio_path, issue)
            except Exception as e:
                logger.warning("Audio analysis failed for %s: %s", audio_path, e)

        # ── Embed ──────────────────────────────────────────────
        emb_text_clip = self.embedder.embed_text(text)
        emb_text_clap = self.embedder.embed_text_for_audio(text) if audio_path else None
        emb_image = self.embedder.embed_image(image_path) if image_path else None
        emb_audio = self.embedder.embed_audio(audio_path) if audio_path else None

        # ── Legacy MSCI (Variant A) ────────────────────────────
        st_i = None
        st_a = None
        si_a = None

        if emb_text_clip is not None and emb_image is not None:
            st_i = float(round(cosine_similarity(emb_text_clip, emb_image), 4))
        if emb_text_clap is not None and emb_audio is not None:
            st_a = float(round(cosine_similarity(emb_text_clap, emb_audio), 4))

        available = {}
        if st_i is not None:
            available["st_i"] = st_i
        if st_a is not None:
            available["st_a"] = st_a

        weights = {"st_i": 0.45, "st_a": 0.45, "si_a": 0.10}
        if len(available) >= 2:
            total_w = sum(weights[k] for k in available if k in weights)
            msci = sum(available[k] * weights[k] for k in available if k in weights) / max(total_w, 1e-6)
        elif len(available) == 1:
            msci = list(available.values())[0]
        else:
            msci = None

        variant_a = msci

        # ── Gramian Volume (Variant B) ─────────────────────────
        gram_ti = None
        gram_ta = None
        gram_tia = None
        gram_coherence_2way = None

        if emb_text_clip is not None and emb_image is not None:
            gram_ti = gram_volume_2d(emb_text_clip, emb_image)

        if emb_text_clap is not None and emb_audio is not None:
            gram_ta = gram_volume_2d(emb_text_clap, emb_audio)

        # 2-way GRAM coherence (average of text-image and text-audio gram coherences)
        gram_coherences = []
        if gram_ti is not None:
            gram_coherences.append(normalized_gram_coherence(gram_ti))
        if gram_ta is not None:
            gram_coherences.append(normalized_gram_coherence(gram_ta))

        if gram_coherences:
            gram_coherence_2way = float(np.mean(gram_coherences))

        variant_b = gram_coherence_2way

        # ── Z-Score Normalization (Variant C) ──────────────────
        z_st_i = None
        z_st_a = None
        z_gram_ti = None
        z_gram_ta = None
        variant_c = variant_b  # default to B if no calibration

        # Channel weight from settings (optimized via LOO-CV)
        w_ti = CMSCI_CHANNEL_WEIGHT_TI
        cal_mode = CMSCI_CALIBRATION_MODE

        if self._calibration is not None:
            if st_i is not None:
                z_st_i = self._calibration.normalize("st_i", st_i)
            if st_a is not None:
                z_st_a = self._calibration.normalize("st_a", st_a)

            # GRAM coherence z-scores (for gram calibration mode)
            if gram_ti is not None:
                gram_coh_ti = normalized_gram_coherence(gram_ti)
                z_gram_ti = self._calibration.normalize("gram_coh_ti", gram_coh_ti)
            if gram_ta is not None:
                gram_coh_ta = normalized_gram_coherence(gram_ta)
                z_gram_ta = self._calibration.normalize("gram_coh_ta", gram_coh_ta)

            # Select calibration mode: cosine z-scores or gram coherence z-scores
            if cal_mode == "gram" and z_gram_ti is not None and z_gram_ta is not None:
                z_mean = w_ti * z_gram_ti + (1.0 - w_ti) * z_gram_ta
            else:
                # Cosine mode (original behavior) with weighted channels
                z_coherences = []
                z_weights = []
                if z_st_i is not None:
                    z_coherences.append(z_st_i)
                    z_weights.append(w_ti)
                if z_st_a is not None:
                    z_coherences.append(z_st_a)
                    z_weights.append(1.0 - w_ti)

                if z_coherences:
                    total_w = sum(z_weights)
                    z_mean = sum(z * wt for z, wt in zip(z_coherences, z_weights)) / total_w
                else:
                    z_mean = None

            if z_mean is not None:
                # Map z-scores back to [0,1] via sigmoid for interpretability
                variant_c = float(1.0 / (1.0 + np.exp(-z_mean)))

        # ── Contrastive Margin (Variant D) ─────────────────────
        # Per-channel margins: m_ti within CLIP space, m_ta within CLAP space.
        # Combined: m = w_ti * m_ti + (1-w_ti) * m_ta (normalized by available channels).
        contrastive_result = None
        variant_d = variant_c  # default to C if no negatives
        margin_alpha = CMSCI_MARGIN_ALPHA

        if self._negative_bank is not None and gram_coherence_2way is not None:
            contrastive_result = self._negative_bank.compute_contrastive_margin(
                gram_ti=gram_ti,
                gram_ta=gram_ta,
                text_clip_emb=emb_text_clip,
                image_emb=emb_image,
                text_clap_emb=emb_text_clap,
                audio_emb=emb_audio,
                domain=domain,
                k=5,
            )

            if contrastive_result["n_negatives"] > 0:
                # Combine per-channel margins with channel weights
                # Only include channels that found negatives
                margin_parts = []
                margin_weights = []
                if contrastive_result["n_negatives_ti"] > 0:
                    margin_parts.append(contrastive_result["margin_ti"])
                    margin_weights.append(w_ti)
                if contrastive_result["n_negatives_ta"] > 0:
                    margin_parts.append(contrastive_result["margin_ta"])
                    margin_weights.append(1.0 - w_ti)

                total_mw = sum(margin_weights)
                margin = sum(m * mw for m, mw in zip(margin_parts, margin_weights)) / max(total_mw, 1e-6)
                # Store combined margin for downstream (optimizer, Variant E/F)
                contrastive_result["margin"] = float(margin)

                # Use the same calibration mode and weighting as Variant C
                if cal_mode == "gram" and z_gram_ti is not None and z_gram_ta is not None:
                    z_mean_d = w_ti * z_gram_ti + (1.0 - w_ti) * z_gram_ta
                else:
                    z_coherences_d = []
                    z_weights_d = []
                    if z_st_i is not None:
                        z_coherences_d.append(z_st_i)
                        z_weights_d.append(w_ti)
                    elif st_i is not None:
                        z_coherences_d.append(st_i)
                        z_weights_d.append(w_ti)
                    if z_st_a is not None:
                        z_coherences_d.append(z_st_a)
                        z_weights_d.append(1.0 - w_ti)
                    elif st_a is not None:
                        z_coherences_d.append(st_a)
                        z_weights_d.append(1.0 - w_ti)

                    if z_coherences_d:
                        total_wd = sum(z_weights_d)
                        z_mean_d = sum(z * wt for z, wt in zip(z_coherences_d, z_weights_d)) / total_wd
                    else:
                        z_mean_d = None

                if z_mean_d is not None:
                    variant_d = float(1.0 / (1.0 + np.exp(-(z_mean_d + margin_alpha * margin))))
                else:
                    variant_d = variant_c

        # ── Cross-Space Complementarity — Variant E ──────────
        # COMPLEMENTARITY: E = sigmoid(z_2d + w_3d * z_compl + alpha * margin)
        # ExMCR projects CLAP audio → CLIP space, enabling measurement of
        # image-audio complementarity (Gramian dispersion in unified space).
        # High complementarity = image and audio contribute unique perspectives.
        # Low complementarity = redundant cross-modal information.
        # z_compl = z_normalize(gram_volume_ia) — positive z = more complementary.
        # w_3d=0 recovers D exactly (safety guarantee).
        audio_projected = None
        variant_e = variant_d  # default to D if no projector
        z_compl = None  # z-normalized complementarity (exported for optimizer)
        gram_ia_volume = None  # raw image-audio Gramian volume
        w_3d = CMSCI_W_3D

        # Reconstruct D's pre-margin z-score (z_2d) for composition
        z_2d = None
        margin = 0.0
        if contrastive_result is not None and contrastive_result["n_negatives"] > 0:
            margin = contrastive_result["margin"]
        if cal_mode == "gram" and z_gram_ti is not None and z_gram_ta is not None:
            z_2d = w_ti * z_gram_ti + (1.0 - w_ti) * z_gram_ta
        elif z_st_i is not None and z_st_a is not None:
            z_2d = w_ti * z_st_i + (1.0 - w_ti) * z_st_a

        # Project audio into CLIP space via ExMCR and compute complementarity
        if self._exmcr is not None and not self._exmcr.is_identity:
            if emb_audio is not None:
                audio_projected = self._exmcr.project_audio(emb_audio)
                if emb_image is not None:
                    si_a = float(round(cosine_similarity(emb_image, audio_projected), 4))
                    # Image-audio Gramian volume = dispersion = complementarity
                    gram_ia_volume = gram_volume_2d(emb_image, audio_projected)
                if emb_text_clip is not None and emb_image is not None and audio_projected is not None:
                    gram_tia = gram_volume_3d(emb_text_clip, emb_image, audio_projected)

        # Z-normalize complementarity (volume, NOT coherence)
        # z_compl = -z_gram_ia_coherence (flipped: high volume = high complementarity)
        if gram_ia_volume is not None and self._calibration is not None:
            gram_ia_coherence = normalized_gram_coherence(gram_ia_volume)
            z_gram_ia_coh = self._calibration.normalize("gram_coh_ia_exmcr", gram_ia_coherence)
            z_compl = -z_gram_ia_coh  # flip: positive = more complementary

        # Compose: E = sigmoid(z_2d + w_3d * z_compl + alpha * margin)
        if z_2d is not None:
            logit_e = z_2d + margin_alpha * margin
            if z_compl is not None:
                logit_e += w_3d * z_compl
            variant_e = float(1.0 / (1.0 + np.exp(-logit_e)))

        # ── Probabilistic Adaptive Weighting (Variant F) ──────
        # ProbVLM drives per-sample channel weights instead of fixed w_ti.
        # adaptive_w = (1/u_ti) / (1/u_ti + 1/u_ta)  — trust more confident channel
        # w_ti_final = (1 - gamma) * base_w + gamma * adaptive_w
        # gamma=0 → w_ti_final = base_w → recovers E exactly (safety guarantee)
        # MC sampling remains metadata only (confidence intervals, not scoring).
        uncertainty_result = None
        variant_f = variant_e  # default to E
        u_ti = None  # per-channel uncertainty (exported for optimizer)
        u_ta = None
        adaptive_w_ti = None
        gamma = CMSCI_GAMMA

        if self._prob_clip is not None or self._prob_clap is not None:
            mc_volumes = []

            # Per-channel uncertainty from ProbVLM adapters
            if self._prob_clip is not None and emb_text_clip is not None and emb_image is not None:
                u_text_clip = self._prob_clip.uncertainty(emb_text_clip)
                u_image_clip = self._prob_clip.uncertainty(emb_image)
                u_ti = float(np.mean([u_text_clip, u_image_clip]))

                # MC samples for confidence interval metadata
                text_samples = self._prob_clip.sample(emb_text_clip, n_mc_samples)
                image_samples = self._prob_clip.sample(emb_image, n_mc_samples)
                for t_s, i_s in zip(text_samples, image_samples):
                    mc_volumes.append(gram_volume_2d(t_s, i_s))

            if self._prob_clap is not None and emb_text_clap is not None and emb_audio is not None:
                u_text_clap = self._prob_clap.uncertainty(emb_text_clap)
                u_audio_clap = self._prob_clap.uncertainty(emb_audio)
                u_ta = float(np.mean([u_text_clap, u_audio_clap]))

                text_samples = self._prob_clap.sample(emb_text_clap, n_mc_samples)
                audio_samples = self._prob_clap.sample(emb_audio, n_mc_samples)
                for t_s, a_s in zip(text_samples, audio_samples):
                    mc_volumes.append(gram_volume_2d(t_s, a_s))

            # Compute adaptive channel weight from uncertainty
            if u_ti is not None and u_ta is not None and u_ti > 0 and u_ta > 0 and gamma > 0:
                inv_ti = 1.0 / u_ti
                inv_ta = 1.0 / u_ta
                adaptive_w = inv_ti / (inv_ti + inv_ta)
                w_ti_final = (1.0 - gamma) * w_ti + gamma * adaptive_w
                adaptive_w_ti = float(w_ti_final)

                # Recompute z_2d with adaptive weights
                if cal_mode == "gram" and z_gram_ti is not None and z_gram_ta is not None:
                    z_2d_adaptive = w_ti_final * z_gram_ti + (1.0 - w_ti_final) * z_gram_ta
                elif z_st_i is not None and z_st_a is not None:
                    z_2d_adaptive = w_ti_final * z_st_i + (1.0 - w_ti_final) * z_st_a
                else:
                    z_2d_adaptive = None

                if z_2d_adaptive is not None:
                    logit_f = z_2d_adaptive + margin_alpha * margin
                    if z_compl is not None:
                        logit_f += w_3d * z_compl
                    variant_f = float(1.0 / (1.0 + np.exp(-logit_f)))

            # MC sampling for confidence intervals (metadata, NOT scoring)
            if mc_volumes:
                mc_coherences = [normalized_gram_coherence(v) for v in mc_volumes]
                mc_mean = float(np.mean(mc_coherences))
                mc_std = float(np.std(mc_coherences))
                mc_ci_lower = float(np.percentile(mc_coherences, 2.5))
                mc_ci_upper = float(np.percentile(mc_coherences, 97.5))
            else:
                mc_mean = mc_std = mc_ci_lower = mc_ci_upper = None

            uncertainty_result = {
                "mc_mean": round(mc_mean, 4) if mc_mean is not None else None,
                "mc_std": round(mc_std, 4) if mc_std is not None else None,
                "mc_ci_lower": round(mc_ci_lower, 4) if mc_ci_lower is not None else None,
                "mc_ci_upper": round(mc_ci_upper, 4) if mc_ci_upper is not None else None,
                "u_ti": round(u_ti, 6) if u_ti is not None else None,
                "u_ta": round(u_ta, 6) if u_ta is not None else None,
                "adaptive_w_ti": round(adaptive_w_ti, 4) if adaptive_w_ti is not None else None,
                "gamma": gamma,
                "n_samples": n_mc_samples,
            }

        # ── Assemble cMSCI ─────────────────────────────────────
        # cMSCI is the highest available variant
        cmsci = variant_f
        active_variant = "F"

        if variant_f == variant_e:
            active_variant = "E" if variant_e != variant_d else "D"
        if variant_e == variant_d:
            active_variant = "D" if variant_d != variant_c else "C"
        if variant_d == variant_c:
            active_variant = "C" if variant_c != variant_b else "B"
        if variant_c == variant_b:
            active_variant = "B" if variant_b is not None else "A"

        # Final cMSCI: use the most sophisticated available variant
        if cmsci is None:
            cmsci = msci  # fallback to legacy
            active_variant = "A"

        logger.info(
            "cMSCI = %.4f (variant %s) | MSCI = %s",
            cmsci if cmsci is not None else 0.0,
            active_variant,
            msci,
        )

        return {
            "cmsci": round(cmsci, 4) if cmsci is not None else None,
            "msci": round(msci, 4) if msci is not None else None,
            "active_variant": active_variant,
            "scores": {
                "st_i": st_i,
                "st_a": st_a,
                "si_a": si_a,
            },
            "gram": {
                "text_image": round(gram_ti, 4) if gram_ti is not None else None,
                "text_audio": round(gram_ta, 4) if gram_ta is not None else None,
                "text_image_audio": round(gram_tia, 4) if gram_tia is not None else None,
                "coherence_2way": round(gram_coherence_2way, 4) if gram_coherence_2way is not None else None,
            },
            "calibration": {
                "z_st_i": round(z_st_i, 4) if z_st_i is not None else None,
                "z_st_a": round(z_st_a, 4) if z_st_a is not None else None,
                "z_gram_ti": round(z_gram_ti, 4) if z_gram_ti is not None else None,
                "z_gram_ta": round(z_gram_ta, 4) if z_gram_ta is not None else None,
                "z_compl": round(z_compl, 4) if z_compl is not None else None,
                "gram_ia_volume": round(gram_ia_volume, 4) if gram_ia_volume is not None else None,
                "u_ti": round(u_ti, 6) if u_ti is not None else None,
                "u_ta": round(u_ta, 6) if u_ta is not None else None,
                "adaptive_w_ti": round(adaptive_w_ti, 4) if adaptive_w_ti is not None else None,
                "cal_mode": cal_mode if self._calibration is not None else None,
                "w_ti": w_ti,
                "w_3d": w_3d,
                "gamma": gamma,
                "margin_alpha": CMSCI_MARGIN_ALPHA if contrastive_result else None,
            },
            "contrastive": contrastive_result,
            "uncertainty": uncertainty_result,
            "audio_quality": audio_quality,
            "variant_scores": {
                "A_msci": round(variant_a, 4) if variant_a is not None else None,
                "B_gram": round(variant_b, 4) if variant_b is not None else None,
                "C_gram_znorm": round(variant_c, 4) if variant_c is not None else None,
                "D_gram_znorm_contrastive": round(variant_d, 4) if variant_d is not None else None,
                "E_gram_znorm_contrastive_exmcr": round(variant_e, 4) if variant_e is not None else None,
                "F_full_cmsci": round(variant_f, 4) if variant_f is not None else None,
            },
        }

    def evaluate_batch(
        self,
        items: List[Dict[str, str]],
        n_mc_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of (text, image_path, audio_path) triples.

        Args:
            items: List of dicts with keys "text", "image_path", "audio_path", "domain".
            n_mc_samples: MC samples per item.

        Returns:
            List of result dicts from evaluate().
        """
        results = []
        for item in items:
            result = self.evaluate(
                text=item.get("text", ""),
                image_path=item.get("image_path"),
                audio_path=item.get("audio_path"),
                domain=item.get("domain", ""),
                n_mc_samples=n_mc_samples,
            )
            results.append(result)
        return results
