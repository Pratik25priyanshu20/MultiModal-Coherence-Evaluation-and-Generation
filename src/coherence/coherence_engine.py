from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.coherence.drift_detector import detect_drift
from src.coherence.scorer import CoherenceScorer
from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class CoherenceEngine:
    """
    Evaluates multimodal coherence using correct embedding spaces.

    Text-Image similarity: CLIP shared space
    Text-Audio similarity: CLAP shared space
    Image-Audio similarity: cross-space (CLIP vs CLAP) — not directly comparable.
        We omit si_a from scoring by default since it would compare embeddings
        from different model spaces without a trained bridge.

    If a trained CrossSpaceBridge is loaded via load_bridge(), si_a is computed
    in the learned bridge space and the full MSCI formula activates:
        MSCI = 0.45 * st_i + 0.45 * st_a + 0.10 * si_a
    """

    def __init__(self, target_dim: int = 512):
        self.embedder = AlignedEmbedder(target_dim=target_dim)
        self.scorer = CoherenceScorer()
        self._bridge = None  # Optional CrossSpaceBridge for si_a

    def load_bridge(self, path: str) -> None:
        """
        Load a trained CrossSpaceBridge to enable image-audio similarity.

        Once loaded, si_a will be computed via the bridge's shared space
        instead of being set to None.

        Args:
            path: Path to saved bridge weights (.pt file)
        """
        from src.embeddings.cross_space_bridge import CrossSpaceBridge
        from pathlib import Path

        bridge_path = Path(path)
        if not bridge_path.exists():
            logger.warning("Bridge file not found: %s — si_a remains disabled", path)
            return

        self._bridge = CrossSpaceBridge.load(bridge_path)
        logger.info("Cross-space bridge loaded — si_a enabled")

    def evaluate(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # CLIP text embedding (for text-image comparison)
        emb_text_clip = self.embedder.embed_text(text)
        # CLAP text embedding (for text-audio comparison)
        emb_text_clap = self.embedder.embed_text_for_audio(text) if audio_path else None

        emb_image = None
        emb_audio = None

        if image_path:
            emb_image = self.embedder.embed_image(image_path)
        if audio_path:
            emb_audio = self.embedder.embed_audio(audio_path)

        scores: Dict[str, float | None] = {}

        # Text-Image: CLIP shared space (meaningful)
        if emb_text_clip is not None and emb_image is not None:
            scores["st_i"] = float(round(cosine_similarity(emb_text_clip, emb_image), 4))
            logger.debug("st_i = %.4f", scores["st_i"])

        # Text-Audio: CLAP shared space (meaningful)
        if emb_text_clap is not None and emb_audio is not None:
            scores["st_a"] = float(round(cosine_similarity(emb_text_clap, emb_audio), 4))
            logger.debug("st_a = %.4f", scores["st_a"])

        # Image-Audio: cross-space (CLIP image vs CLAP audio)
        # Without a bridge, these live in different spaces — similarity is meaningless.
        # With a trained bridge, project both into a shared space for si_a.
        if self._bridge is not None and emb_image is not None and emb_audio is not None:
            scores["si_a"] = float(round(
                self._bridge.compute_similarity(emb_image, emb_audio), 4
            ))
            logger.debug("si_a = %.4f (via bridge)", scores["si_a"])
        else:
            scores["si_a"] = None

        # Compute MSCI from available scores
        available = {k: v for k, v in scores.items() if v is not None}
        if len(available) >= 2:
            weights = {"st_i": 0.45, "st_a": 0.45, "si_a": 0.10}
            total = sum(weights[k] for k in available if k in weights)
            msci = sum(available[k] * weights[k] for k in available if k in weights) / max(total, 1e-6)
            scores["msci"] = float(round(msci, 4))
        elif len(available) == 1:
            scores["msci"] = float(round(list(available.values())[0], 4))
        else:
            scores["msci"] = None

        logger.info("MSCI = %s (from %d pairwise scores)", scores["msci"], len(available))

        drift = detect_drift(
            scores.get("msci"),
            scores.get("st_i"),
            scores.get("st_a"),
            scores.get("si_a"),
        )
        coherence = self.scorer.score(scores=scores, global_drift=drift["global_drift"])

        return {
            "scores": scores,
            "drift": drift,
            "coherence": coherence,
            "classification": coherence["classification"],
            "final_score": coherence["final_score"],
        }


def evaluate_coherence(
    text: str,
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    engine = CoherenceEngine()
    return engine.evaluate(
        text=text,
        image_path=image_path,
        audio_path=audio_path,
    )
