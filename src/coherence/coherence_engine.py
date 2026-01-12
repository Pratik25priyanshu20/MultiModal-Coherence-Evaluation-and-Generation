from __future__ import annotations

from typing import Any, Dict, Optional

from src.coherence.drift_detector import detect_drift
from src.coherence.scorer import CoherenceScorer
from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity


class CoherenceEngine:
    def __init__(self, target_dim: int = 512):
        self.embedder = AlignedEmbedder(target_dim=target_dim)
        self.scorer = CoherenceScorer()

    def evaluate(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        emb_text = self.embedder.embed_text(text)
        emb_image = None
        emb_audio = None

        if image_path:
            emb_image = self.embedder.embed_image(image_path)
        if audio_path:
            emb_audio = self.embedder.embed_audio(audio_path)

        scores: Dict[str, float | None] = {}

        if emb_text is not None and emb_image is not None:
            scores["st_i"] = float(round(cosine_similarity(emb_text, emb_image), 4))
        if emb_text is not None and emb_audio is not None:
            scores["st_a"] = float(round(cosine_similarity(emb_text, emb_audio), 4))
        if emb_image is not None and emb_audio is not None:
            scores["si_a"] = float(round(cosine_similarity(emb_image, emb_audio), 4))

        if len(scores) > 1:
            weights = {"st_i": 0.45, "st_a": 0.45, "si_a": 0.10}
            total = sum(weights[k] for k in scores if k in weights)
            msci = sum(scores[k] * weights[k] for k in scores if k in weights) / max(total, 1e-6)
            scores["msci"] = float(round(msci, 4))
        else:
            scores["msci"] = None

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
