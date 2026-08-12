"""
Public API for cMSCI — the Calibrated Multimodal Semantic Coherence Index.

Quickstart:
    from cmsci import CoherenceScorer

    scorer = CoherenceScorer()
    result = scorer.score(
        text="rain falling in a dense forest",
        image="scene.png",
        audio="rain.wav",
    )
    print(result.cmsci, result.st_i, result.st_a)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cmsci.assets import AssetPaths, resolve_assets


@dataclass
class ScoreResult:
    """Result of a single coherence evaluation."""

    cmsci: Optional[float]
    """Calibrated cMSCI score (higher = more coherent)."""

    msci: Optional[float]
    """Legacy MSCI baseline score, for comparison."""

    variant: str
    """Which cMSCI variant produced the score (A–F, F = full pipeline)."""

    st_i: Optional[float]
    """Text–image cosine similarity (CLIP space)."""

    st_a: Optional[float]
    """Text–audio cosine similarity (CLAP space)."""

    si_a: Optional[float]
    """Image–audio similarity via the cross-space bridge (None without bridge)."""

    variant_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    """Scores of every ablation variant A–F."""

    uncertainty: Optional[Dict[str, Any]] = None
    """Probabilistic-adapter uncertainty estimates, when adapters are loaded."""

    raw: Dict[str, Any] = field(default_factory=dict)
    """Full engine output (gram volumes, calibration z-scores, margins, ...)."""


class CoherenceScorer:
    """
    Scores the semantic coherence of text / image / audio triples.

    Wraps the CalibratedCoherenceEngine with automatic asset resolution
    (calibration stats, Ex-MCR projector, cross-space bridge, probabilistic
    adapters, negative-bank indexes). Any missing asset degrades gracefully.

    Args:
        assets_dir: Directory with the cMSCI assets. Defaults to the assets
            bundled with the package.
        negative_bank: Enable contrastive-margin calibration against hard
            negatives (requires the embedding indexes).
        use_bridge: Load the cross-space bridge for the image–audio channel.
        target_dim: Embedding dimensionality (512 for CLIP/CLAP).
    """

    def __init__(
        self,
        assets_dir: Optional[str] = None,
        negative_bank: bool = True,
        use_bridge: bool = True,
        target_dim: int = 512,
    ):
        from cmsci.coherence.cmsci_engine import CalibratedCoherenceEngine

        self.assets: AssetPaths = resolve_assets(assets_dir)
        a = self.assets
        self._engine = CalibratedCoherenceEngine(
            target_dim=target_dim,
            calibration_path=str(a.calibration) if a.calibration else None,
            exmcr_weights_path=str(a.exmcr) if a.exmcr else None,
            bridge_path=str(a.bridge) if (use_bridge and a.bridge) else None,
            prob_clip_adapter_path=str(a.clip_adapter) if a.clip_adapter else None,
            prob_clap_adapter_path=str(a.clap_adapter) if a.clap_adapter else None,
            negative_bank_enabled=negative_bank,
            image_index_path=str(a.image_index) if a.image_index else None,
            audio_index_path=str(a.audio_index) if a.audio_index else None,
        )

    @property
    def engine(self):
        """The underlying CalibratedCoherenceEngine, for advanced use."""
        return self._engine

    def score(
        self,
        text: str,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        domain: str = "",
    ) -> ScoreResult:
        """
        Evaluate the coherence of a text / image / audio triple.

        Args:
            text: The textual description or narrative.
            image: Path to an image file (optional).
            audio: Path to an audio file — wav/flac/mp3 (optional).
            domain: Optional domain hint ("nature", "urban", "water", ...)
                used to pick harder negatives for contrastive calibration.

        Returns:
            ScoreResult with the calibrated cMSCI score and channel breakdown.
        """
        out = self._engine.evaluate(
            text=text, image_path=image, audio_path=audio, domain=domain
        )
        scores = out.get("scores", {})
        return ScoreResult(
            cmsci=out.get("cmsci"),
            msci=out.get("msci"),
            variant=out.get("active_variant", ""),
            st_i=scores.get("st_i"),
            st_a=scores.get("st_a"),
            si_a=scores.get("si_a"),
            variant_scores=out.get("variant_scores", {}),
            uncertainty=out.get("uncertainty"),
            raw=out,
        )

    def score_batch(self, items: List[Dict[str, Any]]) -> List[ScoreResult]:
        """
        Evaluate a batch of samples.

        Args:
            items: Dicts with keys "text" and optional "image", "audio", "domain".
        """
        return [
            self.score(
                text=item["text"],
                image=item.get("image"),
                audio=item.get("audio"),
                domain=item.get("domain", ""),
            )
            for item in items
        ]
