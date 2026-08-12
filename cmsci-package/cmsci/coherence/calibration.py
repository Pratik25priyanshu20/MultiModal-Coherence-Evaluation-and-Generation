"""
Distribution Normalization for cMSCI.

Scores from different embedding spaces (CLIP vs CLAP) and different
pairwise channels (st_i, st_a, gram_volume) have different natural
distributions. Z-score normalization makes them comparable.

The ReferenceDistribution class fits mean/std from existing experiment
data and normalizes new scores to z-scores or percentile ranks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class ReferenceDistribution:
    """
    Stores mean/std for a single score channel and provides normalization.

    Usage:
        ref = ReferenceDistribution()
        ref.fit(list_of_scores)
        z = ref.normalize(new_score)         # z-score
        p = ref.percentile(new_score)        # percentile rank [0, 1]
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.mean: float = 0.0
        self.std: float = 1.0
        self.n: int = 0
        self._sorted_values: Optional[np.ndarray] = None

    def fit(self, scores: List[float]) -> None:
        """Fit the distribution from a list of observed scores."""
        arr = np.array(scores, dtype=np.float64)
        self.n = len(arr)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr, ddof=1)) if self.n > 1 else 1.0
        if self.std < 1e-10:
            self.std = 1.0
        self._sorted_values = np.sort(arr)

    def normalize(self, score: float) -> float:
        """Z-score normalization: (score - mean) / std."""
        return float((score - self.mean) / self.std)

    def percentile(self, score: float) -> float:
        """
        Percentile rank of score within the reference distribution.

        Returns a value in [0, 1] where 0.5 = median of reference.
        """
        if self._sorted_values is None or len(self._sorted_values) == 0:
            return 0.5
        rank = np.searchsorted(self._sorted_values, score, side="right")
        return float(rank / len(self._sorted_values))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ReferenceDistribution":
        obj = cls(name=d.get("name", ""))
        obj.mean = d["mean"]
        obj.std = d["std"]
        obj.n = d.get("n", 0)
        return obj

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ReferenceDistribution":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class CalibrationStore:
    """
    Collection of ReferenceDistributions for all score channels.

    Provides save/load for the full calibration state.
    """

    def __init__(self):
        self.distributions: Dict[str, ReferenceDistribution] = {}

    def add(self, name: str, scores: List[float]) -> ReferenceDistribution:
        ref = ReferenceDistribution(name=name)
        ref.fit(scores)
        self.distributions[name] = ref
        logger.info(
            "Calibration[%s]: mean=%.4f, std=%.4f, n=%d",
            name, ref.mean, ref.std, ref.n,
        )
        return ref

    def normalize(self, name: str, score: float) -> float:
        if name not in self.distributions:
            return score
        return self.distributions[name].normalize(score)

    def percentile(self, name: str, score: float) -> float:
        if name not in self.distributions:
            return 0.5
        return self.distributions[name].percentile(score)

    def save(self, path: str) -> None:
        data = {name: ref.to_dict() for name, ref in self.distributions.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Calibration saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "CalibrationStore":
        store = cls()
        with open(path) as f:
            data = json.load(f)
        for name, d in data.items():
            store.distributions[name] = ReferenceDistribution.from_dict(d)
        logger.info("Calibration loaded from %s (%d channels)", path, len(store.distributions))
        return store


def has_channel(store: CalibrationStore, name: str) -> bool:
    """Check if a calibration channel exists in the store."""
    return name in store.distributions


def extend_calibration_with_exmcr(
    store: CalibrationStore,
    gram_coh_ia_scores: List[float],
    gram_coh_tia_scores: Optional[List[float]] = None,
) -> CalibrationStore:
    """
    Extend calibration store with ExMCR-derived channels.

    Args:
        store: Existing CalibrationStore to extend.
        gram_coh_ia_scores: Gram coherence of (image_clip, ExMCR(audio_clap)) pairs.
        gram_coh_tia_scores: Optional 3-way gram coherence of (text, image, ExMCR(audio)).

    Returns:
        Extended CalibrationStore (same object, modified in place).
    """
    if gram_coh_ia_scores:
        store.add("gram_coh_ia_exmcr", gram_coh_ia_scores)
    if gram_coh_tia_scores:
        store.add("gram_coh_tia", gram_coh_tia_scores)
    return store


def extend_calibration_with_uncertainty(
    store: CalibrationStore,
    uncertainty_ti_scores: List[float],
    uncertainty_ta_scores: Optional[List[float]] = None,
) -> CalibrationStore:
    """
    Extend calibration store with ProbVLM uncertainty channels.

    Args:
        store: Existing CalibrationStore to extend.
        uncertainty_ti_scores: Per-sample mean uncertainty for text-image (CLIP adapter).
        uncertainty_ta_scores: Per-sample mean uncertainty for text-audio (CLAP adapter).

    Returns:
        Extended CalibrationStore (same object, modified in place).
    """
    if uncertainty_ti_scores:
        store.add("uncertainty_ti", uncertainty_ti_scores)
    if uncertainty_ta_scores:
        store.add("uncertainty_ta", uncertainty_ta_scores)
    # Combined uncertainty channel
    if uncertainty_ti_scores and uncertainty_ta_scores:
        combined = [
            (ti + ta) / 2.0
            for ti, ta in zip(uncertainty_ti_scores, uncertainty_ta_scores)
        ]
        store.add("uncertainty_mean", combined)
    return store


def build_reference_distributions(
    rq1_results_path: str,
) -> CalibrationStore:
    """
    Build reference distributions from existing RQ1 baseline results.

    Extracts st_i, st_a, and msci scores from baseline condition only
    (matched image + audio), fitting a distribution for each channel.

    Args:
        rq1_results_path: Path to rq1_results.json

    Returns:
        CalibrationStore with fitted distributions for st_i, st_a, msci
    """
    with open(rq1_results_path) as f:
        data = json.load(f)

    st_i_scores = []
    st_a_scores = []
    msci_scores = []

    for r in data["results"]:
        if r.get("condition") != "baseline":
            continue
        if r.get("st_i") is not None:
            st_i_scores.append(r["st_i"])
        if r.get("st_a") is not None:
            st_a_scores.append(r["st_a"])
        if r.get("msci") is not None:
            msci_scores.append(r["msci"])

    store = CalibrationStore()
    if st_i_scores:
        store.add("st_i", st_i_scores)
    if st_a_scores:
        store.add("st_a", st_a_scores)
    if msci_scores:
        store.add("msci", msci_scores)

    # GRAM coherence distributions (1 - gram_volume) for gram calibration mode
    # gram_volume = sqrt(1 - cos^2), so gram_coherence = 1 - sqrt(1 - cos^2)
    if st_i_scores:
        gram_coh_ti = [1.0 - np.sqrt(max(0, 1 - s**2)) for s in st_i_scores]
        store.add("gram_coh_ti", gram_coh_ti)
    if st_a_scores:
        gram_coh_ta = [1.0 - np.sqrt(max(0, 1 - s**2)) for s in st_a_scores]
        store.add("gram_coh_ta", gram_coh_ta)

    return store
