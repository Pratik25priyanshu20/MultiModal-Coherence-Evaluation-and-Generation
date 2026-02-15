"""
MSCI v2: Learnable Multimodal Semantic Coherence Index

Extends MSCI with:
1. Learned weights optimized from human evaluation data
2. Optional learned projection heads for embedding alignment
3. Uncertainty estimation via ensemble or dropout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from src.coherence.msci import MSCIResult, compute_msci_v0
from src.embeddings.similarity import cosine_similarity


@dataclass
class MSCIv2Weights:
    """Learned weights for MSCI v2."""
    w_ti: float  # text-image weight
    w_ta: float  # text-audio weight
    w_ia: float  # image-audio weight
    correlation_with_human: float  # Achieved correlation
    n_training_samples: int
    optimization_method: str = "spearman"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_ti": self.w_ti,
            "w_ta": self.w_ta,
            "w_ia": self.w_ia,
            "correlation_with_human": self.correlation_with_human,
            "n_training_samples": self.n_training_samples,
            "optimization_method": self.optimization_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MSCIv2Weights":
        return cls(**data)

    def save(self, path: Path):
        """Save weights to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MSCIv2Weights":
        """Load weights from JSON."""
        with Path(path).open("r") as f:
            return cls.from_dict(json.load(f))


def optimize_msci_weights(
    human_scores: List[float],
    st_i_scores: List[float],
    st_a_scores: List[float],
    si_a_scores: List[float],
    method: str = "spearman",
    constraint: str = "sum_to_one",
) -> MSCIv2Weights:
    """
    Optimize MSCI weights to maximize correlation with human scores.

    Args:
        human_scores: Human coherence ratings (normalized 0-1)
        st_i_scores: Text-image similarity scores
        st_a_scores: Text-audio similarity scores
        si_a_scores: Image-audio similarity scores
        method: Correlation method ("spearman" or "pearson")
        constraint: Weight constraint ("sum_to_one" or "none")

    Returns:
        MSCIv2Weights with optimized values
    """
    human = np.array(human_scores)
    st_i = np.array(st_i_scores)
    st_a = np.array(st_a_scores)
    si_a = np.array(si_a_scores)

    def compute_msci(weights: np.ndarray) -> np.ndarray:
        """Compute MSCI for given weights."""
        w_ti, w_ta, w_ia = weights
        total = w_ti + w_ta + w_ia
        return (w_ti * st_i + w_ta * st_a + w_ia * si_a) / total

    def negative_correlation(weights: np.ndarray) -> float:
        """Objective: negative correlation (to minimize)."""
        msci = compute_msci(weights)

        if method == "spearman":
            corr = stats.spearmanr(msci, human).correlation
        else:
            corr = stats.pearsonr(msci, human).statistic

        if np.isnan(corr):
            return 0.0  # Handle edge cases

        return -corr  # Negative because we minimize

    # Initial weights
    x0 = np.array([0.45, 0.45, 0.10])

    # Bounds: each weight in [0, 1]
    bounds = [(0.01, 1.0), (0.01, 1.0), (0.01, 1.0)]

    # Constraints
    constraints = []
    if constraint == "sum_to_one":
        constraints.append({
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0
        })

    # Optimize
    result = minimize(
        negative_correlation,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    optimal_weights = result.x
    final_msci = compute_msci(optimal_weights)

    if method == "spearman":
        final_corr = stats.spearmanr(final_msci, human).correlation
    else:
        final_corr = stats.pearsonr(final_msci, human).statistic

    return MSCIv2Weights(
        w_ti=float(optimal_weights[0]),
        w_ta=float(optimal_weights[1]),
        w_ia=float(optimal_weights[2]),
        correlation_with_human=float(final_corr),
        n_training_samples=len(human_scores),
        optimization_method=method,
    )


def grid_search_weights(
    human_scores: List[float],
    st_i_scores: List[float],
    st_a_scores: List[float],
    si_a_scores: List[float],
    resolution: int = 20,
) -> Tuple[MSCIv2Weights, Dict[str, Any]]:
    """
    Grid search over weight combinations.

    More interpretable than optimization, shows sensitivity.

    Args:
        human_scores: Human ratings
        st_i_scores, st_a_scores, si_a_scores: Component similarities
        resolution: Grid resolution per dimension

    Returns:
        Tuple of (best weights, full grid results)
    """
    human = np.array(human_scores)
    st_i = np.array(st_i_scores)
    st_a = np.array(st_a_scores)
    si_a = np.array(si_a_scores)

    best_corr = -1.0
    best_weights = (0.45, 0.45, 0.10)
    grid_results = []

    # Generate weight combinations that sum to 1
    for w_ti in np.linspace(0.1, 0.8, resolution):
        for w_ta in np.linspace(0.1, 0.8, resolution):
            w_ia = 1.0 - w_ti - w_ta
            if w_ia < 0.01 or w_ia > 0.5:
                continue

            msci = (w_ti * st_i + w_ta * st_a + w_ia * si_a)
            corr = stats.spearmanr(msci, human).correlation

            grid_results.append({
                "w_ti": w_ti,
                "w_ta": w_ta,
                "w_ia": w_ia,
                "correlation": corr,
            })

            if corr > best_corr:
                best_corr = corr
                best_weights = (w_ti, w_ta, w_ia)

    weights = MSCIv2Weights(
        w_ti=float(best_weights[0]),
        w_ta=float(best_weights[1]),
        w_ia=float(best_weights[2]),
        correlation_with_human=float(best_corr),
        n_training_samples=len(human_scores),
        optimization_method="grid_search",
    )

    return weights, {"grid": grid_results, "resolution": resolution}


@dataclass
class MSCIv2Result:
    """Result from MSCI v2 computation."""
    msci: float
    st_i: float
    st_a: float
    si_a: float
    weights: Dict[str, float]
    version: str = "v2"
    uncertainty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msci": self.msci,
            "st_i": self.st_i,
            "st_a": self.st_a,
            "si_a": self.si_a,
            "weights": self.weights,
            "version": self.version,
            "uncertainty": self.uncertainty,
        }


def compute_msci_v2(
    emb_text: np.ndarray,
    emb_image: np.ndarray,
    emb_audio: np.ndarray,
    weights: Optional[MSCIv2Weights] = None,
    include_image_audio: bool = True,
) -> MSCIv2Result:
    """
    Compute MSCI v2 with learned or custom weights.

    Args:
        emb_text: Text embedding
        emb_image: Image embedding
        emb_audio: Audio embedding
        weights: Optional learned weights (uses default if None)
        include_image_audio: Whether to include image-audio similarity

    Returns:
        MSCIv2Result with score and metadata
    """
    # Use learned weights or defaults
    if weights:
        w_ti = weights.w_ti
        w_ta = weights.w_ta
        w_ia = weights.w_ia
    else:
        # Default v1 weights
        w_ti, w_ta, w_ia = 0.45, 0.45, 0.10

    # Compute similarities
    st_i = cosine_similarity(emb_text, emb_image)
    st_a = cosine_similarity(emb_text, emb_audio)
    si_a = cosine_similarity(emb_image, emb_audio) if include_image_audio else None

    # Compute MSCI
    if include_image_audio and si_a is not None:
        total = w_ti + w_ta + w_ia
        msci = (w_ti * st_i + w_ta * st_a + w_ia * si_a) / total
        weight_dict = {"w_ti": w_ti, "w_ta": w_ta, "w_ia": w_ia}
    else:
        total = w_ti + w_ta
        msci = (w_ti * st_i + w_ta * st_a) / total
        weight_dict = {"w_ti": w_ti, "w_ta": w_ta}

    return MSCIv2Result(
        msci=float(round(msci, 4)),
        st_i=float(round(st_i, 4)),
        st_a=float(round(st_a, 4)),
        si_a=float(round(si_a, 4)) if si_a is not None else None,
        weights=weight_dict,
        version="v2" if weights else "v1",
    )


def compare_msci_versions(
    human_scores: List[float],
    st_i_scores: List[float],
    st_a_scores: List[float],
    si_a_scores: List[float],
) -> Dict[str, Any]:
    """
    Compare MSCI v1 (fixed weights) vs v2 (learned weights).

    Args:
        human_scores: Human ratings
        st_i_scores, st_a_scores, si_a_scores: Component similarities

    Returns:
        Comparison results
    """
    human = np.array(human_scores)
    st_i = np.array(st_i_scores)
    st_a = np.array(st_a_scores)
    si_a = np.array(si_a_scores)

    # V1: Fixed weights
    msci_v1 = (0.45 * st_i + 0.45 * st_a + 0.10 * si_a)
    corr_v1 = stats.spearmanr(msci_v1, human).correlation

    # V2: Optimized weights
    v2_weights = optimize_msci_weights(
        human_scores, st_i_scores, st_a_scores, si_a_scores
    )
    msci_v2 = (v2_weights.w_ti * st_i + v2_weights.w_ta * st_a + v2_weights.w_ia * si_a)
    corr_v2 = stats.spearmanr(msci_v2, human).correlation

    improvement = corr_v2 - corr_v1

    return {
        "v1": {
            "weights": {"w_ti": 0.45, "w_ta": 0.45, "w_ia": 0.10},
            "correlation": corr_v1,
        },
        "v2": {
            "weights": v2_weights.to_dict(),
            "correlation": corr_v2,
        },
        "improvement": improvement,
        "interpretation": (
            f"V2 improves correlation by {improvement:.3f}"
            if improvement > 0.01 else
            "V2 shows minimal improvement over V1"
        ),
        "n_samples": len(human_scores),
    }
