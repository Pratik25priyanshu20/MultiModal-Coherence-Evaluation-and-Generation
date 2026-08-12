"""
Gramian Volume Scoring for Multimodal Coherence.

The Gramian volume measures the geometric dispersion of embedding vectors.
For n L2-normalized vectors, the Gramian matrix G has G_ij = <vi, vj>.

    volume = sqrt(det(G))

Properties:
- Identical vectors → det(G) = 0 → volume = 0 (perfect alignment)
- Mutually orthogonal unit vectors → det(G) = 1 → volume = 1 (max dispersion)
- Coherence = 1 - volume → [0, 1] where 1 = perfect alignment

For 2 unit vectors:
    det(G) = 1 - cos²(θ) = sin²(θ)
    volume = |sin(θ)|
    coherence = 1 - |sin(θ)| ≈ cos(θ) for small angles

For 3 unit vectors:
    det(G) = 1 - cos²(a) - cos²(b) - cos²(c) + 2·cos(a)·cos(b)·cos(c)
    where a, b, c are pairwise angles
    This captures the full tri-modal geometric relationship in one number.
"""

from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector."""
    v = v.astype(np.float64).squeeze()
    norm = np.linalg.norm(v) + eps
    return v / norm


def gram_volume_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Gramian volume for 2 vectors (area of parallelogram).

    For unit vectors: volume = |sin(θ)| where θ is the angle between them.
    Range: [0, 1] — 0 when identical, 1 when orthogonal.
    """
    v1_n = _normalize(v1)
    v2_n = _normalize(v2)
    cos_sim = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    # det(G) = 1 - cos²(θ)
    det_g = 1.0 - cos_sim ** 2
    return float(np.sqrt(max(det_g, 0.0)))


def gram_volume_3d(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
) -> float:
    """
    Gramian volume for 3 vectors (volume of parallelepiped).

    For unit vectors with pairwise cosines a, b, c:
        det(G) = 1 - a² - b² - c² + 2abc

    Range: [0, 1] — 0 when all collinear, 1 when mutually orthogonal.
    """
    v1_n = _normalize(v1)
    v2_n = _normalize(v2)
    v3_n = _normalize(v3)

    a = np.dot(v1_n, v2_n)
    b = np.dot(v1_n, v3_n)
    c = np.dot(v2_n, v3_n)

    det_g = 1.0 - a**2 - b**2 - c**2 + 2.0 * a * b * c
    return float(np.sqrt(max(det_g, 0.0)))


def gram_volume_nd(*vectors: np.ndarray) -> float:
    """
    Gramian volume for n vectors (general case).

    Builds the Gram matrix G_ij = <vi, vj> from L2-normalized vectors
    and returns sqrt(det(G)).

    Args:
        *vectors: Variable number of numpy arrays (embeddings).

    Returns:
        Gramian volume in [0, 1] for unit vectors.
    """
    n = len(vectors)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    if n == 2:
        return gram_volume_2d(vectors[0], vectors[1])
    if n == 3:
        return gram_volume_3d(vectors[0], vectors[1], vectors[2])

    normed = [_normalize(v) for v in vectors]
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dot = np.dot(normed[i], normed[j])
            G[i, j] = dot
            G[j, i] = dot

    det_g = np.linalg.det(G)
    return float(np.sqrt(max(det_g, 0.0)))


def normalized_gram_coherence(volume: float, n_vectors: int = 2) -> float:
    """
    Map Gramian volume to coherence score in [0, 1].

    1 = perfect alignment (volume = 0, all vectors identical)
    0 = maximum dispersion (volume = 1, mutually orthogonal)

    Args:
        volume: Gramian volume (output of gram_volume_* functions).
        n_vectors: Number of vectors used (for documentation; mapping is the same).

    Returns:
        Coherence score in [0, 1].
    """
    return float(max(0.0, min(1.0, 1.0 - volume)))
