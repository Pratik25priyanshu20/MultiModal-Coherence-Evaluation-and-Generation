from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec) + eps
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
