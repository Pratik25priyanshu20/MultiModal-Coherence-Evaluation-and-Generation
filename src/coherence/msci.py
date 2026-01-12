from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

from src.embeddings.similarity import cosine_similarity


@dataclass(frozen=True)
class MSCIResult:
    st_i: float
    st_a: float
    si_a: Optional[float]
    msci: float
    weights: Dict[str, float]


def compute_msci_v0(
    emb_text: np.ndarray,
    emb_image: np.ndarray,
    emb_audio: np.ndarray,
    include_image_audio: bool = True,
    w_ti: float = 0.45,
    w_ta: float = 0.45,
    w_ia: float = 0.10,
) -> MSCIResult:
    st_i = cosine_similarity(emb_text, emb_image)
    st_a = cosine_similarity(emb_text, emb_audio)

    si_a = cosine_similarity(emb_image, emb_audio) if include_image_audio else None

    if include_image_audio:
        total = w_ti + w_ta + w_ia
        msci = (w_ti * st_i + w_ta * st_a + w_ia * (si_a or 0.0)) / total
        weights = {"w_ti": w_ti, "w_ta": w_ta, "w_ia": w_ia}
    else:
        total = w_ti + w_ta
        msci = (w_ti * st_i + w_ta * st_a) / total
        weights = {"w_ti": w_ti, "w_ta": w_ta}

    return MSCIResult(
        st_i=float(round(st_i, 4)),
        st_a=float(round(st_a, 4)),
        si_a=float(round(si_a, 4)) if si_a is not None else None,
        msci=float(round(msci, 4)),
        weights=weights,
    )
