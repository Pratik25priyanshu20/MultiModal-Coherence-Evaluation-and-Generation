from __future__ import annotations

from typing import Literal

RegenTarget = Literal["image", "audio", "text", "none"]


def decide_regeneration(
    msci: float,
    st_i: float,
    st_a: float,
    threshold: float,
) -> RegenTarget:
    if msci >= threshold:
        return "none"

    if st_i < st_a:
        return "image"
    if st_a < st_i:
        return "audio"
    return "text"
