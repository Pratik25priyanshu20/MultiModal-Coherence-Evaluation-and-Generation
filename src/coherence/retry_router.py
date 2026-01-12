from __future__ import annotations

from typing import Optional, Tuple

RETRY_STRATEGY = {
    "si_a": "align_audio_with_visual",
    "st_a": "align_audio_with_text",
    "st_i": "align_image_with_text",
    "msci": "full_regeneration",
}


def select_retry_strategy(metric: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (strategy, target) for a failed metric.
    target is one of: "audio", "image", "full", or None.
    """
    if not metric:
        return None, None

    strategy = RETRY_STRATEGY.get(metric)
    if strategy in {"align_audio_with_visual", "align_audio_with_text"}:
        return strategy, "audio"
    if strategy == "align_image_with_text":
        return strategy, "image"
    if strategy == "full_regeneration":
        return strategy, "full"
    return None, None
