from __future__ import annotations

from typing import Any, Dict, Optional

from src.coherence.retry_strategies import retry_msci, retry_si_a, retry_st_a


def route_retry(
    classification: Dict[str, Any],
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Route retry strategy based on coherence classification and weakest metric.
    """
    label = classification.get("label")
    metric = classification.get("weakest_metric")

    if label == "GLOBAL_FAILURE":
        return retry_msci(context)

    if label != "MODALITY_FAILURE":
        return None

    if metric == "si_a":
        return retry_si_a(context)
    if metric == "st_a":
        return retry_st_a(context)
    return None
