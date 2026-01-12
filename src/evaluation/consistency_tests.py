from __future__ import annotations

from typing import Dict, List


def human_alignment_score(human_criteria: Dict[str, bool]) -> float:
    """
    Simple proxy score: % of human criteria satisfied.
    """
    values = list(human_criteria.values())
    return sum(values) / len(values)


def compare_human_vs_msci(results: List[Dict]):
    diffs = []

    for result in results:
        human_score = human_alignment_score(result["human"])
        diffs.append(
            {
                "id": result["id"],
                "human_score": human_score,
                "msci": result["msci"],
                "delta": result["msci"] - human_score,
            }
        )

    return diffs
