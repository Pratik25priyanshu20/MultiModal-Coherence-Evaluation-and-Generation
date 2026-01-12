from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt


def plot_distribution(values: List[float], title: str, xlabel: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_msci_distributions(results: List[Dict]) -> None:
    plot_distribution(
        [result["msci"] for result in results],
        "MSCI Distribution (Gold v0)",
        "MSCI score",
    )
    plot_distribution(
        [result["st_i"] for result in results],
        "Text–Image Similarity",
        "Cosine similarity",
    )
    plot_distribution(
        [result["st_a"] for result in results],
        "Text–Audio Similarity",
        "Cosine similarity",
    )
