"""
Controlled Experiment Analyzer

Reads controlled_experiment_results.json and produces summary statistics
per condition plus simple comparisons (baseline vs wrong image/audio, planner vs direct).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

METRICS = ["msci", "st_i", "st_a", "si_a"]


def _normalize_results(data: Any) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ValueError("Results file must be a dict keyed by condition.")

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for condition_key, value in data.items():
        if isinstance(value, list):
            normalized[condition_key] = value
        elif isinstance(value, dict):
            normalized[condition_key] = [value]
        else:
            normalized[condition_key] = []
    return normalized


def _collect_scores(items: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], int]:
    values = {m: [] for m in METRICS}
    error_count = 0

    for item in items:
        if not isinstance(item, dict):
            error_count += 1
            continue
        if "error" in item:
            error_count += 1
            continue
        scores = item.get("scores")
        if not isinstance(scores, dict):
            error_count += 1
            continue

        for metric in METRICS:
            val = scores.get(metric)
            if isinstance(val, (int, float)):
                values[metric].append(float(val))

    return values, error_count


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean": float(statistics.mean(values)),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
        "count": int(len(values)),
    }


def _label_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        coherence = item.get("coherence", {}) if isinstance(item, dict) else {}
        classification = coherence.get("classification", {}) if isinstance(coherence, dict) else {}
        label = classification.get("label")
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _condition_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    values, error_count = _collect_scores(items)
    metrics_summary = {m: _summarize(v) for m, v in values.items()}
    return {
        "metrics": metrics_summary,
        "errors": error_count,
        "labels": _label_counts(items),
    }


def _mean_delta(
    summary: Dict[str, Any],
    a_key: str,
    b_key: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    a = summary.get(a_key, {}).get("metrics", {})
    b = summary.get(b_key, {}).get("metrics", {})
    for metric in METRICS:
        a_mean = float(a.get(metric, {}).get("mean", 0.0))
        b_mean = float(b.get(metric, {}).get("mean", 0.0))
        out[metric] = a_mean - b_mean
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze controlled experiment results",
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        default="runs/controlled_experiment/controlled_experiment_results.json",
        help="Path to controlled_experiment_results.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path for summary JSON output",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = _normalize_results(data)

    summary: Dict[str, Any] = {}
    for condition_key, items in normalized.items():
        summary[condition_key] = _condition_summary(items)

    comparisons: Dict[str, Any] = {}
    for mode in ["direct", "planner"]:
        baseline = f"{mode}_baseline"
        wrong_image = f"{mode}_wrong_image"
        wrong_audio = f"{mode}_wrong_audio"
        if baseline in summary and wrong_image in summary:
            comparisons[f"{mode}_baseline_vs_wrong_image"] = _mean_delta(
                summary, baseline, wrong_image
            )
        if baseline in summary and wrong_audio in summary:
            comparisons[f"{mode}_baseline_vs_wrong_audio"] = _mean_delta(
                summary, baseline, wrong_audio
            )

    if "planner_baseline" in summary and "direct_baseline" in summary:
        comparisons["planner_vs_direct_baseline"] = _mean_delta(
            summary, "planner_baseline", "direct_baseline"
        )

    output = {
        "results_file": str(results_path),
        "summary": summary,
        "comparisons": comparisons,
    }

    out_path = Path(args.out) if args.out else results_path.with_name("controlled_experiment_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
