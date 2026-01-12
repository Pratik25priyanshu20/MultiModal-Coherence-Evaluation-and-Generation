from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
import json
import statistics
import time
import uuid

from src.pipeline.generate_and_evaluate import generate_and_evaluate


@dataclass
class BatchResult:
    run_id: str
    prompt: str
    n_samples: int
    individual_runs: List[str]
    coherence_stats: Dict[str, Dict[str, float]]
    meta: Dict[str, Any]


def _new_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def generate_and_evaluate_batch(
    prompt: str,
    n_samples: int = 5,
    out_dir: str = "runs/unified_batch",
    use_ollama: bool = True,
    deterministic: bool = True,
    seed: int = 42,
) -> BatchResult:
    """
    Run multiple independent generations and aggregate coherence metrics.
    """
    run_id = _new_run_id()
    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    scores_by_metric: Dict[str, List[float]] = {}

    for i in range(n_samples):
        print(f"[Batch] Running sample {i + 1}/{n_samples}")

        bundle = generate_and_evaluate(
            prompt=prompt,
            out_dir=str(out_dir),
            use_ollama=use_ollama,
            deterministic=deterministic,
            seed=seed,
        )

        runs.append(bundle.run_id)

        for metric, value in bundle.scores.items():
            scores_by_metric.setdefault(metric, []).append(value)

    coherence_stats = {}
    for metric, values in scores_by_metric.items():
        coherence_stats[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }

    result = BatchResult(
        run_id=run_id,
        prompt=prompt,
        n_samples=n_samples,
        individual_runs=runs,
        coherence_stats=coherence_stats,
        meta={
            "use_ollama": use_ollama,
            "deterministic": deterministic,
            "seed": seed,
            "out_dir": str(out_dir),
        },
    )

    with (out_dir / "bundle_batch.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    print("\n=== BATCH RUN COMPLETE ===")
    print("run_id:", run_id)
    print("samples:", n_samples)
    print("saved to:", out_dir / "bundle_batch.json")

    return result
