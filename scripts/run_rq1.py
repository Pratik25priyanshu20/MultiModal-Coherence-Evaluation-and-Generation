"""
RQ1: Is MSCI sensitive to controlled semantic perturbations?

Design: 30 prompts × 3 seeds × 3 conditions (baseline, wrong_image, wrong_audio)
        = 270 experimental runs

Each prompt is tested under:
- baseline: matched image + matched audio
- wrong_image: deliberately mismatched image (different domain)
- wrong_audio: deliberately mismatched audio (different domain)

Success criteria:
- MSCI(baseline) > MSCI(wrong_image), p < 0.05, Cohen's d > 0.5
- MSCI(baseline) > MSCI(wrong_audio), p < 0.05, Cohen's d > 0.5

Can run without Ollama: --skip-text (uses prompt as text directly)

Usage:
    python scripts/run_rq1.py                    # Full pipeline (needs Ollama)
    python scripts/run_rq1.py --skip-text         # Without text generation
    python scripts/run_rq1.py --n-prompts 10      # Quick test with 10 prompts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SEEDS = [42, 123, 7]
CONDITIONS = ["baseline", "wrong_image", "wrong_audio"]


def load_prompts(path: str, n: int = 30) -> List[Dict[str, str]]:
    with open(path) as f:
        data = json.load(f)
    prompts = data["prompts"][:n]
    return prompts


def run_single(
    prompt_text: str,
    seed: int,
    condition: str,
    out_dir: str,
    skip_text: bool,
) -> Dict[str, Any]:
    """Run a single experimental condition."""
    if skip_text:
        from src.generators.image.generator_improved import generate_image_with_metadata
        from src.generators.audio.retrieval import retrieve_audio_with_metadata
        from src.coherence.coherence_engine import evaluate_coherence
        from src.pipeline.generate_and_evaluate import (
            _collect_all_image_paths,
            _collect_all_audio_paths,
            _select_mismatched_path,
            _infer_domain_from_path,
        )
        from src.utils.seed import set_global_seed

        set_global_seed(seed)
        generated_text = prompt_text

        # Image retrieval
        image_result = generate_image_with_metadata(prompt=prompt_text, min_similarity=0.20)
        image_path = image_result.image_path
        image_sim = image_result.similarity
        image_domain = image_result.domain

        # Audio retrieval
        audio_result = retrieve_audio_with_metadata(prompt=prompt_text, min_similarity=0.10)
        audio_path = audio_result.audio_path
        audio_sim = audio_result.similarity

        # Perturbation
        perturbation = {"applied": condition}
        if condition == "wrong_image":
            orig_domain = _infer_domain_from_path(image_path)
            images = _collect_all_image_paths()
            repl = _select_mismatched_path(images, str(image_path), orig_domain, seed)
            if repl:
                perturbation["original_image"] = str(image_path)
                perturbation["replacement_image"] = repl
                image_path = repl

        if condition == "wrong_audio":
            orig_domain = _infer_domain_from_path(audio_path)
            audios = _collect_all_audio_paths()
            repl = _select_mismatched_path(audios, str(audio_path), orig_domain, seed)
            if repl:
                perturbation["original_audio"] = str(audio_path)
                perturbation["replacement_audio"] = repl
                audio_path = repl

        # Evaluate coherence
        eval_out = evaluate_coherence(
            text=generated_text,
            image_path=str(image_path),
            audio_path=str(audio_path),
        )
        scores = eval_out.get("scores", {})

        return {
            "msci": scores.get("msci"),
            "st_i": scores.get("st_i"),
            "st_a": scores.get("st_a"),
            "si_a": scores.get("si_a"),
            "image_path": str(image_path),
            "image_sim": image_sim,
            "image_domain": image_domain,
            "audio_path": str(audio_path),
            "audio_sim": audio_sim,
            "perturbation": perturbation,
        }
    else:
        from src.pipeline.generate_and_evaluate import generate_and_evaluate

        bundle = generate_and_evaluate(
            prompt=prompt_text,
            out_dir=out_dir,
            use_ollama=True,
            deterministic=True,
            seed=seed,
            mode="direct",
            condition=condition,
        )
        return {
            "msci": bundle.scores.get("msci"),
            "st_i": bundle.scores.get("st_i"),
            "st_a": bundle.scores.get("st_a"),
            "si_a": bundle.scores.get("si_a"),
            "image_path": bundle.image_path,
            "audio_path": bundle.audio_path,
            "perturbation": bundle.meta.get("perturbation", {}),
        }


def main():
    parser = argparse.ArgumentParser(description="RQ1: MSCI Sensitivity Experiment")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--prompts-file", default="data/prompts/experiment_prompts.json")
    parser.add_argument("--out-dir", default="runs/rq1")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel threads (default: 1 = sequential)")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file, args.n_prompts)
    total = len(prompts) * len(SEEDS) * len(CONDITIONS)

    print("=" * 90)
    print("RQ1: IS MSCI SENSITIVE TO CONTROLLED SEMANTIC PERTURBATIONS?")
    print("=" * 90)
    print(f"  Prompts:     {len(prompts)}")
    print(f"  Seeds:       {SEEDS}")
    print(f"  Conditions:  {CONDITIONS}")
    print(f"  Total runs:  {total}")
    print(f"  Skip text:   {args.skip_text}")
    print(f"  Parallel:    {args.parallel}")
    print("=" * 90)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pre-warm shared embedder for parallel mode (avoids each thread loading CLIP+CLAP)
    if args.parallel > 1:
        print("Pre-loading shared embedder for parallel execution...")
        from src.embeddings.shared_embedder import get_shared_embedder
        get_shared_embedder()
        print("Shared embedder ready.")

    # Build flat task list
    tasks = []
    for prompt_info in prompts:
        pid = prompt_info["id"]
        text = prompt_info["text"]
        domain = prompt_info["domain"]
        for seed in SEEDS:
            for condition in CONDITIONS:
                tasks.append({
                    "prompt_id": pid,
                    "prompt_text": text,
                    "domain": domain,
                    "seed": seed,
                    "condition": condition,
                    "out_dir": str(out_path / f"{pid}_s{seed}_{condition}"),
                })

    results = []
    t_start = time.time()

    def _execute_task(task, idx):
        t0 = time.time()
        try:
            result = run_single(
                prompt_text=task["prompt_text"],
                seed=task["seed"],
                condition=task["condition"],
                out_dir=task["out_dir"],
                skip_text=args.skip_text,
            )
            elapsed = time.time() - t0
            msci_str = f"MSCI={result['msci']:.4f}" if result.get('msci') is not None else "MSCI=N/A"
            print(f"[{idx}/{total}] {task['prompt_id']} seed={task['seed']} cond={task['condition']}  {msci_str}  ({elapsed:.1f}s)")
            return {**task, **result}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{idx}/{total}] {task['prompt_id']} seed={task['seed']} cond={task['condition']}  ERROR: {str(e)[:60]}  ({elapsed:.1f}s)")
            return {**task, "error": str(e)}

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(_execute_task, task, i + 1): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i, task in enumerate(tasks):
            results.append(_execute_task(task, i + 1))

    total_time = time.time() - t_start

    # Save raw results
    results_path = out_path / "rq1_results.json"
    with results_path.open("w") as f:
        json.dump({
            "experiment": "RQ1: MSCI Sensitivity",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "n_prompts": len(prompts),
                "seeds": SEEDS,
                "conditions": CONDITIONS,
                "skip_text": args.skip_text,
                "total_runs": total,
                "total_time_sec": round(total_time, 1),
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Quick summary
    _print_quick_summary(results)

    return 0


def _print_quick_summary(results: List[Dict[str, Any]]) -> None:
    """Print quick MSCI summary by condition."""
    print("\n" + "=" * 90)
    print("QUICK SUMMARY")
    print("=" * 90)

    for condition in CONDITIONS:
        scores = [
            r["msci"] for r in results
            if r.get("msci") is not None and r["condition"] == condition
        ]
        if scores:
            print(
                f"  {condition:<14}  N={len(scores):>3}  "
                f"mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  "
                f"median={np.median(scores):.4f}"
            )

    # Deltas
    baseline_scores = {}
    for r in results:
        if r["condition"] == "baseline" and r.get("msci") is not None:
            key = (r["prompt_id"], r["seed"])
            baseline_scores[key] = r["msci"]

    for condition in ["wrong_image", "wrong_audio"]:
        deltas = []
        for r in results:
            if r["condition"] == condition and r.get("msci") is not None:
                key = (r["prompt_id"], r["seed"])
                if key in baseline_scores:
                    deltas.append(baseline_scores[key] - r["msci"])
        if deltas:
            print(
                f"\n  Δ(baseline - {condition}):  "
                f"mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}  "
                f"all_positive={all(d > 0 for d in deltas)}"
            )

    print("\nRun: python scripts/analyze_results.py runs/rq1/rq1_results.json")


if __name__ == "__main__":
    sys.exit(main())
