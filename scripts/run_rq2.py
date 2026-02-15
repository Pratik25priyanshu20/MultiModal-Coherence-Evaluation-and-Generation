"""
RQ2: Does structured planning improve cross-modal alignment?

Design: 30 prompts × 3 seeds × 4 modes (direct, planner, council, extended_prompt)
        = 360 experimental runs

Each prompt is tested under 4 planning modes:
- direct: raw prompt → generators (no planning)
- planner: 1 LLM call → unified plan → generators
- council: 3 LLM calls (factual, style, audio) → merge → generators
- extended_prompt: 1 LLM call with 3× token budget → generators

All runs use baseline condition (no perturbation).

Success criteria:
- MSCI(planner) > MSCI(direct), p < 0.05
- Isolate: structure effect vs token budget effect

REQUIRES: Ollama running locally (planner modes need LLM calls)

Usage:
    python scripts/run_rq2.py                     # Full experiment
    python scripts/run_rq2.py --n-prompts 10      # Quick test
    python scripts/run_rq2.py --modes direct planner  # Subset of modes
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
ALL_MODES = ["direct", "planner", "council", "extended_prompt"]


def load_prompts(path: str, n: int = 30) -> List[Dict[str, str]]:
    with open(path) as f:
        data = json.load(f)
    return data["prompts"][:n]


def run_single(
    prompt_text: str,
    seed: int,
    mode: str,
    out_dir: str,
) -> Dict[str, Any]:
    """Run a single prompt through a specific planning mode."""
    from src.utils.seed import set_global_seed
    set_global_seed(seed)

    # Mode dispatching
    if mode == "direct":
        return _run_direct(prompt_text, seed, out_dir)
    elif mode == "planner":
        return _run_planner(prompt_text, seed, out_dir)
    elif mode == "council":
        return _run_council(prompt_text, seed, out_dir)
    elif mode == "extended_prompt":
        return _run_extended(prompt_text, seed, out_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _run_direct(prompt_text: str, seed: int, out_dir: str) -> Dict[str, Any]:
    """Direct mode: raw prompt → generators."""
    from src.pipeline.generate_and_evaluate import generate_and_evaluate

    bundle = generate_and_evaluate(
        prompt=prompt_text,
        out_dir=out_dir,
        use_ollama=True,
        deterministic=True,
        seed=seed,
        mode="direct",
        condition="baseline",
    )
    return _bundle_to_result(bundle, mode="direct")


def _run_planner(prompt_text: str, seed: int, out_dir: str) -> Dict[str, Any]:
    """Planner mode: 1 LLM call → unified plan → generators."""
    from src.pipeline.generate_and_evaluate import generate_and_evaluate

    bundle = generate_and_evaluate(
        prompt=prompt_text,
        out_dir=out_dir,
        use_ollama=True,
        deterministic=True,
        seed=seed,
        mode="planner",
        condition="baseline",
    )
    return _bundle_to_result(bundle, mode="planner")


def _run_council(prompt_text: str, seed: int, out_dir: str) -> Dict[str, Any]:
    """Council mode: 3 LLM calls → merge → generators."""
    from src.planner.council import SemanticPlanningCouncil
    from src.planner.unified_planner import UnifiedPlannerLLM
    from src.planner.schema_to_text import plan_to_prompts
    from src.generators.text.generator import generate_text
    from src.generators.image.generator_improved import generate_image_with_metadata
    from src.generators.audio.retrieval import retrieve_audio_with_metadata
    from src.coherence.coherence_engine import evaluate_coherence

    t0 = time.time()

    # Council planning: 3 independent LLM calls (same prompt, stochastic diversity) → merge
    planner_a = UnifiedPlannerLLM()
    planner_a.name = "PlannerA"
    planner_b = UnifiedPlannerLLM()
    planner_b.name = "PlannerB"
    planner_c = UnifiedPlannerLLM()
    planner_c.name = "PlannerC"

    council = SemanticPlanningCouncil(planner_a, planner_b, planner_c)
    council_result = council.run(prompt_text)
    plan = council_result.merged_plan
    prompts = plan_to_prompts(plan)

    planning_time = time.time() - t0

    # Generate
    generated_text = generate_text(prompt=prompts["text_prompt"], use_ollama=True, deterministic=True)
    image_result = generate_image_with_metadata(prompt=prompts["image_prompt"], min_similarity=0.20)
    audio_result = retrieve_audio_with_metadata(prompt=prompts["audio_prompt"], min_similarity=0.10)

    # Evaluate
    eval_out = evaluate_coherence(
        text=generated_text,
        image_path=image_result.image_path,
        audio_path=audio_result.audio_path,
    )
    scores = eval_out.get("scores", {})

    return {
        "msci": scores.get("msci"),
        "st_i": scores.get("st_i"),
        "st_a": scores.get("st_a"),
        "si_a": scores.get("si_a"),
        "image_path": image_result.image_path,
        "audio_path": audio_result.audio_path,
        "mode": "council",
        "planning_time_sec": round(planning_time, 2),
        "council_agreement": getattr(council_result.merge_report, "agreement_score", None),
    }


def _run_extended(prompt_text: str, seed: int, out_dir: str) -> Dict[str, Any]:
    """Extended prompt mode: 1 LLM call with 3× token budget."""
    from src.planner.extended_prompt_planner import ExtendedPromptPlanner
    from src.planner.schema_to_text import plan_to_prompts
    from src.generators.text.generator import generate_text
    from src.generators.image.generator_improved import generate_image_with_metadata
    from src.generators.audio.retrieval import retrieve_audio_with_metadata
    from src.coherence.coherence_engine import evaluate_coherence

    t0 = time.time()

    # Extended planning (1 LLM call, 3× tokens)
    planner = ExtendedPromptPlanner()
    plan = planner.plan(prompt_text)
    prompts = plan_to_prompts(plan)

    planning_time = time.time() - t0

    # Generate
    generated_text = generate_text(prompt=prompts["text_prompt"], use_ollama=True, deterministic=True)
    image_result = generate_image_with_metadata(prompt=prompts["image_prompt"], min_similarity=0.20)
    audio_result = retrieve_audio_with_metadata(prompt=prompts["audio_prompt"], min_similarity=0.10)

    # Evaluate
    eval_out = evaluate_coherence(
        text=generated_text,
        image_path=image_result.image_path,
        audio_path=audio_result.audio_path,
    )
    scores = eval_out.get("scores", {})

    return {
        "msci": scores.get("msci"),
        "st_i": scores.get("st_i"),
        "st_a": scores.get("st_a"),
        "si_a": scores.get("si_a"),
        "image_path": image_result.image_path,
        "audio_path": audio_result.audio_path,
        "mode": "extended_prompt",
        "planning_time_sec": round(planning_time, 2),
    }


def _bundle_to_result(bundle, mode: str) -> Dict[str, Any]:
    """Convert a GenerationBundle to a result dict."""
    return {
        "msci": bundle.scores.get("msci"),
        "st_i": bundle.scores.get("st_i"),
        "st_a": bundle.scores.get("st_a"),
        "si_a": bundle.scores.get("si_a"),
        "image_path": bundle.image_path,
        "audio_path": bundle.audio_path,
        "mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="RQ2: Planning Effect Experiment")
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--prompts-file", default="data/prompts/experiment_prompts.json")
    parser.add_argument("--out-dir", default="runs/rq2")
    parser.add_argument(
        "--modes", nargs="+", choices=ALL_MODES, default=ALL_MODES,
        help="Which modes to test (default: all 4)"
    )
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel threads (default: 1 = sequential)")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file, args.n_prompts)
    modes = args.modes
    total = len(prompts) * len(SEEDS) * len(modes)

    print("=" * 90)
    print("RQ2: DOES STRUCTURED PLANNING IMPROVE CROSS-MODAL ALIGNMENT?")
    print("=" * 90)
    print(f"  Prompts:     {len(prompts)}")
    print(f"  Seeds:       {SEEDS}")
    print(f"  Modes:       {modes}")
    print(f"  Total runs:  {total}")
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
            for mode in modes:
                tasks.append({
                    "prompt_id": pid,
                    "prompt_text": text,
                    "domain": domain,
                    "seed": seed,
                    "mode": mode,
                    "out_dir": str(out_path / f"{pid}_s{seed}_{mode}"),
                })

    results = []
    t_start = time.time()

    def _execute_task(task, idx):
        t0 = time.time()
        try:
            result = run_single(
                prompt_text=task["prompt_text"],
                seed=task["seed"],
                mode=task["mode"],
                out_dir=task["out_dir"],
            )
            elapsed = time.time() - t0
            msci_str = f"MSCI={result['msci']:.4f}" if result.get('msci') is not None else "MSCI=N/A"
            print(f"[{idx}/{total}] {task['prompt_id']} seed={task['seed']} mode={task['mode']}  {msci_str}  ({elapsed:.1f}s)")
            return {**task, **result}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{idx}/{total}] {task['prompt_id']} seed={task['seed']} mode={task['mode']}  ERROR: {str(e)[:60]}  ({elapsed:.1f}s)")
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
    results_path = out_path / "rq2_results.json"
    with results_path.open("w") as f:
        json.dump({
            "experiment": "RQ2: Planning Effect",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "n_prompts": len(prompts),
                "seeds": SEEDS,
                "modes": modes,
                "total_runs": total,
                "total_time_sec": round(total_time, 1),
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    _print_quick_summary(results, modes)

    return 0


def _print_quick_summary(results: List[Dict[str, Any]], modes: List[str]) -> None:
    """Print quick MSCI summary by mode."""
    print("\n" + "=" * 90)
    print("QUICK SUMMARY")
    print("=" * 90)

    for mode in modes:
        scores = [
            r["msci"] for r in results
            if r.get("msci") is not None and r["mode"] == mode
        ]
        if scores:
            print(
                f"  {mode:<18}  N={len(scores):>3}  "
                f"mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  "
                f"median={np.median(scores):.4f}"
            )

    print("\nRun: python scripts/analyze_results.py runs/rq2/rq2_results.json")


if __name__ == "__main__":
    sys.exit(main())
