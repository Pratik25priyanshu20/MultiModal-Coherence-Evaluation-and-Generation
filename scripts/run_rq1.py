"""
RQ1: Is MSCI sensitive to controlled semantic perturbations?

Retrieval mode (default):
    30 prompts × 3 seeds × 3 conditions = 270 runs
    Uses CLIP-based image retrieval + CLAP-based audio retrieval.

Generative mode (--generative):
    30 prompts × N seeds × 3 conditions = 90-270 runs
    Uses SDXL for image generation + AudioLDM 2 for audio generation.
    Three-phase architecture to stay within 16GB RAM:
        Phase A: Load SDXL, generate ALL images, unload
        Phase B: Load AudioLDM 2, generate ALL audio, unload
        Phase C: Load CLIP+CLAP (lightweight), evaluate all triples

Each prompt is tested under:
- baseline: matched image + matched audio
- wrong_image: deliberately mismatched image (different domain prompt)
- wrong_audio: deliberately mismatched audio (different domain prompt)

Success criteria:
- MSCI(baseline) > MSCI(wrong_image), p < 0.05, Cohen's d > 0.5
- MSCI(baseline) > MSCI(wrong_audio), p < 0.05, Cohen's d > 0.5

Usage:
    python scripts/run_rq1.py --skip-text                          # Retrieval mode
    python scripts/run_rq1.py --generative --device mps            # Generative (3 seeds)
    python scripts/run_rq1.py --generative --device mps --seeds 1  # Generative (1 seed, faster)
    python scripts/run_rq1.py --n-prompts 5 --generative --device mps  # Quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

ALL_SEEDS = [42, 123, 7]
CONDITIONS = ["baseline", "wrong_image", "wrong_audio"]

# Domain mapping for perturbation prompt selection
DOMAIN_GROUPS = {
    "nature": ["nature"],
    "urban": ["urban"],
    "water": ["water"],
    "mixed": ["mixed"],
}

# For mismatched prompts, pick from a different domain
MISMATCH_DOMAINS = {
    "nature": "urban",
    "urban": "water",
    "water": "nature",
    "mixed": "urban",
}


def load_prompts(path: str, n: int = 30) -> List[Dict[str, str]]:
    with open(path) as f:
        data = json.load(f)
    prompts = data["prompts"][:n]
    return prompts


def _pick_mismatched_prompt(
    prompts: List[Dict[str, str]],
    current_prompt: Dict[str, str],
    seed: int,
) -> Dict[str, str]:
    """
    Select a prompt from a different domain for perturbation.

    For generative mode: instead of swapping retrieved files, we generate
    an image/audio from a semantically mismatched prompt.

    Args:
        prompts: All experiment prompts
        current_prompt: The prompt being perturbed
        seed: For deterministic selection

    Returns:
        A prompt dict from a different domain
    """
    target_domain = MISMATCH_DOMAINS.get(current_prompt["domain"], "urban")
    candidates = [p for p in prompts if p["domain"] == target_domain and p["id"] != current_prompt["id"]]
    if not candidates:
        # Fallback: any prompt from a different domain
        candidates = [p for p in prompts if p["domain"] != current_prompt["domain"]]
    if not candidates:
        # Last resort: any other prompt
        candidates = [p for p in prompts if p["id"] != current_prompt["id"]]

    rng = np.random.default_rng(seed + hash(current_prompt["id"]) % (2**31))
    return candidates[rng.integers(len(candidates))]


# ═══════════════════════════════════════════════════════════════
# RETRIEVAL MODE (original)
# ═══════════════════════════════════════════════════════════════


def run_single_retrieval(
    prompt_text: str,
    seed: int,
    condition: str,
    out_dir: str,
    skip_text: bool,
) -> Dict[str, Any]:
    """Run a single retrieval-based experimental condition."""
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
            "image_backend": "retrieval",
            "audio_backend": "retrieval",
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
            "image_backend": "retrieval",
            "audio_backend": "retrieval",
            "perturbation": bundle.meta.get("perturbation", {}),
        }


# ═══════════════════════════════════════════════════════════════
# GENERATIVE MODE (new — three-phase architecture)
# ═══════════════════════════════════════════════════════════════


def _build_generative_tasks(
    prompts: List[Dict[str, str]],
    seeds: List[int],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Build task list for generative mode.

    For each (prompt, seed, condition), determines:
    - image_prompt: which prompt to use for image generation
    - audio_prompt: which prompt to use for audio generation

    baseline: image_prompt = audio_prompt = original prompt
    wrong_image: image_prompt = mismatched prompt from different domain
    wrong_audio: audio_prompt = mismatched prompt from different domain
    """
    tasks = []
    for prompt_info in prompts:
        pid = prompt_info["id"]
        text = prompt_info["text"]
        domain = prompt_info["domain"]
        for seed in seeds:
            for condition in CONDITIONS:
                # Determine which prompts to use for generation
                image_prompt = text
                audio_prompt = text
                perturbation = {"applied": condition}

                if condition == "wrong_image":
                    mismatch = _pick_mismatched_prompt(prompts, prompt_info, seed)
                    image_prompt = mismatch["text"]
                    perturbation["mismatched_prompt"] = mismatch["text"]
                    perturbation["mismatched_domain"] = mismatch["domain"]
                    perturbation["mismatched_id"] = mismatch["id"]

                elif condition == "wrong_audio":
                    mismatch = _pick_mismatched_prompt(prompts, prompt_info, seed)
                    audio_prompt = mismatch["text"]
                    perturbation["mismatched_prompt"] = mismatch["text"]
                    perturbation["mismatched_domain"] = mismatch["domain"]
                    perturbation["mismatched_id"] = mismatch["id"]

                task_id = f"{pid}_s{seed}_{condition}"
                tasks.append({
                    "prompt_id": pid,
                    "prompt_text": text,
                    "domain": domain,
                    "seed": seed,
                    "condition": condition,
                    "image_prompt": image_prompt,
                    "audio_prompt": audio_prompt,
                    "perturbation": perturbation,
                    "image_path": str(out_dir / "images" / f"{task_id}.png"),
                    "audio_path": str(out_dir / "audio" / f"{task_id}.wav"),
                })
    return tasks


def run_generative(
    prompts: List[Dict[str, str]],
    seeds: List[int],
    out_dir: Path,
    device: str = "mps",
    num_inference_steps: int = 30,
    audio_duration: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Run generative RQ1 experiment with three-phase architecture.

    Phase A: Load SDXL → generate all images → unload (~6GB peak)
    Phase B: Load AudioLDM 2 → generate all audio → unload (~4GB peak)
    Phase C: Load CLIP+CLAP → evaluate all triples (~2GB peak)

    This sequential loading is critical for the 16GB RAM constraint.
    """
    tasks = _build_generative_tasks(prompts, seeds, out_dir)
    total = len(tasks)

    # Create output directories
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    # ── Phase A: Image Generation ──────────────────────────
    print("\n" + "─" * 90)
    print("PHASE A: IMAGE GENERATION (SDXL)")
    print("─" * 90)

    # Deduplicate: same image_prompt + seed → same image
    image_jobs = {}
    for task in tasks:
        key = (task["image_prompt"], task["seed"])
        if key not in image_jobs:
            image_jobs[key] = task["image_path"]

    print(f"  Unique images to generate: {len(image_jobs)} (of {total} tasks)")
    t_phase_a = time.time()

    from src.generators.image.generator_hybrid import HybridImageGenerator

    img_gen = HybridImageGenerator(
        force_sd=True,
        sd_model="sdxl",
        device=device,
        num_inference_steps=num_inference_steps,
    )

    if img_gen._sd_pipe is None:
        print("  WARNING: SD pipeline failed to load. Check GPU/MPS availability.")
        print(f"  Error: {img_gen._sd_error}")
        return []

    image_results = {}
    for i, ((prompt, seed), img_path) in enumerate(image_jobs.items(), 1):
        t0 = time.time()
        try:
            result = img_gen.generate(prompt=prompt, out_path=img_path, seed=seed)
            image_results[(prompt, seed)] = result
            elapsed = time.time() - t0
            print(f"  [{i}/{len(image_jobs)}] {result.backend} → {Path(img_path).name}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(image_jobs)}] ERROR: {str(e)[:80]}  ({elapsed:.1f}s)")
            image_results[(prompt, seed)] = None

    img_gen.unload()
    print(f"  Phase A complete: {time.time() - t_phase_a:.0f}s. SDXL unloaded.")

    # ── Phase B: Audio Generation ──────────────────────────
    print("\n" + "─" * 90)
    print("PHASE B: AUDIO GENERATION (AudioLDM 2)")
    print("─" * 90)

    audio_jobs = {}
    for task in tasks:
        key = (task["audio_prompt"], task["seed"])
        if key not in audio_jobs:
            audio_jobs[key] = task["audio_path"]

    print(f"  Unique audio clips to generate: {len(audio_jobs)} (of {total} tasks)")
    t_phase_b = time.time()

    from src.generators.audio.generator import AudioGenerator

    aud_gen = AudioGenerator(device=device, force_audioldm=True)

    if aud_gen._audioldm_pipe is None:
        print("  WARNING: AudioLDM 2 failed to load. Check availability.")
        print(f"  Error: {aud_gen._audioldm_error}")
        print("  Falling back to deterministic ambient generator.")

    audio_results = {}
    for i, ((prompt, seed), aud_path) in enumerate(audio_jobs.items(), 1):
        t0 = time.time()
        Path(aud_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            result = aud_gen.generate(prompt=prompt, out_path=aud_path, duration_sec=audio_duration, seed=seed)
            audio_results[(prompt, seed)] = result
            elapsed = time.time() - t0
            print(f"  [{i}/{len(audio_jobs)}] {result.backend} → {Path(aud_path).name}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(audio_jobs)}] ERROR: {str(e)[:80]}  ({elapsed:.1f}s)")
            audio_results[(prompt, seed)] = None

    aud_gen.unload()
    print(f"  Phase B complete: {time.time() - t_phase_b:.0f}s. AudioLDM 2 unloaded.")

    # ── Phase C: Evaluation ────────────────────────────────
    print("\n" + "─" * 90)
    print("PHASE C: COHERENCE EVALUATION (CLIP + CLAP)")
    print("─" * 90)
    t_phase_c = time.time()

    from src.coherence.coherence_engine import evaluate_coherence

    results = []
    for i, task in enumerate(tasks, 1):
        t0 = time.time()

        img_key = (task["image_prompt"], task["seed"])
        aud_key = (task["audio_prompt"], task["seed"])
        img_result = image_results.get(img_key)
        aud_result = audio_results.get(aud_key)

        if img_result is None or aud_result is None:
            print(f"  [{i}/{total}] {task['prompt_id']} cond={task['condition']}  SKIPPED (generation failed)")
            results.append({**task, "error": "generation failed"})
            continue

        try:
            eval_out = evaluate_coherence(
                text=task["prompt_text"],
                image_path=img_result.image_path,
                audio_path=aud_result.audio_path,
            )
            scores = eval_out.get("scores", {})
            elapsed = time.time() - t0
            msci_str = f"MSCI={scores.get('msci', 0):.4f}" if scores.get('msci') is not None else "MSCI=N/A"
            print(f"  [{i}/{total}] {task['prompt_id']} seed={task['seed']} cond={task['condition']}  {msci_str}  ({elapsed:.1f}s)")

            results.append({
                "prompt_id": task["prompt_id"],
                "prompt_text": task["prompt_text"],
                "domain": task["domain"],
                "seed": task["seed"],
                "condition": task["condition"],
                "msci": scores.get("msci"),
                "st_i": scores.get("st_i"),
                "st_a": scores.get("st_a"),
                "si_a": scores.get("si_a"),
                "image_path": img_result.image_path,
                "audio_path": aud_result.audio_path,
                "image_backend": img_result.backend,
                "audio_backend": aud_result.backend,
                "perturbation": task["perturbation"],
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{i}/{total}] {task['prompt_id']} cond={task['condition']}  ERROR: {str(e)[:60]}  ({elapsed:.1f}s)")
            results.append({**task, "error": str(e)})

    print(f"  Phase C complete: {time.time() - t_phase_c:.0f}s")
    return results


def main():
    parser = argparse.ArgumentParser(description="RQ1: MSCI Sensitivity Experiment")
    parser.add_argument("--skip-text", action="store_true",
                        help="Skip text generation (use prompt directly)")
    parser.add_argument("--generative", action="store_true",
                        help="Use generative models (SDXL + AudioLDM 2) instead of retrieval")
    parser.add_argument("--device", default="cpu",
                        help="Device for generative models (cpu, mps, cuda)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds: 1 for quick run, 3 for full (default: 3)")
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--prompts-file", default="data/prompts/experiment_prompts.json")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: runs/rq1 or runs/rq1_gen)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Parallel threads for retrieval mode (default: 1)")
    parser.add_argument("--num-inference-steps", type=int, default=30,
                        help="SD inference steps (default: 30)")
    parser.add_argument("--audio-duration", type=float, default=6.0,
                        help="Audio duration in seconds (default: 6.0)")
    args = parser.parse_args()

    # Set default output dir based on mode
    if args.out_dir is None:
        args.out_dir = "runs/rq1_gen" if args.generative else "runs/rq1"

    seeds = ALL_SEEDS[:args.seeds]
    prompts = load_prompts(args.prompts_file, args.n_prompts)
    total = len(prompts) * len(seeds) * len(CONDITIONS)
    mode_label = "GENERATIVE (SDXL + AudioLDM 2)" if args.generative else "RETRIEVAL"

    print("=" * 90)
    print("RQ1: IS MSCI SENSITIVE TO CONTROLLED SEMANTIC PERTURBATIONS?")
    print("=" * 90)
    print(f"  Mode:        {mode_label}")
    print(f"  Prompts:     {len(prompts)}")
    print(f"  Seeds:       {seeds}")
    print(f"  Conditions:  {CONDITIONS}")
    print(f"  Total runs:  {total}")
    if args.generative:
        print(f"  Device:      {args.device}")
        print(f"  SD steps:    {args.num_inference_steps}")
        print(f"  Audio dur:   {args.audio_duration}s")
    else:
        print(f"  Skip text:   {args.skip_text}")
        print(f"  Parallel:    {args.parallel}")
    print("=" * 90)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if args.generative:
        # ── GENERATIVE MODE: three-phase architecture ──
        results = run_generative(
            prompts=prompts,
            seeds=seeds,
            out_dir=out_path,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            audio_duration=args.audio_duration,
        )
    else:
        # ── RETRIEVAL MODE: original logic ──
        if args.parallel > 1:
            print("Pre-loading shared embedder for parallel execution...")
            from src.embeddings.shared_embedder import get_shared_embedder
            get_shared_embedder()
            print("Shared embedder ready.")

        tasks = []
        for prompt_info in prompts:
            pid = prompt_info["id"]
            text = prompt_info["text"]
            domain = prompt_info["domain"]
            for seed in seeds:
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

        def _execute_task(task, idx):
            t0 = time.time()
            try:
                result = run_single_retrieval(
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
    results_path = out_path / "rq1_gen_results.json" if args.generative else out_path / "rq1_results.json"
    with results_path.open("w") as f:
        json.dump({
            "experiment": f"RQ1: MSCI Sensitivity ({mode_label})",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "mode": "generative" if args.generative else "retrieval",
                "n_prompts": len(prompts),
                "seeds": seeds,
                "conditions": CONDITIONS,
                "skip_text": args.skip_text if not args.generative else False,
                "device": args.device if args.generative else "cpu",
                "total_runs": total,
                "total_time_sec": round(total_time, 1),
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    _print_quick_summary(results, total)

    return 0


def _print_quick_summary(results: List[Dict[str, Any]], total: int = 0) -> None:
    """Print quick MSCI summary by condition."""
    print("\n" + "=" * 90)
    print("QUICK SUMMARY")
    print("=" * 90)

    for condition in CONDITIONS:
        scores = [
            r["msci"] for r in results
            if r.get("msci") is not None and r.get("condition") == condition
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
        if r.get("condition") == "baseline" and r.get("msci") is not None:
            key = (r["prompt_id"], r["seed"])
            baseline_scores[key] = r["msci"]

    for condition in ["wrong_image", "wrong_audio"]:
        deltas = []
        for r in results:
            if r.get("condition") == condition and r.get("msci") is not None:
                key = (r["prompt_id"], r["seed"])
                if key in baseline_scores:
                    deltas.append(baseline_scores[key] - r["msci"])
        if deltas:
            print(
                f"\n  Δ(baseline - {condition}):  "
                f"mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}  "
                f"all_positive={all(d > 0 for d in deltas)}"
            )

    results_file = "rq1_gen_results.json" if any(r.get("image_backend") not in (None, "retrieval") for r in results) else "rq1_results.json"
    print(f"\nRun: python scripts/analyze_results.py {Path(results[0].get('image_path', 'runs/rq1')).parent.parent / results_file if results else 'runs/rq1/rq1_results.json'}")


if __name__ == "__main__":
    sys.exit(main())
