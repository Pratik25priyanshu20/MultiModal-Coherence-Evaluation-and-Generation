#!/usr/bin/env python3
"""
RQ2 Hybrid: Planning strategies with SDXL-generated images + CLAP-retrieved audio.

Tests whether structured planning improves cross-modal alignment when images
are generated (SDXL) rather than retrieved from a fixed index.

Design: 10 prompts × 1 seed × 4 modes = 40 runs
  - direct: SDXL(original_prompt) + CLAP(original_prompt)
  - planner: SDXL(planned_image_prompt) + CLAP(planned_audio_prompt)
  - council: SDXL(planned_image_prompt) + CLAP(planned_audio_prompt)
  - extended_prompt: SDXL(planned_image_prompt) + CLAP(planned_audio_prompt)

MSCI is computed using the original prompt as text anchor (skip-text mode)
so the only variation is in image generation and audio retrieval quality.

Three-phase architecture (fits in 16 GB RAM):
  Phase A: Planning (Ollama) — generate planned prompts for each mode
  Phase B: SDXL image generation — load once, generate all 40 images, unload
  Phase C: CLAP retrieval + MSCI evaluation

Usage:
    python scripts/run_rq2_hybrid.py
    python scripts/run_rq2_hybrid.py --n-prompts 10 --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import gc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALL_MODES = ["direct", "planner", "council", "extended_prompt"]

# Stratified prompt selection: pick balanced set across domains
STRATIFIED_IDS = {
    10: ["nat_01", "nat_03", "nat_07",
         "urb_01", "urb_04", "urb_07",
         "wat_01", "wat_05",
         "mix_01", "mix_05"],
}


def select_prompts(all_prompts: list, n: int) -> list:
    """Select n prompts with domain balance."""
    if n in STRATIFIED_IDS:
        target_ids = set(STRATIFIED_IDS[n])
        selected = [p for p in all_prompts if p["id"] in target_ids]
        # Fill if any IDs not found
        if len(selected) < n:
            remaining = [p for p in all_prompts if p["id"] not in target_ids]
            selected.extend(remaining[:n - len(selected)])
        return selected
    # Fallback: stratified round-robin
    by_domain = {}
    for p in all_prompts:
        by_domain.setdefault(p["domain"], []).append(p)
    selected = []
    domains = sorted(by_domain.keys())
    idx = 0
    while len(selected) < n and idx < max(len(v) for v in by_domain.values()):
        for d in domains:
            if idx < len(by_domain[d]) and len(selected) < n:
                selected.append(by_domain[d][idx])
        idx += 1
    return selected


def run_planning(prompt_text: str, mode: str) -> Dict[str, str]:
    """Run planning to get image_prompt, audio_prompt, text_prompt."""
    if mode == "direct":
        return {
            "image_prompt": prompt_text,
            "audio_prompt": prompt_text,
            "text_prompt": prompt_text,
        }
    elif mode == "planner":
        from src.planner.unified_planner import UnifiedPlannerLLM
        from src.planner.schema_to_text import plan_to_prompts
        planner = UnifiedPlannerLLM()
        plan = planner.plan(prompt_text)
        return plan_to_prompts(plan)
    elif mode == "council":
        from src.planner.council import SemanticPlanningCouncil
        from src.planner.unified_planner import UnifiedPlannerLLM
        from src.planner.schema_to_text import plan_to_prompts
        planner_a = UnifiedPlannerLLM()
        planner_a.name = "PlannerA"
        planner_b = UnifiedPlannerLLM()
        planner_b.name = "PlannerB"
        planner_c = UnifiedPlannerLLM()
        planner_c.name = "PlannerC"
        council = SemanticPlanningCouncil(planner_a, planner_b, planner_c)
        council_result = council.run(prompt_text)
        return plan_to_prompts(council_result.merged_plan)
    elif mode == "extended_prompt":
        from src.planner.extended_prompt_planner import ExtendedPromptPlanner
        from src.planner.schema_to_text import plan_to_prompts
        planner = ExtendedPromptPlanner()
        plan = planner.plan(prompt_text)
        return plan_to_prompts(plan)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="RQ2 Hybrid: Planning with SDXL + CLAP retrieval")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--prompts-file", default="data/prompts/experiment_prompts.json")
    parser.add_argument("--out-dir", default="runs/rq2_hybrid")
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=ALL_MODES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load and select prompts
    with open(args.prompts_file) as f:
        all_prompts = json.load(f)["prompts"]
    prompts = select_prompts(all_prompts, args.n_prompts)
    modes = args.modes
    seed = args.seed
    total = len(prompts) * len(modes)

    print("=" * 90)
    print("RQ2 HYBRID: PLANNING STRATEGIES WITH SDXL IMAGES + CLAP RETRIEVAL")
    print("=" * 90)
    print(f"  Prompts:     {len(prompts)} (stratified)")
    print(f"  Domains:     {dict(sorted({d: sum(1 for p in prompts if p['domain'] == d) for d in set(p['domain'] for p in prompts)}.items()))}")
    print(f"  Seed:        {seed}")
    print(f"  Modes:       {modes}")
    print(f"  Total runs:  {total}")
    print(f"  Device:      {args.device}")
    print(f"  Steps:       {args.num_inference_steps}")
    print("=" * 90)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    img_dir = out_path / "images"
    img_dir.mkdir(exist_ok=True)

    t_start = time.time()

    # ──────────────────────────────────────────────────────
    # PHASE A: Planning (Ollama)
    # ──────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("PHASE A: Planning (Ollama)")
    print(f"{'─'*90}")

    tasks = []  # Each task: {prompt_id, prompt_text, domain, mode, image_prompt, audio_prompt}
    planning_errors = 0

    for i, prompt_info in enumerate(prompts):
        pid = prompt_info["id"]
        text = prompt_info["text"]
        domain = prompt_info["domain"]

        for mode in modes:
            t0 = time.time()
            try:
                planned = run_planning(text, mode)
                elapsed = time.time() - t0
                img_prompt = planned.get("image_prompt", text)
                aud_prompt = planned.get("audio_prompt", text)
                print(f"  [{pid}] {mode:<18} planned in {elapsed:.1f}s  img_prompt={img_prompt[:60]}...")

                tasks.append({
                    "prompt_id": pid,
                    "prompt_text": text,
                    "domain": domain,
                    "seed": seed,
                    "mode": mode,
                    "image_prompt": img_prompt,
                    "audio_prompt": aud_prompt,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{pid}] {mode:<18} PLANNING ERROR: {str(e)[:60]}  ({elapsed:.1f}s)")
                planning_errors += 1
                # Fallback to direct prompt
                tasks.append({
                    "prompt_id": pid,
                    "prompt_text": text,
                    "domain": domain,
                    "seed": seed,
                    "mode": mode,
                    "image_prompt": text,
                    "audio_prompt": text,
                    "planning_error": str(e),
                })

    planning_time = time.time() - t_start
    print(f"\nPhase A complete: {len(tasks)} tasks planned in {planning_time:.0f}s ({planning_errors} errors)")

    # ──────────────────────────────────────────────────────
    # PHASE B: SDXL Image Generation
    # ──────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("PHASE B: SDXL Image Generation")
    print(f"{'─'*90}")

    t_phase_b = time.time()

    # Build unique image generation jobs (same prompt+seed = same image)
    image_jobs = {}  # (image_prompt, seed) → output_path
    for task in tasks:
        key = (task["image_prompt"], task["seed"])
        if key not in image_jobs:
            safe_name = f"{task['prompt_id']}_{task['mode']}_s{seed}"
            image_jobs[key] = str(img_dir / f"{safe_name}.png")

    print(f"  Unique images to generate: {len(image_jobs)}")
    print(f"  Estimated time: ~{len(image_jobs) * 4:.0f} min")

    # Load SDXL
    import torch
    from src.generators.image.generator_hybrid import HybridImageGenerator

    print("  Loading SDXL...")
    gen = HybridImageGenerator(
        force_sd=True,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
    )
    if gen._sd_pipe is None:
        print(f"  ERROR: SDXL failed to load: {gen._sd_error}")
        print("  Cannot proceed without SDXL. Exiting.")
        return 1
    print(f"  SDXL loaded ({gen._sd_backend_name}).")

    # Generate all images using the public generate() API
    image_results = {}  # (image_prompt, seed) → actual_path
    for idx, ((img_prompt, s), out_file) in enumerate(image_jobs.items()):
        t0 = time.time()
        try:
            result = gen.generate(
                prompt=img_prompt,
                out_path=out_file,
                seed=s,
            )
            image_results[(img_prompt, s)] = result.image_path
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(image_jobs)}] Generated: {Path(result.image_path).name}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(image_jobs)}] ERROR: {str(e)[:80]}  ({elapsed:.1f}s)")
            image_results[(img_prompt, s)] = None

    # Unload SDXL
    print("  Unloading SDXL...")
    gen.unload()
    del gen
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    elif args.device == "cuda":
        torch.cuda.empty_cache()

    phase_b_time = time.time() - t_phase_b
    print(f"\nPhase B complete: {sum(1 for v in image_results.values() if v)} images in {phase_b_time:.0f}s ({phase_b_time/60:.1f}min)")

    # ──────────────────────────────────────────────────────
    # PHASE C: CLAP Retrieval + MSCI Evaluation
    # ──────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("PHASE C: CLAP Retrieval + MSCI Evaluation")
    print(f"{'─'*90}")

    t_phase_c = time.time()

    from src.generators.audio.retrieval import retrieve_audio_with_metadata
    from src.coherence.coherence_engine import evaluate_coherence

    results = []
    for idx, task in enumerate(tasks):
        t0 = time.time()

        # Get SDXL image
        img_key = (task["image_prompt"], task["seed"])
        image_path = image_results.get(img_key)

        if not image_path or not Path(image_path).exists():
            print(f"  [{idx+1}/{total}] {task['prompt_id']} mode={task['mode']}  SKIPPED (no image)")
            results.append({**task, "error": "no image"})
            continue

        # Retrieve audio via CLAP
        try:
            audio_result = retrieve_audio_with_metadata(
                prompt=task["audio_prompt"], min_similarity=0.10
            )
            audio_path = audio_result.audio_path
            audio_sim = audio_result.similarity
        except Exception as e:
            print(f"  [{idx+1}/{total}] {task['prompt_id']} mode={task['mode']}  AUDIO ERROR: {str(e)[:60]}")
            results.append({**task, "error": f"audio retrieval: {e}"})
            continue

        # Evaluate MSCI using ORIGINAL prompt as text anchor (skip-text)
        try:
            eval_out = evaluate_coherence(
                text=task["prompt_text"],  # Original prompt, not planned text
                image_path=str(image_path),
                audio_path=str(audio_path),
            )
            scores = eval_out.get("scores", {})
            elapsed = time.time() - t0
            msci = scores.get("msci", 0)
            print(f"  [{idx+1}/{total}] {task['prompt_id']} mode={task['mode']:<18}  MSCI={msci:.4f}  st_i={scores.get('st_i', 0):.3f}  st_a={scores.get('st_a', 0):.3f}  ({elapsed:.1f}s)")

            results.append({
                **task,
                "msci": scores.get("msci"),
                "st_i": scores.get("st_i"),
                "st_a": scores.get("st_a"),
                "si_a": scores.get("si_a"),
                "image_path": str(image_path),
                "audio_path": str(audio_path),
                "audio_sim": audio_sim,
                "image_backend": "sdxl",
                "audio_backend": "retrieval",
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{total}] {task['prompt_id']} mode={task['mode']}  EVAL ERROR: {str(e)[:60]}  ({elapsed:.1f}s)")
            results.append({**task, "error": str(e)})

    phase_c_time = time.time() - t_phase_c
    total_time = time.time() - t_start

    # ──────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────
    results_path = out_path / "rq2_hybrid_results.json"
    with results_path.open("w") as f:
        json.dump({
            "experiment": "RQ2: Planning Effect (HYBRID: SDXL images + CLAP retrieval audio)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "mode": "hybrid",
                "image_backend": "sdxl",
                "audio_backend": "clap_retrieval",
                "text_mode": "skip-text (original prompt)",
                "n_prompts": len(prompts),
                "seed": seed,
                "modes": modes,
                "total_runs": total,
                "num_inference_steps": args.num_inference_steps,
                "device": args.device,
                "planning_time_sec": round(planning_time, 1),
                "generation_time_sec": round(phase_b_time, 1),
                "evaluation_time_sec": round(phase_c_time, 1),
                "total_time_sec": round(total_time, 1),
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Planning: {planning_time:.0f}s  |  SDXL: {phase_b_time:.0f}s  |  Eval: {phase_c_time:.0f}s")

    # ──────────────────────────────────────────────────────
    # Quick summary
    # ──────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("QUICK SUMMARY")
    print(f"{'='*90}")

    for mode in modes:
        scores = [r["msci"] for r in results if r.get("msci") is not None and r["mode"] == mode]
        if scores:
            print(f"  {mode:<18}  N={len(scores):>3}  mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  median={np.median(scores):.4f}")

    # Paired comparisons vs direct
    direct_scores = {}
    for r in results:
        if r["mode"] == "direct" and r.get("msci") is not None:
            direct_scores[r["prompt_id"]] = r["msci"]

    if direct_scores:
        print(f"\n  Paired differences vs direct (N={len(direct_scores)}):")
        for mode in modes:
            if mode == "direct":
                continue
            deltas = []
            for r in results:
                if r["mode"] == mode and r.get("msci") is not None:
                    if r["prompt_id"] in direct_scores:
                        deltas.append(r["msci"] - direct_scores[r["prompt_id"]])
            if deltas:
                mean_d = np.mean(deltas)
                std_d = np.std(deltas, ddof=1) if len(deltas) > 1 else 0
                cohens_d = mean_d / std_d if std_d > 0 else 0
                print(f"    {mode:<18}  Δ={mean_d:+.4f}  std={std_d:.4f}  d={cohens_d:+.2f}")

    print(f"\nRun: python scripts/analyze_results.py {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
