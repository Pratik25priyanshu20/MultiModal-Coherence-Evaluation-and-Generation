"""
PHASE 3 — Sanity Check: 5 prompts × 1 seed × 6 conditions

Verifies that the pipeline is experimentally sound before large-scale runs.

Checks:
1. Image retrieval: forest prompt → forest image (not neon city)
2. Perturbation quality: wrong_image/wrong_audio are genuinely mismatched
3. Audio differentiation: different prompts → different audio
4. MSCI sensitivity: MSCI drops under perturbation conditions

Run:
    python scripts/sanity_check.py

Requires:
- Embedding indexes built (run scripts/build_embedding_indexes.py first)
- Ollama running locally for text generation (optional: --skip-text)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity
from src.coherence.coherence_engine import evaluate_coherence


# ── Test prompts (5 diverse, spanning domains) ──────────────────────────
SANITY_PROMPTS = [
    "A peaceful forest at dawn with birdsong",
    "A bustling city street at night with neon lights",
    "Ocean waves crashing on a sandy beach at sunset",
    "A rainy day in a European city with cobblestone streets",
    "A snowy mountain peak under a clear blue sky",
]

# Expected domains for each prompt (for retrieval check)
EXPECTED_DOMAINS = {
    "A peaceful forest at dawn with birdsong": "nature",
    "A bustling city street at night with neon lights": "urban",
    "Ocean waves crashing on a sandy beach at sunset": "water",
    "A rainy day in a European city with cobblestone streets": "urban",
    "A snowy mountain peak under a clear blue sky": "nature",
}

# Domains that would indicate a BAD retrieval for each prompt
WRONG_DOMAINS = {
    "nature": {"urban"},
    "urban": {"nature", "water"},
    "water": {"urban"},
}

SEED = 42
MODES = ["direct", "planner"]
CONDITIONS = ["baseline", "wrong_image", "wrong_audio"]


@dataclass
class SanityResult:
    prompt: str
    mode: str
    condition: str
    msci: Optional[float]
    st_i: Optional[float]
    st_a: Optional[float]
    si_a: Optional[float]
    image_path: Optional[str]
    image_domain: Optional[str]
    image_similarity: Optional[float]
    retrieval_failed: bool
    audio_path: Optional[str]
    audio_backend: Optional[str]
    perturbation: Dict[str, Any]
    error: Optional[str] = None


def check_1_retrieval_quality(results: List[SanityResult]) -> tuple[bool, list[str]]:
    """
    CHECK 1: Image retrieval returns domain-appropriate images.
    Forest prompts should NOT retrieve city images and vice versa.
    """
    issues = []
    baselines = [r for r in results if r.condition == "baseline" and r.error is None]

    for r in baselines:
        expected = EXPECTED_DOMAINS.get(r.prompt)
        if not expected:
            continue

        # Check if the retrieved image domain is compatible
        wrong = WRONG_DOMAINS.get(expected, set())
        if r.image_domain and r.image_domain in wrong:
            issues.append(
                f"  FAIL: \"{r.prompt[:50]}\" retrieved {r.image_domain} image "
                f"(expected {expected}): {Path(r.image_path).name if r.image_path else 'N/A'}"
            )
        elif r.retrieval_failed:
            issues.append(
                f"  WARN: \"{r.prompt[:50]}\" retrieval_failed=True "
                f"(sim={r.image_similarity:.4f})"
            )

    passed = len(issues) == 0
    return passed, issues


def check_2_perturbation_quality(results: List[SanityResult]) -> tuple[bool, list[str]]:
    """
    CHECK 2: Perturbations use genuinely different modalities.
    wrong_image should use a different domain image.
    wrong_audio should use a different audio file.
    """
    issues = []

    for r in results:
        if r.error is not None:
            continue

        if r.condition == "wrong_image":
            orig = r.perturbation.get("original_image")
            repl = r.perturbation.get("replacement_image")
            orig_dom = r.perturbation.get("original_domain")
            repl_dom = r.perturbation.get("replacement_domain")

            if not repl:
                issues.append(
                    f"  FAIL: \"{r.prompt[:40]}\" [{r.mode}] wrong_image had no replacement"
                )
            elif orig == repl:
                issues.append(
                    f"  FAIL: \"{r.prompt[:40]}\" [{r.mode}] wrong_image same as original"
                )
            elif orig_dom and repl_dom and orig_dom == repl_dom and orig_dom != "other":
                issues.append(
                    f"  WARN: \"{r.prompt[:40]}\" [{r.mode}] wrong_image same domain "
                    f"({orig_dom}→{repl_dom})"
                )

        if r.condition == "wrong_audio":
            orig = r.perturbation.get("original_audio")
            repl = r.perturbation.get("replacement_audio")

            if not repl:
                issues.append(
                    f"  FAIL: \"{r.prompt[:40]}\" [{r.mode}] wrong_audio had no replacement"
                )
            elif orig == repl:
                issues.append(
                    f"  FAIL: \"{r.prompt[:40]}\" [{r.mode}] wrong_audio same as original"
                )

    passed = len([i for i in issues if "FAIL" in i]) == 0
    return passed, issues


def check_3_audio_differentiation(results: List[SanityResult]) -> tuple[bool, list[str]]:
    """
    CHECK 3: Audio retrieval and differentiation.
    - Checks that not ALL prompts retrieve the exact same audio file
    - Checks that baseline st_a scores show meaningful signal (not all near zero)
    - WARN (not FAIL) if some prompts share audio due to limited pool
    """
    issues = []
    baselines = [r for r in results if r.condition == "baseline" and r.error is None]

    if len(baselines) < 2:
        issues.append("  WARN: Not enough baseline results to compare")
        return True, issues

    # Check 3a: Audio file diversity (are different files being retrieved?)
    audio_paths = [r.audio_path for r in baselines if r.audio_path]
    unique_paths = set(audio_paths)
    diversity = len(unique_paths) / max(len(audio_paths), 1)

    if diversity < 0.2:
        issues.append(
            f"  WARN: Low audio diversity ({len(unique_paths)}/{len(audio_paths)} unique files). "
            "Audio pool may be too small for fine-grained differentiation."
        )

    if len(unique_paths) == 1 and len(audio_paths) > 1:
        issues.append(
            f"  WARN: All {len(audio_paths)} prompts retrieved the same audio file: "
            f"{Path(audio_paths[0]).name}"
        )

    # Check 3b: st_a signal quality (are baseline text-audio scores meaningful?)
    sta_scores = [r.st_a for r in baselines if r.st_a is not None]
    if sta_scores:
        avg_sta = np.mean(sta_scores)
        min_sta = min(sta_scores)
        max_sta = max(sta_scores)
        sta_range = max_sta - min_sta

        if avg_sta < 0.05:
            issues.append(
                f"  FAIL: Baseline st_a scores near zero (avg={avg_sta:.4f}). "
                "Audio retrieval is not producing semantically meaningful matches."
            )
        elif sta_range < 0.01 and len(sta_scores) > 2:
            issues.append(
                f"  WARN: st_a scores have very low variance (range={sta_range:.4f}). "
                "Audio may not differentiate well across prompts."
            )
        else:
            issues.append(
                f"  INFO: st_a range={sta_range:.4f} avg={avg_sta:.4f} "
                f"(min={min_sta:.4f} max={max_sta:.4f})"
            )

    passed = len([i for i in issues if "FAIL" in i]) == 0
    return passed, issues


def check_4_msci_sensitivity(results: List[SanityResult]) -> tuple[bool, list[str]]:
    """
    CHECK 4: MSCI drops under perturbation conditions.
    baseline MSCI should be higher than wrong_image and wrong_audio MSCI.
    """
    issues = []

    for prompt in SANITY_PROMPTS:
        for mode in MODES:
            prompt_results = [
                r for r in results
                if r.prompt == prompt and r.mode == mode and r.error is None and r.msci is not None
            ]
            if len(prompt_results) < 2:
                continue

            baseline = [r for r in prompt_results if r.condition == "baseline"]
            wrong_img = [r for r in prompt_results if r.condition == "wrong_image"]
            wrong_aud = [r for r in prompt_results if r.condition == "wrong_audio"]

            if not baseline:
                continue
            b_msci = baseline[0].msci

            if wrong_img:
                wi_msci = wrong_img[0].msci
                delta = b_msci - wi_msci
                if delta < 0:
                    issues.append(
                        f"  WARN: \"{prompt[:35]}\" [{mode}] wrong_image MSCI INCREASED "
                        f"({b_msci:.4f}→{wi_msci:.4f}, delta={delta:+.4f})"
                    )

            if wrong_aud:
                wa_msci = wrong_aud[0].msci
                delta = b_msci - wa_msci
                if delta < 0:
                    issues.append(
                        f"  WARN: \"{prompt[:35]}\" [{mode}] wrong_audio MSCI INCREASED "
                        f"({b_msci:.4f}→{wa_msci:.4f}, delta={delta:+.4f})"
                    )

    # Overall: check that on AVERAGE perturbations lower MSCI
    baselines = [r.msci for r in results if r.condition == "baseline" and r.msci is not None]
    perturbeds = [r.msci for r in results if r.condition != "baseline" and r.msci is not None]

    if baselines and perturbeds:
        avg_base = np.mean(baselines)
        avg_pert = np.mean(perturbeds)
        if avg_pert >= avg_base:
            issues.append(
                f"  FAIL: Average perturbed MSCI ({avg_pert:.4f}) >= "
                f"baseline ({avg_base:.4f}) — metric not sensitive"
            )

    passed = len([i for i in issues if "FAIL" in i]) == 0
    return passed, issues


def run_pipeline_condition(
    prompt: str,
    mode: str,
    condition: str,
    seed: int,
    out_dir: str,
    skip_text: bool = False,
) -> SanityResult:
    """Run one prompt × mode × condition through the pipeline."""
    try:
        if skip_text:
            return _run_retrieval_only(prompt, mode, condition, seed, out_dir)
        else:
            return _run_full_pipeline(prompt, mode, condition, seed, out_dir)
    except Exception as e:
        return SanityResult(
            prompt=prompt,
            mode=mode,
            condition=condition,
            msci=None,
            st_i=None,
            st_a=None,
            si_a=None,
            image_path=None,
            image_domain=None,
            image_similarity=None,
            retrieval_failed=False,
            audio_path=None,
            audio_backend=None,
            perturbation={},
            error=str(e),
        )


def _run_full_pipeline(
    prompt: str, mode: str, condition: str, seed: int, out_dir: str,
) -> SanityResult:
    """Run through the full generate_and_evaluate pipeline."""
    from src.pipeline.generate_and_evaluate import generate_and_evaluate

    bundle = generate_and_evaluate(
        prompt=prompt,
        out_dir=out_dir,
        use_ollama=True,
        deterministic=True,
        seed=seed,
        mode=mode,
        condition=condition,
    )

    meta = bundle.meta
    scores = bundle.scores

    return SanityResult(
        prompt=prompt,
        mode=mode,
        condition=condition,
        msci=scores.get("msci"),
        st_i=scores.get("st_i"),
        st_a=scores.get("st_a"),
        si_a=scores.get("si_a"),
        image_path=bundle.image_path,
        image_domain=(meta.get("image_retrieval") or {}).get("domain"),
        image_similarity=(meta.get("image_retrieval") or {}).get("similarity"),
        retrieval_failed=(meta.get("image_retrieval") or {}).get("retrieval_failed", False),
        audio_path=bundle.audio_path,
        audio_backend=(meta.get("audio_backend") or {}).get("backend"),
        perturbation=meta.get("perturbation", {}),
    )


def _run_retrieval_only(
    prompt: str, mode: str, condition: str, seed: int, out_dir: str,
) -> SanityResult:
    """
    Run without text generation (no Ollama needed).
    Tests image retrieval + audio retrieval + MSCI computation directly.
    """
    from src.generators.image.generator_improved import generate_image_with_metadata
    from src.generators.audio.retrieval import retrieve_audio_with_metadata
    from src.pipeline.generate_and_evaluate import (
        _collect_all_image_paths,
        _collect_all_audio_paths,
        _select_mismatched_path,
        _infer_domain_from_path,
    )
    from src.utils.seed import set_global_seed

    set_global_seed(seed)

    # Use prompt directly as "generated text" for embedding purposes
    generated_text = prompt

    # Image retrieval (CLIP text → CLIP image)
    image_meta = None
    try:
        image_result = generate_image_with_metadata(prompt=prompt, min_similarity=0.20)
        image_path = image_result.image_path
        image_meta = {
            "similarity": image_result.similarity,
            "domain": image_result.domain,
            "retrieval_failed": image_result.retrieval_failed,
        }
    except Exception as e:
        image_path = None
        image_meta = {"error": str(e), "retrieval_failed": True}

    # Audio retrieval (CLAP text → CLAP audio)
    audio_backend = "retrieval"
    try:
        audio_result = retrieve_audio_with_metadata(prompt=prompt, min_similarity=0.10)
        audio_path = audio_result.audio_path
    except Exception:
        # Fallback to synthetic if no index
        from src.generators.audio.generator import generate_audio_with_metadata as gen_audio
        prompt_hash = abs(hash(prompt)) % 10000
        fallback_dir = Path(out_dir) / f"audio_{prompt_hash}_{mode}_{condition}"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fb_result = gen_audio(prompt=prompt, out_dir=str(fallback_dir), seed=seed)
        audio_path = fb_result.audio_path
        audio_backend = "fallback_ambient"

    # Perturbation
    perturbation_meta = {"applied": condition}
    if condition == "wrong_image" and image_path:
        orig_domain = _infer_domain_from_path(image_path)
        images = _collect_all_image_paths()
        replacement = _select_mismatched_path(
            images, exclude=str(image_path), original_domain=orig_domain, seed=seed,
        )
        if replacement:
            perturbation_meta["original_image"] = str(image_path)
            perturbation_meta["replacement_image"] = replacement
            perturbation_meta["original_domain"] = orig_domain
            perturbation_meta["replacement_domain"] = _infer_domain_from_path(replacement)
            image_path = replacement

    if condition == "wrong_audio":
        orig_domain = _infer_domain_from_path(audio_path)
        audios = _collect_all_audio_paths()
        replacement = _select_mismatched_path(
            audios, exclude=str(audio_path), original_domain=orig_domain, seed=seed,
        )
        if replacement:
            perturbation_meta["original_audio"] = str(audio_path)
            perturbation_meta["replacement_audio"] = replacement
            perturbation_meta["original_domain"] = orig_domain
            perturbation_meta["replacement_domain"] = _infer_domain_from_path(replacement)
            audio_path = replacement

    # MSCI computation via CoherenceEngine (uses correct embedding spaces)
    st_i = st_a = si_a = msci_val = None

    if image_path and Path(image_path).exists() and audio_path and Path(audio_path).exists():
        eval_out = evaluate_coherence(
            text=generated_text,
            image_path=str(image_path),
            audio_path=str(audio_path),
        )
        scores = eval_out.get("scores", {})
        st_i = scores.get("st_i")
        st_a = scores.get("st_a")
        si_a = scores.get("si_a")
        msci_val = scores.get("msci")

    return SanityResult(
        prompt=prompt,
        mode=mode,
        condition=condition,
        msci=msci_val,
        st_i=st_i,
        st_a=st_a,
        si_a=si_a,
        image_path=str(image_path) if image_path else None,
        image_domain=(image_meta or {}).get("domain"),
        image_similarity=(image_meta or {}).get("similarity"),
        retrieval_failed=(image_meta or {}).get("retrieval_failed", False),
        audio_path=str(audio_path),
        audio_backend=audio_backend,
        perturbation=perturbation_meta,
    )


def print_results_table(results: List[SanityResult]) -> None:
    """Print a compact results table."""
    print("\n" + "=" * 100)
    print("SANITY CHECK RESULTS")
    print("=" * 100)
    print(
        f"{'Prompt':<35} {'Mode':<8} {'Cond':<12} "
        f"{'MSCI':>7} {'st_i':>7} {'st_a':>7} {'ImgDom':<8} {'ImgSim':>7} {'Err'}"
    )
    print("-" * 100)

    for r in results:
        prompt_short = r.prompt[:33] + ".." if len(r.prompt) > 35 else r.prompt
        err = r.error[:20] if r.error else ""
        print(
            f"{prompt_short:<35} {r.mode:<8} {r.condition:<12} "
            f"{r.msci or 0:>7.4f} {r.st_i or 0:>7.4f} {r.st_a or 0:>7.4f} "
            f"{(r.image_domain or 'N/A'):<8} {r.image_similarity or 0:>7.4f} {err}"
        )


def print_msci_comparison(results: List[SanityResult]) -> None:
    """Print MSCI comparison: baseline vs perturbations."""
    print("\n" + "=" * 100)
    print("MSCI COMPARISON: BASELINE vs PERTURBATIONS")
    print("=" * 100)

    for prompt in SANITY_PROMPTS:
        print(f"\n  \"{prompt[:60]}\"")
        for mode in MODES:
            runs = {
                r.condition: r
                for r in results
                if r.prompt == prompt and r.mode == mode and r.error is None
            }
            if "baseline" not in runs:
                continue

            b = runs["baseline"]
            line = f"    [{mode:>7}] baseline={b.msci or 0:.4f}"

            if "wrong_image" in runs:
                wi = runs["wrong_image"]
                delta = (b.msci or 0) - (wi.msci or 0)
                line += f"  wrong_img={wi.msci or 0:.4f} (Δ={delta:+.4f})"

            if "wrong_audio" in runs:
                wa = runs["wrong_audio"]
                delta = (b.msci or 0) - (wa.msci or 0)
                line += f"  wrong_aud={wa.msci or 0:.4f} (Δ={delta:+.4f})"

            print(line)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Sanity Check")
    parser.add_argument(
        "--skip-text", action="store_true",
        help="Skip text generation (no Ollama needed). Uses prompt as text directly.",
    )
    parser.add_argument(
        "--out-dir", default="runs/sanity_check",
        help="Output directory for sanity check runs",
    )
    parser.add_argument(
        "--mode", choices=["direct", "planner", "both"], default="direct",
        help="Which mode(s) to test. 'planner' requires Ollama. Default: direct",
    )
    args = parser.parse_args()

    modes = MODES if args.mode == "both" else [args.mode]
    skip_text = args.skip_text

    # If planner mode is selected, we need Ollama regardless
    if "planner" in modes and skip_text:
        print("WARNING: planner mode requires Ollama. Will use direct mode only with --skip-text.")
        modes = ["direct"]

    print("=" * 100)
    print("PHASE 3: SANITY CHECK")
    print(f"  Prompts:    {len(SANITY_PROMPTS)}")
    print(f"  Modes:      {modes}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Seed:       {SEED}")
    print(f"  Skip text:  {skip_text}")
    print(f"  Total runs: {len(SANITY_PROMPTS) * len(modes) * len(CONDITIONS)}")
    print("=" * 100)

    # Verify indexes exist
    for idx in ["data/embeddings/image_index.npz", "data/embeddings/audio_index.npz"]:
        if not Path(idx).exists():
            print(f"\nERROR: Index not found: {idx}")
            print("Run: python scripts/build_embedding_indexes.py")
            sys.exit(1)

    results: List[SanityResult] = []
    total = len(SANITY_PROMPTS) * len(modes) * len(CONDITIONS)
    count = 0

    for prompt in SANITY_PROMPTS:
        for mode in modes:
            for condition in CONDITIONS:
                count += 1
                short = prompt[:45]
                print(f"\n[{count}/{total}] {short}  mode={mode}  cond={condition}")

                t0 = time.time()
                result = run_pipeline_condition(
                    prompt=prompt,
                    mode=mode,
                    condition=condition,
                    seed=SEED,
                    out_dir=args.out_dir,
                    skip_text=skip_text,
                )
                elapsed = time.time() - t0

                if result.error:
                    print(f"  ERROR: {result.error[:80]}")
                else:
                    print(
                        f"  MSCI={result.msci:.4f}  st_i={result.st_i:.4f}  "
                        f"st_a={result.st_a:.4f}  img_dom={result.image_domain}  "
                        f"({elapsed:.1f}s)"
                    )

                results.append(result)

    # ── Print results ──────────────────────────────────────────────────
    print_results_table(results)
    print_msci_comparison(results)

    # ── Run checks ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)

    checks = [
        ("CHECK 1: Retrieval Quality (domain-appropriate images)", check_1_retrieval_quality),
        ("CHECK 2: Perturbation Quality (genuine mismatches)", check_2_perturbation_quality),
        ("CHECK 3: Audio Differentiation (different prompts → different audio)", check_3_audio_differentiation),
        ("CHECK 4: MSCI Sensitivity (drops under perturbation)", check_4_msci_sensitivity),
    ]

    all_passed = True
    for name, check_fn in checks:
        passed, issues = check_fn(results)
        status = "PASS" if passed else "FAIL"
        print(f"\n{name}")
        print(f"  Status: {status}")
        if issues:
            for issue in issues:
                print(issue)
        if not passed:
            all_passed = False

    # ── Save results ───────────────────────────────────────────────────
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "prompts": len(SANITY_PROMPTS),
            "modes": modes,
            "conditions": CONDITIONS,
            "seed": SEED,
            "skip_text": skip_text,
        },
        "results": [
            {
                "prompt": r.prompt,
                "mode": r.mode,
                "condition": r.condition,
                "msci": r.msci,
                "st_i": r.st_i,
                "st_a": r.st_a,
                "si_a": r.si_a,
                "image_path": r.image_path,
                "image_domain": r.image_domain,
                "image_similarity": r.image_similarity,
                "retrieval_failed": r.retrieval_failed,
                "audio_path": r.audio_path,
                "audio_backend": r.audio_backend,
                "perturbation": r.perturbation,
                "error": r.error,
            }
            for r in results
        ],
    }
    report_path = out_path / "sanity_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    # ── Final verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    if all_passed:
        print("VERDICT: ALL CHECKS PASSED — Safe to proceed to Phase 4 (RQ1)")
    else:
        print("VERDICT: SOME CHECKS FAILED — Review issues above before proceeding")
    print("=" * 100)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
