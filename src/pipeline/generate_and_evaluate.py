"""
End-to-end generation and evaluation pipeline.

Updated for Phase 1-2:
- Uses domain-gated image retrieval with explicit retrieval_failed tracking
- Uses deterministic audio with explicit backend tracking
- Perturbations draw from ALL available data (not just processed/)
- Ensures wrong_image/wrong_audio are genuinely mismatched
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time
import uuid

from src.embeddings.aligned_embeddings import AlignedEmbedder

logger = logging.getLogger(__name__)
from src.explainability.plan_embedder import PlanEmbedder
from src.explainability.semantic_drift import compute_drift
from src.utils.seed import set_global_seed
from src.pipeline.conditioning_validator import (
    validate_conditioning_strictness,
    check_plan_completeness,
)


@dataclass
class GenerationBundle:
    run_id: str
    prompt: str
    generated_text: str
    image_path: str
    audio_path: str
    scores: Dict[str, float]
    coherence: Dict[str, Any]
    semantic_drift: Dict[str, float]
    meta: Dict[str, Any]


def _new_run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _collect_all_image_paths() -> list[Path]:
    """Collect images from ALL data directories for perturbation pool."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    dirs = [
        Path("data/processed/images"),
        Path("data/wikimedia/images"),
    ]
    paths = []
    seen = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in exts and p.name not in seen:
                seen.add(p.name)
                paths.append(p)
    return paths


def _collect_all_audio_paths() -> list[Path]:
    """Collect audio from ALL data directories for perturbation pool."""
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    dirs = [
        Path("data/processed/audio"),
        Path("data/freesound/audio"),
    ]
    paths = []
    seen = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in exts and p.name not in seen:
                seen.add(p.name)
                paths.append(p)
    return paths


def _select_mismatched_path(
    paths: list[Path],
    exclude: str,
    original_domain: Optional[str],
    seed: int | None,
) -> str | None:
    """
    Select a replacement path that is genuinely mismatched.

    Prefers files from a DIFFERENT domain than the original.
    Falls back to any file != exclude if no domain info.
    """
    import random as random_mod

    rng = random_mod.Random(seed) if seed is not None else random_mod

    # Filter out the original
    candidates = [str(p) for p in paths if str(p) != exclude]
    if not candidates:
        return None

    if original_domain:
        # Prefer candidates from a different domain
        from src.generators.image.generator_improved import DOMAIN_KEYWORDS
        different_domain = []
        for c in candidates:
            name = Path(c).stem.lower()
            c_domain = None
            for domain, keywords in DOMAIN_KEYWORDS.items():
                if any(kw in name for kw in keywords):
                    c_domain = domain
                    break
            if c_domain and c_domain != original_domain:
                different_domain.append(c)

        if different_domain:
            return rng.choice(different_domain)

    # Fallback: any candidate that isn't the original
    return rng.choice(candidates)


def _infer_domain_from_path(filepath: str) -> Optional[str]:
    """Infer domain from a file path."""
    name = Path(filepath).stem.lower()
    from src.generators.image.generator_improved import DOMAIN_KEYWORDS
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return domain
    return None


def generate_and_evaluate(
    prompt: str,
    out_dir: str = "runs/unified",
    use_ollama: bool = True,
    deterministic: bool = True,
    seed: int = 42,
    mode: str = "planner",
    condition: str = "baseline",
    use_stable_diffusion: bool = False,
) -> GenerationBundle:
    """
    One prompt -> (optionally) semantic plan -> modality prompts -> generate text + image + audio
    -> evaluate coherence -> save bundle.json

    Args:
        prompt: Input prompt
        mode: "direct" (bypass planner) or "planner" (use semantic planner)
        condition: "baseline", "wrong_image", or "wrong_audio" (for controlled experiments)
    """
    valid_modes = {"direct", "planner"}
    valid_conditions = {"baseline", "wrong_image", "wrong_audio"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {sorted(valid_modes)}.")
    if condition not in valid_conditions:
        raise ValueError(
            f"Unknown condition '{condition}'. Expected one of {sorted(valid_conditions)}."
        )

    if deterministic:
        set_global_seed(seed)

    run_id = _new_run_id()
    out_path = Path(out_dir) / run_id
    _ensure_dir(out_path)

    logger.info("Run %s: mode=%s, condition=%s, seed=%d", run_id, mode, condition, seed)

    # -------------------------
    # 1) PROMPT GENERATION (mode-dependent)
    # -------------------------
    plan = None
    plan_dict = None

    if mode == "direct":
        text_prompt = prompt
        image_prompt = prompt
        audio_prompt = prompt
        shared_brief = prompt
        is_valid = True
        validation_reason = "Direct mode: no planner validation"
    else:
        from src.planner.unified_planner import UnifiedPlanner
        from src.planner.schema_to_text import plan_to_prompts

        planner = UnifiedPlanner()
        plan = planner.plan(prompt)
        prompts = plan_to_prompts(plan)

        text_prompt = prompts["text_prompt"]
        image_prompt = prompts["image_prompt"]
        audio_prompt = prompts["audio_prompt"]
        shared_brief = prompts["shared_brief"]

        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
        is_valid, validation_reason = validate_conditioning_strictness(
            original_prompt=prompt,
            plan=plan_dict,
            modality_prompts=prompts,
        )
        if not is_valid:
            import warnings
            warnings.warn(f"Conditioning validation failed: {validation_reason}")

    # -------------------------
    # 2) TEXT GENERATION
    # -------------------------
    from src.generators.text.generator import generate_text

    generated_text: str = generate_text(
        prompt=text_prompt,
        use_ollama=use_ollama,
        deterministic=deterministic,
    )

    # -------------------------
    # 3) IMAGE GENERATION / RETRIEVAL
    # -------------------------
    image_retrieval_meta = None

    if use_stable_diffusion:
        from src.generators.image.generator_hybrid import HybridImageGenerator

        hybrid_gen = HybridImageGenerator(
            force_sd=True,
            index_path="data/embeddings/image_index.npz",
        )
        hybrid_result = hybrid_gen.generate(
            prompt=image_prompt,
            out_path=str(out_path / "image" / "generated.png"),
            seed=seed if deterministic else None,
        )
        image_path = hybrid_result.image_path
        image_retrieval_meta = hybrid_result.to_dict()
        logger.info("Image generation: backend=%s", hybrid_result.backend)
    else:
        from src.generators.image.generator_improved import (
            generate_image_with_metadata,
            generate_image_improved,
        )

        try:
            image_result = generate_image_with_metadata(
                prompt=image_prompt,
                min_similarity=0.20,
            )
            image_path = image_result.image_path
            image_retrieval_meta = {
                "similarity": image_result.similarity,
                "domain": image_result.domain,
                "retrieval_failed": image_result.retrieval_failed,
                "candidates_considered": image_result.candidates_considered,
                "candidates_above_threshold": image_result.candidates_above_threshold,
                "top_5": image_result.top_5,
            }
            logger.info("Image retrieval: sim=%.4f, domain=%s, failed=%s",
                         image_result.similarity, image_result.domain, image_result.retrieval_failed)
        except Exception as exc:
            logger.warning("Image retrieval with metadata failed (%s), falling back", exc)
            image_path = generate_image_improved(
                prompt=image_prompt,
                out_dir=str(out_path / "image"),
            )

    # -------------------------
    # 4) AUDIO RETRIEVAL (CLAP-based, mirrors image retrieval)
    # -------------------------
    # Primary: retrieve real audio from indexed pool using CLAP text-audio space.
    # Fallback: synthetic ambient generation if no index available.
    from src.generators.audio.retrieval import retrieve_audio_with_metadata
    from src.generators.audio.generator import generate_audio_with_metadata

    audio_retrieval_meta = None
    try:
        audio_retrieval_result = retrieve_audio_with_metadata(
            prompt=audio_prompt,
            min_similarity=0.10,
        )
        audio_path = audio_retrieval_result.audio_path
        audio_retrieval_meta = {
            "similarity": audio_retrieval_result.similarity,
            "retrieval_failed": audio_retrieval_result.retrieval_failed,
            "candidates_considered": audio_retrieval_result.candidates_considered,
            "candidates_above_threshold": audio_retrieval_result.candidates_above_threshold,
            "top_5": audio_retrieval_result.top_5,
            "backend": "retrieval",
        }
        logger.info("Audio retrieval: sim=%.4f, failed=%s",
                     audio_retrieval_result.similarity, audio_retrieval_result.retrieval_failed)
    except Exception as exc:
        logger.warning("Audio retrieval failed (%s), falling back to synthetic ambient", exc)
        # Fallback to synthetic ambient generation
        audio_gen_result = generate_audio_with_metadata(
            prompt=audio_prompt,
            out_dir=str(out_path / "audio"),
            deterministic=deterministic,
            seed=seed,
        )
        audio_path = audio_gen_result.audio_path
        audio_retrieval_meta = audio_gen_result.to_dict()
        audio_retrieval_meta["backend"] = "fallback_ambient"

    # -------------------------
    # 4.5) PERTURBATION CONDITIONS
    # -------------------------
    perturbation_meta = {"applied": condition}

    if condition == "wrong_image":
        original_domain = _infer_domain_from_path(image_path)
        images = _collect_all_image_paths()
        replacement = _select_mismatched_path(
            images,
            exclude=str(image_path),
            original_domain=original_domain,
            seed=seed if deterministic else None,
        )
        if replacement:
            perturbation_meta["original_image"] = str(image_path)
            perturbation_meta["replacement_image"] = replacement
            perturbation_meta["original_domain"] = original_domain
            perturbation_meta["replacement_domain"] = _infer_domain_from_path(replacement)
            image_path = replacement

    if condition == "wrong_audio":
        original_domain = _infer_domain_from_path(audio_path)
        audios = _collect_all_audio_paths()
        replacement = _select_mismatched_path(
            audios,
            exclude=str(audio_path),
            original_domain=original_domain,
            seed=seed if deterministic else None,
        )
        if replacement:
            perturbation_meta["original_audio"] = str(audio_path)
            perturbation_meta["replacement_audio"] = replacement
            perturbation_meta["original_domain"] = original_domain
            perturbation_meta["replacement_domain"] = _infer_domain_from_path(replacement)
            audio_path = replacement

    # -------------------------
    # 5) SEMANTIC DRIFT (plan -> outputs)
    # -------------------------
    # Use correct embedding spaces:
    #   text drift: CLIP text(plan) vs CLIP text(output) — same space
    #   image drift: CLIP text(plan) vs CLIP image(output) — CLIP shared space
    #   audio drift: CLAP text(plan) vs CLAP audio(output) — CLAP shared space
    embedder = AlignedEmbedder(target_dim=512)
    text_emb = embedder.embed_text(generated_text)
    image_emb = embedder.embed_image(image_path)
    audio_emb = embedder.embed_audio(audio_path)

    if mode == "planner" and plan is not None:
        plan_embedder = PlanEmbedder(embedder=embedder)
        plan_emb_clip = plan_embedder.embed(plan)  # CLIP text space
        plan_text = plan.scene_summary if hasattr(plan, "scene_summary") else prompt
        plan_emb_clap = embedder.embed_text_for_audio(plan_text)  # CLAP text space
        semantic_drift = {
            "text": round(compute_drift(plan_emb_clip, text_emb), 4),
            "image": round(compute_drift(plan_emb_clip, image_emb), 4),
            "audio": round(compute_drift(plan_emb_clap, audio_emb), 4),
        }
    else:
        prompt_emb_clip = embedder.embed_text(prompt)
        prompt_emb_clap = embedder.embed_text_for_audio(prompt)
        semantic_drift = {
            "text": round(compute_drift(prompt_emb_clip, text_emb), 4),
            "image": round(compute_drift(prompt_emb_clip, image_emb), 4),
            "audio": round(compute_drift(prompt_emb_clap, audio_emb), 4),
        }

    # -------------------------
    # 6) COHERENCE EVALUATION
    # -------------------------
    from src.coherence.coherence_engine import evaluate_coherence

    eval_out: Dict[str, Any] = evaluate_coherence(
        text=generated_text,
        image_path=image_path,
        audio_path=audio_path,
    )

    scores = eval_out.get("scores", {})
    coherence = eval_out.get("coherence", eval_out)

    # -------------------------
    # 7) BUILD METADATA
    # -------------------------
    meta = {
        "out_dir": str(out_path),
        "use_ollama": use_ollama,
        "deterministic": deterministic,
        "seed": seed,
        "mode": mode,
        "condition": condition,
        "modality_prompts": {
            "text_prompt": text_prompt,
            "image_prompt": image_prompt,
            "audio_prompt": audio_prompt,
            "shared_brief": shared_brief,
        },
        "audio_retrieval": audio_retrieval_meta,
        "image_retrieval": image_retrieval_meta,
        "perturbation": perturbation_meta,
    }

    if mode == "planner" and plan is not None:
        meta["semantic_plan"] = plan.model_dump() if hasattr(plan, "model_dump") else plan
        meta["conditioning_validation"] = {
            "is_valid": is_valid,
            "reason": validation_reason,
            "plan_completeness": check_plan_completeness(plan_dict) if plan_dict else None,
        }
    else:
        meta["semantic_plan"] = None
        meta["conditioning_validation"] = {
            "is_valid": True,
            "reason": "Direct mode: no planner validation",
            "plan_completeness": None,
        }

    bundle = GenerationBundle(
        run_id=run_id,
        prompt=prompt,
        generated_text=generated_text,
        image_path=str(image_path),
        audio_path=str(audio_path),
        scores=scores,
        coherence=coherence,
        semantic_drift=semantic_drift,
        meta=meta,
    )

    with (out_path / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(bundle), f, indent=2, ensure_ascii=False)

    return bundle
