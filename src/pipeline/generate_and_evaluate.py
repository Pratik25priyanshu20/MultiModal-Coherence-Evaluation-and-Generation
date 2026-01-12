from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import json
import time
import uuid

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.explainability.plan_embedder import PlanEmbedder
from src.explainability.semantic_drift import compute_drift
from src.utils.seed import set_global_seed
from src.pipeline.conditioning_validator import validate_conditioning_strictness, check_plan_completeness


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


def _looks_mismatched(plan_brief: str, image_path: str) -> bool:
    """
    Very cheap heuristic guard to avoid obvious drift with your current image sources.
    (City plan -> beach file, etc.)
    We only retry once to keep runtime sane.
    """
    brief = (plan_brief or "").lower()
    name = Path(image_path).name.lower()

    # Add/remove keywords based on what is in your processed image bank
    city_like = any(k in brief for k in ["city", "street", "downtown", "skyscraper", "neon", "urban"])
    beach_like = any(k in brief for k in ["beach", "ocean", "waves", "shore", "sea"])

    if city_like and any(k in name for k in ["beach", "waves", "ocean", "shore", "sea"]):
        return True
    if beach_like and any(k in name for k in ["city", "street", "downtown", "neon", "urban"]):
        return True

    return False


def generate_and_evaluate(
    prompt: str,
    out_dir: str = "runs/unified",
    use_ollama: bool = True,
    deterministic: bool = True,
    seed: int = 42,
) -> GenerationBundle:
    """
    One prompt -> semantic plan -> modality prompts -> generate text + image + audio
    -> evaluate coherence -> save bundle.json
    """
    if deterministic:
        set_global_seed(seed)

    run_id = _new_run_id()
    out_dir = Path(out_dir) / run_id
    _ensure_dir(out_dir)

    # -------------------------
    # 1) SEMANTIC PLAN (planner)
    # -------------------------
    from src.planner.unified_planner import UnifiedPlanner
    from src.planner.schema_to_text import plan_to_prompts

    planner = UnifiedPlanner()
    plan = planner.plan(prompt)
    prompts = plan_to_prompts(plan)

    text_prompt = prompts["text_prompt"]
    image_prompt = prompts["image_prompt"]
    audio_prompt = prompts["audio_prompt"]
    shared_brief = prompts["shared_brief"]

    # Phase 3: Validate conditioning strictness (no raw prompt leakage)
    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
    is_valid, validation_reason = validate_conditioning_strictness(
        original_prompt=prompt,
        plan=plan_dict,
        modality_prompts=prompts,
    )
    if not is_valid:
        # Log warning but continue (non-fatal for now)
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
    # 3) IMAGE GENERATION
    # -------------------------
    # Use improved generator with similarity filtering
    image_dir = out_dir / "image"
    try:
        from src.generators.image.generator_improved import generate_image_improved
        image_path: str = generate_image_improved(
            prompt=image_prompt,
            out_dir=str(image_dir),
            min_similarity=0.15,  # Filter low-similarity matches
        )
    except ImportError:
        # Fallback to original generator
        from src.generators.image.generator import generate_image
        image_path: str = generate_image(prompt=image_prompt, out_dir=str(image_dir))

    # Simple 1x retry if it is obviously wrong
    if _looks_mismatched(shared_brief, image_path):
        # Retry with same generator (improved or fallback)
        try:
            from src.generators.image.generator_improved import generate_image_improved
            image_path = generate_image_improved(
                prompt=image_prompt,
                out_dir=str(image_dir),
                min_similarity=0.15,
            )
        except ImportError:
            from src.generators.image.generator import generate_image
            image_path = generate_image(prompt=image_prompt, out_dir=str(image_dir))

    # -------------------------
    # 4) AUDIO GENERATION
    # -------------------------
    from src.generators.audio.generator import generate_audio

    audio_path: str = generate_audio(
        prompt=audio_prompt,
        out_dir=str(out_dir / "audio"),
        deterministic=deterministic,
        seed=seed,
    )

    # -------------------------
    # 4.5) SEMANTIC DRIFT (plan -> outputs)
    # -------------------------
    embedder = AlignedEmbedder(target_dim=512)
    plan_embedder = PlanEmbedder(embedder=embedder)
    plan_emb = plan_embedder.embed(plan)
    text_emb = embedder.embed_text(generated_text)
    image_emb = embedder.embed_image(image_path)
    audio_emb = embedder.embed_audio(audio_path)

    semantic_drift = {
        "text": round(compute_drift(plan_emb, text_emb), 4),
        "image": round(compute_drift(plan_emb, image_emb), 4),
        "audio": round(compute_drift(plan_emb, audio_emb), 4),
    }

    # -------------------------
    # 5) COHERENCE EVALUATION
    # -------------------------
    from src.coherence.coherence_engine import evaluate_coherence

    eval_out: Dict[str, Any] = evaluate_coherence(
        text=generated_text,
        image_path=image_path,
        audio_path=audio_path,
    )

    scores = eval_out.get("scores", {})
    coherence = eval_out.get("coherence", eval_out)

    bundle = GenerationBundle(
        run_id=run_id,
        prompt=prompt,
        generated_text=generated_text,
        image_path=str(image_path),
        audio_path=str(audio_path),
        scores=scores,
        coherence=coherence,
        semantic_drift=semantic_drift,
        meta={
            "out_dir": str(out_dir),
            "use_ollama": use_ollama,
            "deterministic": deterministic,
            "seed": seed,
            "semantic_plan": plan.model_dump() if hasattr(plan, "model_dump") else plan,
            "modality_prompts": {
                "text_prompt": text_prompt,
                "image_prompt": image_prompt,
                "audio_prompt": audio_prompt,
                "shared_brief": shared_brief,
            },
            "conditioning_validation": {
                "is_valid": is_valid,
                "reason": validation_reason,
                "plan_completeness": check_plan_completeness(plan_dict),
            },
        },
    )

    with (out_dir / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(bundle), f, indent=2, ensure_ascii=False)

    return bundle
