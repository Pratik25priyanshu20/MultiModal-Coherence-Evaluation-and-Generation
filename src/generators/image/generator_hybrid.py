"""
Hybrid image generator: Stable Diffusion (SDXL -> SD 1.5) with retrieval fallback.

Follows the AudioLDM pattern in src/generators/audio/generator.py:
- Try SDXL -> fallback SD 1.5 -> fallback retrieval
- force_stable_diffusion flag (default False for backward compat)
- Deterministic generation via fixed seed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageGenerationResult:
    """Result of image generation with full metadata."""
    image_path: str
    backend: str  # "sdxl", "sd1.5", or "retrieval"
    seed: Optional[int] = None
    similarity: Optional[float] = None  # Only for retrieval
    retrieval_failed: Optional[bool] = None  # Only for retrieval
    num_inference_steps: Optional[int] = None  # Only for SD
    guidance_scale: Optional[float] = None  # Only for SD
    domain: Optional[str] = None
    candidates_considered: Optional[int] = None
    candidates_above_threshold: Optional[int] = None
    top_5: Optional[List[Tuple[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# SD model configs
SD_MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


class HybridImageGenerator:
    """
    Image generator with SDXL -> SD 1.5 -> retrieval fallback chain.

    Default behavior (force_sd=False): pure retrieval (backward compatible).
    With force_sd=True: tries Stable Diffusion first, falls back to retrieval on error.
    """

    def __init__(
        self,
        force_sd: bool = False,
        sd_model: str = "sdxl",
        device: str = "cpu",
        index_path: str = "data/embeddings/image_index.npz",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        self.force_sd = force_sd
        self.device = device
        self.index_path = index_path
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self._sd_pipe = None
        self._sd_backend_name = None
        self._sd_error = None
        self._torch = None

        if force_sd:
            self._init_sd_pipeline(sd_model)

        # Retrieval generator (always available as fallback)
        self._retrieval_generator = None

    def _init_sd_pipeline(self, sd_model: str) -> None:
        """Try loading SD pipeline: SDXL first, then SD 1.5."""
        models_to_try = []
        if sd_model == "sdxl":
            models_to_try = ["sdxl", "sd1.5"]
        elif sd_model == "sd1.5":
            models_to_try = ["sd1.5"]
        else:
            models_to_try = [sd_model]

        for model_key in models_to_try:
            model_id = SD_MODELS.get(model_key, model_key)
            try:
                import torch
                self._torch = torch

                if model_key == "sdxl":
                    from diffusers import StableDiffusionXLPipeline
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        variant="fp16" if self.device != "cpu" else None,
                    )
                else:
                    from diffusers import StableDiffusionPipeline
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    )

                pipe = pipe.to(self.device)
                self._sd_pipe = pipe
                self._sd_backend_name = model_key
                logger.info("Loaded SD pipeline: %s (%s)", model_key, model_id)
                return

            except Exception as exc:
                logger.warning("Failed to load %s: %s", model_key, exc)
                self._sd_error = f"{model_key}: {exc}"
                continue

        logger.warning("All SD models failed to load. Will use retrieval fallback.")

    def _get_retrieval_generator(self):
        """Lazy-load retrieval generator."""
        if self._retrieval_generator is None:
            from src.generators.image.generator_improved import ImprovedImageRetrievalGenerator
            self._retrieval_generator = ImprovedImageRetrievalGenerator(
                index_path=self.index_path,
            )
        return self._retrieval_generator

    def generate(
        self,
        prompt: str,
        out_path: str,
        seed: Optional[int] = None,
        min_similarity: float = 0.20,
    ) -> ImageGenerationResult:
        """
        Generate an image for a prompt.

        If SD is loaded: generate with SD, catch errors and fallback to retrieval.
        Otherwise: use retrieval directly.
        """
        if self._sd_pipe is not None:
            try:
                return self._generate_sd(prompt, out_path, seed)
            except Exception as exc:
                logger.warning(
                    "SD generation failed (%s), falling back to retrieval: %s",
                    self._sd_backend_name, exc,
                )

        return self._generate_retrieval(prompt, min_similarity)

    def _generate_sd(
        self,
        prompt: str,
        out_path: str,
        seed: Optional[int],
    ) -> ImageGenerationResult:
        """Generate image with Stable Diffusion."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        generator = None
        if seed is not None and self._torch is not None:
            generator = self._torch.Generator(device=self.device).manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if generator is not None:
            kwargs["generator"] = generator

        result = self._sd_pipe(**kwargs)
        image = result.images[0]
        image.save(str(out_path))

        return ImageGenerationResult(
            image_path=str(out_path),
            backend=self._sd_backend_name,
            seed=seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )

    def _generate_retrieval(
        self,
        prompt: str,
        min_similarity: float = 0.20,
    ) -> ImageGenerationResult:
        """Fallback: retrieve image from index."""
        generator = self._get_retrieval_generator()
        result = generator.retrieve(prompt, min_similarity=min_similarity)

        return ImageGenerationResult(
            image_path=result.image_path,
            backend="retrieval",
            similarity=result.similarity,
            retrieval_failed=result.retrieval_failed,
            domain=result.domain,
            candidates_considered=result.candidates_considered,
            candidates_above_threshold=result.candidates_above_threshold,
            top_5=result.top_5,
        )


def generate_image_hybrid(
    prompt: str,
    out_path: str = "/tmp/generated_image.png",
    force_sd: bool = False,
    sd_model: str = "sdxl",
    device: str = "cpu",
    seed: Optional[int] = None,
    index_path: str = "data/embeddings/image_index.npz",
) -> ImageGenerationResult:
    """
    Convenience function: generate an image with hybrid SD/retrieval backend.

    Default (force_sd=False): pure retrieval (current behavior).
    With force_sd=True: SDXL -> SD 1.5 -> retrieval chain.
    """
    generator = HybridImageGenerator(
        force_sd=force_sd,
        sd_model=sd_model,
        device=device,
        index_path=index_path,
    )
    return generator.generate(prompt, out_path, seed=seed)
