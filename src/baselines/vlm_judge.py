"""
VLM-as-Judge Baseline for Multimodal Coherence.

Uses a Vision-Language Model to rate coherence of text-image-audio triples
on a 1-5 scale (matching human rating scale).

Approach:
1. Show image + audio spectrogram + text caption to VLM
2. Prompt VLM to rate coherence on 1-5 scale
3. Parse numeric response

Supports:
- Qwen-VL (via Ollama) — preferred, runs locally
- LLaVA (via Ollama or HuggingFace) — alternative
- GPT-4V (via API) — if available

Usage:
    from src.baselines.vlm_judge import VLMJudge
    judge = VLMJudge(model="qwen2-vl")
    result = judge.score("A sunset beach", "beach.jpg", "waves.wav")
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

COHERENCE_PROMPT = """You are evaluating the semantic coherence of a multimodal content bundle.

Given:
- A text caption describing a scene
- An image that should match the caption
- A spectrogram of audio that should match the caption

Rate the overall coherence on a scale of 1-5:
1 = No coherence (elements are completely unrelated)
2 = Low coherence (weak thematic connection)
3 = Moderate coherence (related but with mismatches)
4 = Good coherence (elements clearly related)
5 = Excellent coherence (elements perfectly synchronized)

Text caption: "{text}"

Look at the image and the audio spectrogram below.
Consider how well:
- The image matches the text description
- The audio (shown as spectrogram) matches the text description
- The image and audio complement each other

Respond with ONLY a single integer from 1 to 5, nothing else."""

COHERENCE_PROMPT_COT = """You are evaluating the semantic coherence of a multimodal content bundle.

Given:
- A text caption describing a scene
- An image that should match the caption
- A spectrogram of audio that should match the caption

Text caption: "{text}"

Please evaluate step by step:

Step 1 - Image-Text Coherence: How well does the image match the text description?
Rate 1-5 (1=unrelated, 5=perfect match).

Step 2 - Audio-Text Coherence: How well does the audio spectrogram match the text description?
Rate 1-5 (1=unrelated, 5=perfect match).

Step 3 - Image-Audio Complementarity: How well do the image and audio complement each other to create a coherent scene?
Rate 1-5 (1=no complementarity, 5=perfectly complementary).

Step 4 - Provide your final overall coherence rating considering all three dimensions.

Format your response exactly as:
STEP 1: X/5 - [brief explanation]
STEP 2: X/5 - [brief explanation]
STEP 3: X/5 - [brief explanation]
OVERALL: X/5"""

RECOMMENDED_MODELS = {
    "llava:13b": {"vram_gb": 10, "quality": "high", "speed": "slow"},
    "llava:7b": {"vram_gb": 6, "quality": "good", "speed": "medium"},
    "qwen2-vl:7b": {"vram_gb": 6, "quality": "good", "speed": "medium"},
    "minicpm-v:8b": {"vram_gb": 6, "quality": "good", "speed": "medium"},
    "bakllava:7b": {"vram_gb": 6, "quality": "fair", "speed": "fast"},
}


class VLMJudge:
    """VLM-based coherence judge using vision-language models."""

    def __init__(
        self,
        model: str = "llava",
        ollama_host: str = "http://localhost:11434",
        use_cot: bool = False,
    ):
        self.model = model
        self.ollama_host = ollama_host
        self._use_cot = use_cot
        self._available = None

        # Response cache
        self._cache_dir = Path(".cache/vlm_judge")
        self._cache: Dict[str, Any] = {}
        self._cache_file = self._cache_dir / "vlm_cache.json"
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info("Loaded %d cached VLM responses", len(self._cache))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load VLM cache: %s", e)
                self._cache = {}

    def _cache_key(self, text: str, image_path: str, audio_path: str) -> str:
        """Return SHA256 hash key for caching VLM responses."""
        prompt_mode = "cot" if self._use_cot else "standard"
        key_str = f"{self.model}|{prompt_mode}|{text}|{image_path}|{audio_path}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        """Persist the response cache to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save VLM cache: %s", e)

    def _generate_spectrogram(self, audio_path: str) -> str:
        """Generate spectrogram image from audio file.

        Returns path to temporary PNG file.
        """
        import librosa
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        waveform, sr = librosa.load(audio_path, sr=22050, mono=True)
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title("Audio Spectrogram")
        plt.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return tmp.name

    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image as base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_ollama(
        self, text: str, image_path: str, spectrogram_path: str,
    ) -> Optional[int]:
        """Call Ollama VLM API with image(s).

        Retries up to 3 times with exponential backoff on timeout/connection
        errors. Timeouts are [120, 240, 480] seconds per attempt.
        """
        import requests

        template = COHERENCE_PROMPT_COT if self._use_cot else COHERENCE_PROMPT
        prompt = template.format(text=text)

        # Encode images
        img_b64 = self._encode_image_base64(image_path)
        spec_b64 = self._encode_image_base64(spectrogram_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64, spec_b64],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        timeouts = [120, 240, 480]
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                resp = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=timeouts[attempt],
                )
                resp.raise_for_status()
                response_text = resp.json().get("response", "")
                return self._parse_rating(response_text)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_attempts - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        "Ollama attempt %d/%d failed (%s), retrying in %ds...",
                        attempt + 1, max_attempts, e, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(
                        "Ollama call failed after %d attempts: %s",
                        max_attempts, e,
                    )
                    return None
            except Exception as e:
                logger.warning("Ollama call failed: %s", e)
                return None

        return None

    def _call_transformers(
        self, text: str, image_path: str, spectrogram_path: str,
    ) -> Optional[int]:
        """Fallback: Use HuggingFace transformers for VLM inference."""
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            from PIL import Image

            model_id = "llava-hf/llava-1.5-7b-hf"
            processor = AutoProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16
            )

            image = Image.open(image_path).convert("RGB")
            prompt = f"USER: <image>\n{COHERENCE_PROMPT.format(text=text)}\nASSISTANT:"

            inputs = processor(text=prompt, images=image, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=10, temperature=0.0)
            response = processor.decode(output[0], skip_special_tokens=True)

            return self._parse_rating(response.split("ASSISTANT:")[-1])
        except Exception as e:
            logger.warning("Transformers VLM failed: %s", e)
            return None

    def _parse_rating(self, response: str) -> Optional[int]:
        """Parse integer rating from VLM response.

        CoT-aware parsing strategy:
        1. Look for OVERALL: X/5 pattern (chain-of-thought format)
        2. Fall back to last digit 1-5 in the response
        3. Fall back to first digit 1-5 in the response
        """
        response = response.strip()

        # 1. CoT format: look for "OVERALL: X/5" pattern
        overall_match = re.search(r'OVERALL:\s*(\d)\s*/\s*5', response)
        if overall_match:
            val = int(overall_match.group(1))
            if 1 <= val <= 5:
                return val

        # 2. Find the last digit 1-5 in the response
        all_matches = re.findall(r'[1-5]', response)
        if all_matches:
            return int(all_matches[-1])

        # 3. No valid digit found
        return None

    def score(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Score a single text-image-audio triple using VLM.

        Returns:
            Dict with method, score (1-5 normalized to 0-1), raw_rating.
        """
        # Check cache first
        cache_key = self._cache_key(text, image_path, audio_path)
        if cache_key in self._cache:
            logger.debug("VLM cache hit for key %s", cache_key[:12])
            return self._cache[cache_key]

        # Generate spectrogram from audio
        spec_path = None
        try:
            spec_path = self._generate_spectrogram(audio_path)
        except Exception as e:
            return {"method": "VLM_Judge", "score": None, "error": f"spectrogram_failed: {e}"}

        # Try Ollama first, then transformers
        rating = self._call_ollama(text, image_path, spec_path)

        if rating is None:
            rating = self._call_transformers(text, image_path, spec_path)

        # Clean up temp file
        if spec_path:
            Path(spec_path).unlink(missing_ok=True)

        if rating is None:
            return {"method": "VLM_Judge", "score": None, "error": "no_response"}

        # Normalize 1-5 to 0-1
        normalized_score = (rating - 1) / 4.0

        result = {
            "method": "VLM_Judge",
            "score": float(normalized_score),
            "raw_rating": rating,
            "model": self.model,
        }

        # Cache the result and persist
        self._cache[cache_key] = result
        self._save_cache()

        return result


def evaluate_vlm_judge(
    samples: list,
    human_scores: Optional[dict] = None,
    model: str = "llava",
    use_cot: bool = False,
) -> Dict[str, Any]:
    """Run VLM judge on all samples.

    Args:
        samples: List of sample dicts.
        human_scores: Optional human score dict.
        model: VLM model name for Ollama.
        use_cot: If True, use chain-of-thought prompt for richer evaluation.

    Returns:
        Dict with results and correlations.
    """
    judge = VLMJudge(model=model, use_cot=use_cot)
    results = []

    for i, s in enumerate(samples):
        result = judge.score(
            text=s["prompt_text"],
            image_path=s.get("image_path", ""),
            audio_path=s.get("audio_path", ""),
        )
        result["sample_id"] = s["sample_id"]
        results.append(result)
        status = f"rating={result.get('raw_rating', '?')}" if result.get("score") is not None else "FAILED"
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}: {status}", end="\r")

    print()

    # Compute correlation
    correlations = {}
    if human_scores is not None:
        scores = []
        humans = []
        for r in results:
            sid = r["sample_id"]
            if sid in human_scores and r.get("score") is not None:
                scores.append(r["score"])
                humans.append(human_scores[sid]["weighted_score"]["mean"])

        if len(scores) >= 5:
            rho, p = sp_stats.spearmanr(scores, humans)
            correlations["VLM_Judge"] = {
                "rho": float(rho),
                "p": float(p),
                "n": len(scores),
                "significant": p < 0.05,
            }

    return {
        "results": results,
        "correlations": correlations,
    }
