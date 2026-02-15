"""
Audio generator with explicit backend tracking.

Phase 2: Stability over realism.
- Backend is always recorded (never ambiguous)
- Fallback ambient is the deterministic baseline
- AudioLDM used only if explicitly available and stable
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
from pathlib import Path


@dataclass(frozen=True)
class AudioGenResult:
    """Result of audio generation with full metadata."""
    audio_path: str
    backend: str  # "audioldm" or "fallback_ambient" — always explicit
    prompt_hash: int  # Deterministic hash of (prompt, seed) for reproducibility
    duration_sec: float
    sample_rate: int
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioGenerator:
    """
    Audio generator with explicit backend selection.

    Strategy (Phase 2B):
    - Default: fallback_ambient (fully deterministic, always works)
    - Optional: AudioLDM (if force_audioldm=True and model is available)

    The fallback ambient generator produces prompt-seeded ambient soundscapes.
    This is acceptable for a case study testing alignment behavior, not audio realism.
    """

    def __init__(self, device: str = "cpu", force_audioldm: bool = False):
        self.device = device
        self._audioldm_pipe = None
        self._audioldm_backend_name = None
        self._torch = None
        self._audioldm_error = None

        if force_audioldm:
            try:
                from diffusers import AudioLDMPipeline
                import torch

                model_id = "cvssp/audioldm"
                self._audioldm_pipe = AudioLDMPipeline.from_pretrained(model_id)
                self._audioldm_pipe.to(self.device)
                self._audioldm_backend_name = f"AudioLDMPipeline({model_id})"
                self._torch = torch
            except Exception as exc:
                self._audioldm_error = str(exc)

    def generate(
        self,
        prompt: str,
        out_path: str,
        duration_sec: float = 6.0,
        sr: int = 48000,
        seed: Optional[int] = None,
    ) -> AudioGenResult:
        """
        Generate audio for a prompt.

        Backend selection:
        1. If AudioLDM was loaded (force_audioldm=True): try it, fallback on error
        2. Otherwise: use fallback_ambient (deterministic baseline)
        """
        if self._audioldm_pipe is not None:
            try:
                return self._generate_audioldm(prompt, out_path, duration_sec, sr, seed)
            except Exception as exc:
                return self._generate_fallback(
                    prompt, out_path, duration_sec, sr, seed,
                    note=f"AudioLDM failed at runtime: {exc}",
                )

        return self._generate_fallback(
            prompt, out_path, duration_sec, sr, seed,
            note=self._audioldm_error or "Using deterministic fallback (default)",
        )

    def _generate_audioldm(
        self, prompt: str, out_path: str, duration_sec: float, sr: int, seed: Optional[int],
    ) -> AudioGenResult:
        """Generate with AudioLDM."""
        generator = None
        if seed is not None and self._torch is not None:
            generator = self._torch.Generator(device=self.device).manual_seed(seed)
        kwargs = {"audio_length_in_s": duration_sec}
        if generator is not None:
            kwargs["generator"] = generator
        result = self._audioldm_pipe(prompt, **kwargs)
        audio = result.audios[0]
        sf.write(out_path, audio, sr)

        prompt_hash = abs(hash((prompt, seed))) % (2**32)
        return AudioGenResult(
            audio_path=out_path,
            backend="audioldm",
            prompt_hash=prompt_hash,
            duration_sec=duration_sec,
            sample_rate=sr,
        )

    def _generate_fallback(
        self,
        prompt: str,
        out_path: str,
        duration_sec: float,
        sr: int,
        seed: Optional[int],
        note: str = "",
    ) -> AudioGenResult:
        """
        Deterministic ambient soundscape generator.

        Produces prompt-dependent audio by seeding RNG from hash(prompt) + seed.
        Different prompts produce different spectral characteristics:
        - Drone frequency varies with prompt
        - Noise filtering varies with prompt
        - Amplitude envelope varies with prompt

        This ensures wrong_audio perturbations produce genuinely different audio.
        """
        # Deterministic seed from prompt content
        base_seed = abs(hash(prompt)) % (2**32)
        if seed is not None:
            base_seed = (base_seed + seed) % (2**32)
        rng = np.random.default_rng(base_seed)

        n = int(duration_sec * sr)
        t = np.linspace(0, duration_sec, n, endpoint=False)

        # Prompt-dependent parameters — different prompts get different sounds
        prompt_val = sum(ord(c) for c in prompt)
        drone_freq = 80.0 + (prompt_val % 200)  # 80-280 Hz range
        filter_width = 2000 + (prompt_val % 6000)  # 2000-8000 sample filter
        noise_amplitude = 0.02 + (prompt_val % 50) * 0.001  # 0.02-0.07
        drone_amplitude = 0.06 + (prompt_val % 40) * 0.001  # 0.06-0.10

        # Generate noise with prompt-dependent filtering
        noise = rng.normal(0, 1, size=n).astype(np.float32)
        kernel = np.ones(filter_width, dtype=np.float32) / filter_width
        noise = np.convolve(noise, kernel, mode="same")

        # Prompt-dependent drone
        drone = drone_amplitude * np.sin(2 * np.pi * drone_freq * t).astype(np.float32)

        # Add second harmonic for richer sound
        harmonic_freq = drone_freq * 1.5 + (prompt_val % 100)
        harmonic = (drone_amplitude * 0.3) * np.sin(2 * np.pi * harmonic_freq * t).astype(np.float32)

        audio = (noise_amplitude * noise + drone + harmonic).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        sf.write(out_path, audio, sr)

        return AudioGenResult(
            audio_path=out_path,
            backend="fallback_ambient",
            prompt_hash=base_seed,
            duration_sec=duration_sec,
            sample_rate=sr,
            note=note,
        )


def generate_audio(
    prompt: str,
    out_dir: str,
    filename: str = "audio.wav",
    device: str = "cpu",
    deterministic: bool = True,
    seed: int = 42,
) -> str:
    """
    Generate audio for a prompt. Returns path to audio file.

    Uses deterministic fallback by default (stable for experiments).
    """
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator = AudioGenerator(device=device)
    seed_value = seed if deterministic else None
    result = generator.generate(prompt=prompt, out_path=str(out_path), seed=seed_value)
    return result.audio_path


def generate_audio_with_metadata(
    prompt: str,
    out_dir: str,
    filename: str = "audio.wav",
    device: str = "cpu",
    deterministic: bool = True,
    seed: int = 42,
) -> AudioGenResult:
    """
    Generate audio and return full metadata.

    Use this in experiment pipelines where backend tracking matters.
    """
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator = AudioGenerator(device=device)
    seed_value = seed if deterministic else None
    return generator.generate(prompt=prompt, out_path=str(out_path), seed=seed_value)
