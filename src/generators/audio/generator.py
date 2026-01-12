from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
from pathlib import Path


@dataclass(frozen=True)
class AudioGenResult:
    audio_path: str
    backend: str
    note: Optional[str] = None


class AudioGenerator:
    """
    V1 soundscape generator.
    - Primary: AudioLDM via diffusers (if available)
    - Fallback: lightweight ambient synth (always works)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._audioldm_pipe = None
        self._audioldm_backend_name = None
        self._torch = None

        try:
            from diffusers import AudioLDMPipeline  # type: ignore
            import torch  # type: ignore

            model_id = "cvssp/audioldm"
            self._audioldm_pipe = AudioLDMPipeline.from_pretrained(model_id)
            self._audioldm_pipe.to(self.device)
            self._audioldm_backend_name = f"AudioLDMPipeline({model_id})"
            self._torch = torch
        except Exception as exc:
            self._audioldm_pipe = None
            self._audioldm_backend_name = None
            self._torch = None
            self._init_error = str(exc)

    def generate(
        self,
        prompt: str,
        out_path: str,
        duration_sec: float = 6.0,
        sr: int = 48000,
        seed: Optional[int] = None,
    ) -> AudioGenResult:
        if self._audioldm_pipe is not None:
            try:
                generator = None
                if seed is not None and self._torch is not None:
                    generator = self._torch.Generator(device=self.device).manual_seed(seed)
                kwargs = {"audio_length_in_s": duration_sec}
                if generator is not None:
                    kwargs["generator"] = generator
                result = self._audioldm_pipe(prompt, **kwargs)
                audio = result.audios[0]
                sf.write(out_path, audio, sr)
                return AudioGenResult(
                    audio_path=out_path,
                    backend=self._audioldm_backend_name or "audioldm",
                    note=None,
                )
            except Exception as exc:
                return self._fallback_ambient(
                    prompt,
                    out_path,
                    duration_sec,
                    sr,
                    note=f"AudioLDM failed: {exc}",
                    seed=seed,
                )

        return self._fallback_ambient(
            prompt,
            out_path,
            duration_sec,
            sr,
            note=f"AudioLDM unavailable: {getattr(self, '_init_error', 'unknown')}",
            seed=seed,
        )

    def _fallback_ambient(
        self,
        prompt: str,
        out_path: str,
        duration_sec: float,
        sr: int,
        note: str,
        seed: Optional[int] = None,
    ) -> AudioGenResult:
        base_seed = abs(hash(prompt)) % (2**32)
        if seed is not None:
            base_seed = (base_seed + seed) % (2**32)
        rng = np.random.default_rng(base_seed)

        n = int(duration_sec * sr)
        t = np.linspace(0, duration_sec, n, endpoint=False)

        noise = rng.normal(0, 1, size=n).astype(np.float32)
        noise = np.convolve(
            noise, np.ones(4000, dtype=np.float32) / 4000.0, mode="same"
        )
        drone = 0.08 * np.sin(2 * np.pi * 110.0 * t).astype(np.float32)

        audio = (0.03 * noise + drone).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        sf.write(out_path, audio, sr)
        return AudioGenResult(audio_path=out_path, backend="fallback_ambient", note=note)


def generate_audio(
    prompt: str,
    out_dir: str,
    filename: str = "audio.wav",
    device: str = "cpu",
    deterministic: bool = True,
    seed: int = 42,
) -> str:
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator = AudioGenerator(device=device)
    seed_value = seed if deterministic else None
    result = generator.generate(prompt=prompt, out_path=str(out_path), seed=seed_value)
    return result.audio_path
