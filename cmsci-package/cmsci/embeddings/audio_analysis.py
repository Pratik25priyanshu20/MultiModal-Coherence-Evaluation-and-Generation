"""
Audio Silence Detection + Windowed CLAP Embeddings.

Provides tools for:
1. Silence/near-silence detection via RMS energy + spectral flatness
2. Windowed CLAP: split audio into overlapping windows, embed each,
   aggregate via max-pooling or attention-weighted mean

Usage:
    from cmsci.embeddings.audio_analysis import AudioAnalyzer
    analyzer = AudioAnalyzer()
    report = analyzer.analyze(audio_path)
    windowed_emb = analyzer.embed_windowed(audio_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np


@dataclass
class AudioQualityReport:
    """Report on audio file quality for coherence evaluation."""
    path: str
    duration_sec: float
    rms_energy: float
    rms_db: float
    spectral_flatness_mean: float
    spectral_flatness_std: float
    silence_fraction: float  # fraction of frames below silence threshold
    is_mostly_silence: bool
    is_low_complexity: bool
    n_windows: int  # number of CLAP windows
    issues: List[str]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "duration_sec": round(self.duration_sec, 2),
            "rms_energy": round(self.rms_energy, 6),
            "rms_db": round(self.rms_db, 1),
            "spectral_flatness_mean": round(self.spectral_flatness_mean, 4),
            "spectral_flatness_std": round(self.spectral_flatness_std, 4),
            "silence_fraction": round(self.silence_fraction, 3),
            "is_mostly_silence": self.is_mostly_silence,
            "is_low_complexity": self.is_low_complexity,
            "n_windows": self.n_windows,
            "issues": self.issues,
        }


class AudioAnalyzer:
    """Audio quality analysis and windowed CLAP embedding.

    Args:
        target_sr: Target sample rate for analysis.
        silence_rms_db: RMS threshold (dB) below which a frame is considered silent.
        silence_fraction_threshold: Fraction of silent frames to flag as mostly-silence.
        spectral_flatness_threshold: Mean spectral flatness above which audio is low-complexity.
        window_sec: CLAP window duration in seconds.
        hop_sec: CLAP window hop in seconds.
    """

    def __init__(
        self,
        target_sr: int = 48000,
        silence_rms_db: float = -40.0,
        silence_fraction_threshold: float = 0.7,
        spectral_flatness_threshold: float = 0.85,
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
    ):
        self.target_sr = target_sr
        self.silence_rms_db = silence_rms_db
        self.silence_fraction_threshold = silence_fraction_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        self.window_sec = window_sec
        self.hop_sec = hop_sec

    def analyze(self, audio_path: str) -> AudioQualityReport:
        """Analyze audio file for silence and spectral quality.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioQualityReport with quality metrics and flags.
        """
        waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        duration = len(waveform) / sr

        # RMS energy
        rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
        rms_mean = float(np.mean(rms))
        rms_db = float(20 * np.log10(max(rms_mean, 1e-10)))

        # Per-frame silence detection
        rms_db_frames = 20 * np.log10(np.maximum(rms, 1e-10))
        silence_mask = rms_db_frames < self.silence_rms_db
        silence_fraction = float(np.mean(silence_mask))

        # Spectral flatness (1.0 = white noise, 0.0 = pure tone)
        sflatness = librosa.feature.spectral_flatness(y=waveform, hop_length=512)[0]
        sf_mean = float(np.mean(sflatness))
        sf_std = float(np.std(sflatness))

        # Flags
        is_mostly_silence = silence_fraction >= self.silence_fraction_threshold
        is_low_complexity = sf_mean >= self.spectral_flatness_threshold

        # Count CLAP windows
        window_samples = int(self.window_sec * sr)
        hop_samples = int(self.hop_sec * sr)
        n_windows = max(1, 1 + (len(waveform) - window_samples) // hop_samples) if len(waveform) >= window_samples else 1

        # Collect issues
        issues = []
        if is_mostly_silence:
            issues.append(f"mostly_silence ({silence_fraction:.0%} silent frames)")
        if is_low_complexity:
            issues.append(f"low_spectral_complexity (flatness={sf_mean:.3f})")
        if duration < 1.0:
            issues.append(f"very_short ({duration:.1f}s)")
        if rms_db < -50:
            issues.append(f"very_quiet ({rms_db:.0f} dB)")

        return AudioQualityReport(
            path=audio_path,
            duration_sec=duration,
            rms_energy=rms_mean,
            rms_db=rms_db,
            spectral_flatness_mean=sf_mean,
            spectral_flatness_std=sf_std,
            silence_fraction=silence_fraction,
            is_mostly_silence=is_mostly_silence,
            is_low_complexity=is_low_complexity,
            n_windows=n_windows,
            issues=issues,
        )

    def get_windows(self, audio_path: str) -> List[np.ndarray]:
        """Split audio into overlapping windows for windowed CLAP.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of waveform windows (each shape [window_samples]).
        """
        waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

        window_samples = int(self.window_sec * sr)
        hop_samples = int(self.hop_sec * sr)

        # If audio is shorter than one window, return the full waveform
        if len(waveform) < window_samples:
            # Pad to window size
            padded = np.zeros(window_samples, dtype=waveform.dtype)
            padded[:len(waveform)] = waveform
            return [padded]

        windows = []
        start = 0
        while start + window_samples <= len(waveform):
            windows.append(waveform[start:start + window_samples])
            start += hop_samples

        # Include final partial window if it covers at least 50% of window_sec
        remaining = len(waveform) - start
        if remaining >= window_samples // 2:
            padded = np.zeros(window_samples, dtype=waveform.dtype)
            padded[:remaining] = waveform[start:]
            windows.append(padded)

        return windows

    def _is_window_silent(self, window: np.ndarray) -> bool:
        """Check if a single waveform window is mostly silent.

        Uses the same RMS-dB threshold as the full-file analyzer.
        A window is silent if the fraction of frames below the threshold
        exceeds silence_fraction_threshold.
        """
        rms = librosa.feature.rms(y=window, frame_length=2048, hop_length=512)[0]
        rms_db_frames = 20 * np.log10(np.maximum(rms, 1e-10))
        silent_frames = rms_db_frames < self.silence_rms_db
        return float(np.mean(silent_frames)) >= self.silence_fraction_threshold

    def embed_windowed(
        self,
        audio_path: str,
        embedder=None,
        aggregation: str = "max",
        drop_silent: bool = True,
    ) -> np.ndarray:
        """Embed audio using windowed CLAP with silence filtering + aggregation.

        Splits audio into overlapping windows, drops silent windows, embeds
        only the active windows with CLAP, and aggregates the results.

        If all windows are silent, falls back to the full-clip embedding
        so the pipeline never returns None.

        Args:
            audio_path: Path to audio file.
            embedder: AudioEmbedder instance (created if None).
            aggregation: "max" (element-wise max-pooling) or "mean" (average).
            drop_silent: If True, skip windows that are mostly silent.

        Returns:
            Aggregated 512-d embedding.
        """
        if embedder is None:
            from cmsci.embeddings.audio_embedder import AudioEmbedder
            embedder = AudioEmbedder()

        windows = self.get_windows(audio_path)

        if len(windows) == 1:
            # Just use the full-clip embedding for single-window audio
            return embedder.embed(audio_path)

        # Filter out silent windows
        if drop_silent:
            active_windows = [w for w in windows if not self._is_window_silent(w)]
            n_dropped = len(windows) - len(active_windows)
            if n_dropped > 0:
                import logging
                logging.getLogger(__name__).info(
                    "Windowed CLAP [%s]: dropped %d/%d silent windows",
                    audio_path, n_dropped, len(windows),
                )
            # If ALL windows are silent, fall back to full-clip embedding
            if not active_windows:
                import logging
                logging.getLogger(__name__).warning(
                    "Windowed CLAP [%s]: all %d windows silent, using full-clip fallback",
                    audio_path, len(windows),
                )
                return embedder.embed(audio_path)
            windows = active_windows

        # Embed each active window
        import tempfile
        import soundfile as sf
        embeddings = []

        for window in windows:
            # Write window to temp file for CLAP processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, window, self.target_sr)
                emb = embedder.embed(tmp.name)
                embeddings.append(emb)
                Path(tmp.name).unlink(missing_ok=True)

        embeddings = np.stack(embeddings)

        if aggregation == "max":
            return np.max(embeddings, axis=0)
        elif aggregation == "mean":
            return np.mean(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def analyze_all_rq3_audio(samples: list) -> List[AudioQualityReport]:
    """Run audio analysis on all RQ3 samples.

    Args:
        samples: List of sample dicts with 'audio_path' key.

    Returns:
        List of AudioQualityReport objects.
    """
    analyzer = AudioAnalyzer()
    reports = []

    for s in samples:
        audio_path = s.get("audio_path")
        if audio_path and Path(audio_path).exists():
            report = analyzer.analyze(audio_path)
            reports.append(report)

    return reports
