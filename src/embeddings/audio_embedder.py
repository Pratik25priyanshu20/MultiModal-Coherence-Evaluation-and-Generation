from __future__ import annotations

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor


class AudioEmbedder:
    """
    CLAP-based audio embedder.
    Optimized for environmental soundscape semantics.
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cpu",
        target_sr: int = 48000,
    ):
        self.device = device
        self.target_sr = target_sr
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, audio_path: str) -> np.ndarray:
        waveform, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)

        inputs = self.processor(
            audios=waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.get_audio_features(**inputs)
        emb = outputs[0]
        return emb.cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using CLAP's text encoder (for text-audio comparison)."""
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)[0]
        return feats.cpu().numpy().astype("float32")
