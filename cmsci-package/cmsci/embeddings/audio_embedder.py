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

    def _extract_features(self, output, projection: str) -> torch.Tensor:
        """Extract 1-D projected embedding (512-d) from model output.

        Handles both raw tensors and BaseModelOutputWithPooling objects
        across different transformers versions.
        """
        target_dim = getattr(self.model.config, "projection_dim", 512)
        if not isinstance(output, torch.Tensor):
            # BaseModelOutputWithPooling — extract pooled features
            pooled = output.pooler_output
            # Only project if not already at target dim
            if pooled.shape[-1] != target_dim:
                proj = getattr(self.model, projection, None)
                if proj is not None:
                    pooled = proj(pooled)
            output = pooled
        if output.dim() == 3:
            pooled = output[:, 0, :]
            if pooled.shape[-1] != target_dim:
                proj = getattr(self.model, projection, None)
                if proj is not None:
                    pooled = proj(pooled)
            output = pooled
        if output.dim() == 2:
            output = output[0]
        return output

    @torch.no_grad()
    def embed(self, audio_path: str) -> np.ndarray:
        waveform, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)

        # Use 'audio' (newer transformers) with fallback to 'audios' (older)
        try:
            inputs = self.processor(
                audio=waveform,
                sampling_rate=self.target_sr,
                return_tensors="pt",
            ).to(self.device)
        except TypeError:
            inputs = self.processor(
                audios=waveform,
                sampling_rate=self.target_sr,
                return_tensors="pt",
            ).to(self.device)

        outputs = self.model.get_audio_features(**inputs)
        emb = self._extract_features(outputs, "audio_projection")
        return emb.cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using CLAP's text encoder (for text-audio comparison)."""
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = self._extract_features(feats, "text_projection")
        return feats.cpu().numpy().astype("float32")
