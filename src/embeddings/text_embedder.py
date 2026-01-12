from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


class TextEmbedder:
    """
    CLIP projected text features (512-d).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)[0]
        return feats.cpu().numpy().astype("float32")
