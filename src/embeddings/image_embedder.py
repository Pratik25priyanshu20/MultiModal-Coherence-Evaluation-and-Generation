from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEmbedder:
    """
    CLIP projected image features (512-d).
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
    def embed(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)[0]
        return feats.cpu().numpy().astype("float32")
