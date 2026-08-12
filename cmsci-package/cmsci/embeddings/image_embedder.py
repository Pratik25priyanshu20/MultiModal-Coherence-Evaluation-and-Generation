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
        feats = self.model.get_image_features(**inputs)
        # Handle different transformers versions
        target_dim = getattr(self.model.config, "projection_dim", 512)
        if not isinstance(feats, torch.Tensor):
            pooled = feats.pooler_output
            if pooled.shape[-1] != target_dim:
                proj = getattr(self.model, "visual_projection", None)
                if proj is not None:
                    pooled = proj(pooled)
            feats = pooled
        if feats.dim() == 3:
            pooled = feats[:, 0, :]
            if pooled.shape[-1] != target_dim:
                proj = getattr(self.model, "visual_projection", None)
                if proj is not None:
                    pooled = proj(pooled)
            feats = pooled
        if feats.dim() == 2:
            feats = feats[0]
        return feats.cpu().numpy().astype("float32")
