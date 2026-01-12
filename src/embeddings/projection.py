from __future__ import annotations

import numpy as np
import torch


class ProjectionHead:
    """
    Projects embeddings from arbitrary dim -> shared dim.
    No training yet (random init) – fine for Phase-2 V1.
    """

    def __init__(self, in_dim: int, out_dim: int = 512):
        self.layer = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.layer.eval()

    @torch.no_grad()
    def project(self, emb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(emb).float()
        y = self.layer(x)
        return y.numpy().astype("float32")
