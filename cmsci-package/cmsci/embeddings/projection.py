from __future__ import annotations

import numpy as np
import torch


class ProjectionHead:
    """
    Projects embeddings from arbitrary dim -> shared dim.

    When in_dim == out_dim: uses IDENTITY (pass-through).
    This preserves pre-trained alignment (CLIP text-image, CLAP text-audio).
    A random linear projection would destroy that alignment.

    When in_dim != out_dim: uses a linear layer (would need training for
    meaningful results; acceptable only if you train it).
    """

    def __init__(self, in_dim: int, out_dim: int = 512):
        self._identity = (in_dim == out_dim)
        self.layer = None
        if not self._identity:
            self.layer = torch.nn.Linear(in_dim, out_dim, bias=False)
            self.layer.eval()

    @torch.no_grad()
    def project(self, emb: np.ndarray) -> np.ndarray:
        if self._identity:
            return emb.astype("float32")
        x = torch.from_numpy(emb).float()
        y = self.layer(x)
        return y.numpy().astype("float32")
