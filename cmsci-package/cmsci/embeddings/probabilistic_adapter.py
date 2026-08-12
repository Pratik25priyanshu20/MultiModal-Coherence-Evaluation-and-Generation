"""
ProbVLM-Style Probabilistic Adapter for Uncertainty Estimation.

Converts point embeddings into distributions (Generalized Gaussian)
following the BayesCap approach from ProbVLM.

Each adapter takes a frozen embedding and predicts:
    mu:    Shift from the input embedding (residual)
    alpha: Scale parameter (controls spread)
    beta:  Shape parameter (controls tail behavior)

These define a Generalized Gaussian distribution:
    p(x) ∝ exp(-(|x - mu| / alpha)^beta)

MC sampling from this distribution produces N embedding samples,
which propagate uncertainty through the Gramian volume computation.

Architecture: BayesCap_MLP
    input → Linear(d, hidden) → ReLU → Dropout
          → Linear(hidden, hidden) → ReLU → Dropout
          → Three heads: mu_head, alpha_head, beta_head
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for ProbabilisticAdapter")


class ProbabilisticAdapter(nn.Module):
    """
    BayesCap-style adapter that maps point embeddings to distributions.

    Takes a frozen embedding (from CLIP or CLAP) and predicts
    Generalized Gaussian parameters: (mu, alpha, beta).

    The adapter is lightweight (~0.5M params) and trains in minutes
    on small datasets.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        _check_torch()
        super().__init__()

        self.input_dim = input_dim

        # Shared backbone
        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Three output heads
        self.mu_head = nn.Linear(hidden_dim, input_dim)
        self.alpha_head = nn.Linear(hidden_dim, input_dim)
        self.beta_head = nn.Linear(hidden_dim, input_dim)

        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    def forward(
        self, embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict distribution parameters from a point embedding.

        Args:
            embedding: Input embedding [batch, input_dim].

        Returns:
            mu: Location parameter [batch, input_dim] (embedding + residual)
            alpha: Scale parameter [batch, input_dim] (> 0, via softplus)
            beta: Shape parameter [batch, input_dim] (> 0, via softplus)
        """
        h = self.backbone(embedding)

        # mu: residual + input (anchored to original embedding)
        mu = embedding + self.mu_head(h)

        # alpha, beta: positive via softplus
        alpha = F.softplus(self.alpha_head(h)) + 1e-6
        beta = F.softplus(self.beta_head(h)) + 1e-6

        return mu, alpha, beta

    def sample(
        self,
        embedding: np.ndarray,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Draw Monte Carlo samples from the predicted distribution.

        Uses the reparameterization trick for Generalized Gaussian:
            x = mu + alpha * sign(u) * |u|^(1/beta)
        where u ~ Uniform(-1, 1)

        Args:
            embedding: Input embedding, shape (dim,) or (1, dim).
            n_samples: Number of MC samples.

        Returns:
            Samples array, shape (n_samples, dim).
        """
        _check_torch()
        self.eval()

        emb = embedding.squeeze()
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]

        with torch.no_grad():
            x = torch.tensor(emb, dtype=torch.float32)
            mu, alpha, beta = self.forward(x)

            # Expand for sampling: [1, dim] -> [n_samples, dim]
            mu = mu.expand(n_samples, -1)
            alpha = alpha.expand(n_samples, -1)
            beta = beta.expand(n_samples, -1)

            # Reparameterized sampling from Generalized Gaussian
            u = torch.rand_like(mu) * 2 - 1  # Uniform(-1, 1)
            sign = torch.sign(u)
            samples = mu + alpha * sign * (torch.abs(u) + 1e-8).pow(1.0 / beta)

            # L2 normalize samples (stay on unit sphere)
            samples = F.normalize(samples, p=2, dim=-1)

        return samples.cpu().numpy()

    def uncertainty(self, embedding: np.ndarray) -> float:
        """
        Compute scalar aleatoric uncertainty for an embedding.

        Returns the mean predicted alpha (scale parameter) across dimensions.
        High alpha → high uncertainty → wide distribution.

        Args:
            embedding: Input embedding, shape (dim,) or (1, dim).

        Returns:
            Scalar uncertainty value (mean alpha).
        """
        _check_torch()
        self.eval()

        emb = embedding.squeeze()
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]

        with torch.no_grad():
            x = torch.tensor(emb, dtype=torch.float32)
            _, alpha, _ = self.forward(x)
            return float(alpha.mean().item())

    def save(self, path: str) -> None:
        """Save adapter weights + config."""
        _check_torch()
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)
        config_path = p.with_suffix(".json")
        with config_path.open("w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("Saved ProbabilisticAdapter to %s", path)

    @classmethod
    def load(cls, path: str) -> "ProbabilisticAdapter":
        """Load adapter from saved weights."""
        _check_torch()
        import json
        p = Path(path)
        config_path = p.with_suffix(".json")
        with config_path.open("r") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(p, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Loaded ProbabilisticAdapter from %s", path)
        return model
