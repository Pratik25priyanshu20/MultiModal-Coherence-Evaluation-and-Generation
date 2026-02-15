"""
Learned Projection Heads for Embedding Alignment

Trains projection heads to map CLIP, CLAP, and text embeddings
to a shared semantic space where cross-modal similarity is meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def check_torch():
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for learned projections. "
            "Install with: pip install torch"
        )


class AlignedProjectionHead(nn.Module):
    """
    Single projection head that maps embeddings to a shared space.

    Architecture:
        input_dim → hidden_dim → ReLU → output_dim → L2 normalize
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 384,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        check_torch()
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2 normalize."""
        projected = self.projection(x)
        return F.normalize(projected, p=2, dim=-1)


class LearnedProjection(nn.Module):
    """
    Multi-modal projection that aligns text, image, and audio embeddings.

    Three separate projection heads map each modality to a shared space.
    """

    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 512,
        audio_dim: int = 512,
        shared_dim: int = 256,
        hidden_dim: int = 384,
        dropout: float = 0.1,
    ):
        check_torch()
        super().__init__()

        self.text_proj = AlignedProjectionHead(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=shared_dim,
            dropout=dropout,
        )
        self.image_proj = AlignedProjectionHead(
            input_dim=image_dim,
            hidden_dim=hidden_dim,
            output_dim=shared_dim,
            dropout=dropout,
        )
        self.audio_proj = AlignedProjectionHead(
            input_dim=audio_dim,
            hidden_dim=hidden_dim,
            output_dim=shared_dim,
            dropout=dropout,
        )

        self.shared_dim = shared_dim
        self.config = {
            "text_dim": text_dim,
            "image_dim": image_dim,
            "audio_dim": audio_dim,
            "shared_dim": shared_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        image_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Project embeddings to shared space.

        Args:
            text_emb: Text embeddings [batch, text_dim]
            image_emb: Image embeddings [batch, image_dim]
            audio_emb: Audio embeddings [batch, audio_dim]

        Returns:
            Dictionary with projected embeddings
        """
        result = {}

        if text_emb is not None:
            result["text"] = self.text_proj(text_emb)
        if image_emb is not None:
            result["image"] = self.image_proj(image_emb)
        if audio_emb is not None:
            result["audio"] = self.audio_proj(audio_emb)

        return result

    def project_text(self, emb: np.ndarray) -> np.ndarray:
        """Project text embedding (numpy interface)."""
        check_torch()
        with torch.no_grad():
            tensor = torch.tensor(emb, dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            projected = self.text_proj(tensor)
            return projected.numpy().squeeze()

    def project_image(self, emb: np.ndarray) -> np.ndarray:
        """Project image embedding (numpy interface)."""
        check_torch()
        with torch.no_grad():
            tensor = torch.tensor(emb, dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            projected = self.image_proj(tensor)
            return projected.numpy().squeeze()

    def project_audio(self, emb: np.ndarray) -> np.ndarray:
        """Project audio embedding (numpy interface)."""
        check_torch()
        with torch.no_grad():
            tensor = torch.tensor(emb, dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            projected = self.audio_proj(tensor)
            return projected.numpy().squeeze()

    def save(self, path: Path):
        """Save model weights and config."""
        check_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), path)

        # Save config
        config_path = path.with_suffix(".json")
        with config_path.open("w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LearnedProjection":
        """Load model from saved weights."""
        check_torch()
        path = Path(path)

        # Load config
        config_path = path.with_suffix(".json")
        with config_path.open("r") as f:
            config = json.load(f)

        # Create model
        model = cls(**config)

        # Load weights
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)

        return model


class ProjectedEmbedder:
    """
    Wrapper that applies learned projections to raw embeddings.

    Use this to integrate learned projections with existing embedder.
    """

    def __init__(
        self,
        base_embedder,
        projection: LearnedProjection,
    ):
        """
        Args:
            base_embedder: AlignedEmbedder or similar
            projection: Trained LearnedProjection model
        """
        self.base = base_embedder
        self.projection = projection
        self.projection.eval()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text with projection."""
        raw = self.base.embed_text(text)
        return self.projection.project_text(raw)

    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed image with projection."""
        raw = self.base.embed_image(image_path)
        return self.projection.project_image(raw)

    def embed_audio(self, audio_path: str) -> np.ndarray:
        """Embed audio with projection."""
        raw = self.base.embed_audio(audio_path)
        return self.projection.project_audio(raw)

    @property
    def target_dim(self) -> int:
        """Dimension of projected embeddings."""
        return self.projection.shared_dim
