"""
Ex-MCR Cross-Space Alignment: CLAP Audio → CLIP Space.

Ex-MCR (Ex-Modal Contrastive Retrieval) projects CLAP audio embeddings
INTO CLIP space while keeping CLIP embeddings unchanged. This lets us
compute meaningful image-audio similarity and full 3-way Gramian volume.

Architecture decision: Ex-MCR over C-MCR because:
- Ex-MCR keeps CLIP embeddings frozen (no recomputation needed)
- C-MCR projects BOTH spaces into a new space (breaks everything)

The projector is a lightweight MLP:
    CLAP 512-d → Linear(512, 512) → ReLU → Linear(512, 512) → L2 norm

If Ex-MCR weights are not available, falls back to an untrained identity
projection (which is equivalent to not using the projector).

CLAP compatibility note:
    Our project uses `laion/clap-htsat-unfused`.
    Ex-MCR uses `laion_clap_fullset_fusion` (different model).
    If projections are poor with our CLAP, switch to the fusion model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExMCRProjector:
    """
    Projects CLAP audio embeddings into CLIP space.

    Usage:
        proj = ExMCRProjector("models/exmcr/ex_clap.pt")
        audio_in_clip = proj.project_audio(clap_embedding)  # now comparable to CLIP
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            weights_path: Path to Ex-MCR CLAP→CLIP projection weights (.pt).
                If None or file doesn't exist, uses identity (passthrough).
            device: Torch device for inference.
        """
        self._model = None
        self._device = device
        self._identity_mode = True

        if weights_path and Path(weights_path).exists() and TORCH_AVAILABLE:
            self._load_weights(weights_path)
        elif weights_path and not Path(weights_path).exists():
            logger.warning(
                "Ex-MCR weights not found: %s — using identity projection", weights_path
            )

    def _load_weights(self, path: str) -> None:
        """Load Ex-MCR projection head from saved weights."""
        state_dict = torch.load(path, map_location=self._device, weights_only=True)

        # Detect architecture from state dict keys
        # Ex-MCR uses: layers.0.weight, layers.0.bias, layers.2.weight, layers.2.bias
        # or: 0.weight, 0.bias, 2.weight, 2.bias
        keys = list(state_dict.keys())

        # Build matching MLP
        if any("layers" in k for k in keys):
            # Format: layers.0.weight etc.
            in_dim = state_dict["layers.0.weight"].shape[1]
            hidden_dim = state_dict["layers.0.weight"].shape[0]
            out_dim = state_dict["layers.2.weight"].shape[0]
            model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            # Rename keys to match sequential
            new_state = {}
            for k, v in state_dict.items():
                new_key = k.replace("layers.", "")
                new_state[new_key] = v
            model.load_state_dict(new_state)
        elif any(k.startswith("0.") for k in keys):
            # Format: 0.weight, 0.bias, 2.weight, 2.bias (Sequential)
            in_dim = state_dict["0.weight"].shape[1]
            hidden_dim = state_dict["0.weight"].shape[0]
            out_dim = state_dict["2.weight"].shape[0]
            model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            model.load_state_dict(state_dict)
        else:
            # Generic: try to infer from weight shapes
            weight_keys = [k for k in keys if "weight" in k]
            if len(weight_keys) >= 2:
                first_w = state_dict[weight_keys[0]]
                last_w = state_dict[weight_keys[-1]]
                in_dim = first_w.shape[1]
                hidden_dim = first_w.shape[0]
                out_dim = last_w.shape[0]
                model = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim),
                )
                model.load_state_dict(state_dict)
            else:
                logger.warning("Unrecognized Ex-MCR weight format — using identity")
                return

        model.to(self._device)
        model.eval()
        self._model = model
        self._identity_mode = False
        logger.info(
            "Ex-MCR projector loaded: %d → %d → %d (from %s)",
            in_dim, hidden_dim, out_dim, path,
        )

    @property
    def is_identity(self) -> bool:
        """True if projector is passthrough (no trained weights loaded)."""
        return self._identity_mode

    def project_audio(self, clap_embedding: np.ndarray) -> np.ndarray:
        """
        Project CLAP audio embedding into CLIP space.

        Args:
            clap_embedding: CLAP audio embedding, shape (512,) or (N, 512).

        Returns:
            Projected embedding in CLIP space, L2-normalized.
        """
        if self._identity_mode:
            emb = clap_embedding.squeeze().astype(np.float32)
            norm = np.linalg.norm(emb) + 1e-12
            return emb / norm

        if not TORCH_AVAILABLE:
            return clap_embedding.squeeze().astype(np.float32)

        was_1d = clap_embedding.ndim == 1 or (
            clap_embedding.ndim == 2 and clap_embedding.shape[0] == 1
        )
        emb = clap_embedding.squeeze()
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]

        with torch.no_grad():
            x = torch.tensor(emb, dtype=torch.float32, device=self._device)
            projected = self._model(x)
            projected = F.normalize(projected, p=2, dim=-1)
            result = projected.cpu().numpy()

        if was_1d:
            return result.squeeze(0)
        return result

    def project_audio_batch(self, clap_embeddings: np.ndarray) -> np.ndarray:
        """
        Batch projection of CLAP audio embeddings into CLIP space.

        Args:
            clap_embeddings: Shape (N, 512).

        Returns:
            Projected embeddings in CLIP space, shape (N, 512), L2-normalized.
        """
        if self._identity_mode:
            norms = np.linalg.norm(clap_embeddings, axis=1, keepdims=True) + 1e-12
            return (clap_embeddings / norms).astype(np.float32)

        if not TORCH_AVAILABLE:
            norms = np.linalg.norm(clap_embeddings, axis=1, keepdims=True) + 1e-12
            return (clap_embeddings / norms).astype(np.float32)

        with torch.no_grad():
            x = torch.tensor(clap_embeddings, dtype=torch.float32, device=self._device)
            projected = self._model(x)
            projected = F.normalize(projected, p=2, dim=-1)
            return projected.cpu().numpy()
