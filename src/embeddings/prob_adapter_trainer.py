"""
Training Loop for ProbVLM-Style Probabilistic Adapters.

Trains lightweight post-hoc adapters on top of frozen CLIP/CLAP encoders.
Each adapter learns to predict uncertainty (Generalized Gaussian parameters)
for a single embedding space.

Two adapters to train:
    1. CLIP adapter: trained on (image_embedding, text_embedding) pairs
    2. CLAP adapter: trained on (audio_embedding, text_embedding) pairs

Training data:
    - Our 57 images paired with text descriptions (CLIP pairs)
    - Our 104 audio files paired with text descriptions (CLAP pairs)
    - All 30 RQ1 prompts × matched media as additional pairs

Loss:
    L = L1(mu, target) + GenGaussLoss(mu, alpha, beta, target)

GenGaussLoss:
    -log p(target | mu, alpha, beta) ∝ log(alpha) - log(beta) + (|target - mu| / alpha)^beta
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.embeddings.probabilistic_adapter import ProbabilisticAdapter


class EmbeddingPairDataset(Dataset):
    """Dataset of (input_embedding, target_embedding) pairs."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        assert len(inputs) == len(targets)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class GenGaussNLL(nn.Module):
    """
    Negative log-likelihood loss for Generalized Gaussian distribution.

    -log p(x | mu, alpha, beta) = log(2*alpha) + log(Gamma(1/beta)/beta) + (|x - mu| / alpha)^beta

    Simplified (dropping constant terms):
        L = log(alpha) + (|target - mu| / alpha)^beta
    """

    def forward(
        self,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        residual = torch.abs(target - mu)
        # Clamp alpha to avoid division by zero
        alpha_c = torch.clamp(alpha, min=1e-6)
        nll = torch.log(alpha_c) + (residual / alpha_c).pow(beta)
        return nll.mean()


def train_prob_adapter(
    input_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 32,
    val_split: float = 0.15,
    patience: int = 15,
    output_path: Optional[str] = None,
    adapter_name: str = "adapter",
) -> ProbabilisticAdapter:
    """
    Train a ProbabilisticAdapter on paired embeddings.

    Args:
        input_embeddings: Source embeddings [N, 512] (e.g. image CLIP or audio CLAP).
        target_embeddings: Target embeddings [N, 512] (e.g. text CLIP or text CLAP).
        epochs: Maximum training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        val_split: Fraction for validation.
        patience: Early stopping patience.
        output_path: If set, save best model here.
        adapter_name: Name for logging.

    Returns:
        Trained ProbabilisticAdapter.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for training")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Build dataset
    dataset = EmbeddingPairDataset(input_embeddings, target_embeddings)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    input_dim = input_embeddings.shape[1]
    adapter = ProbabilisticAdapter(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    l1_loss = nn.L1Loss()
    gg_loss = GenGaussNLL()

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(
        "Training %s adapter: %d train, %d val, %d epochs, device=%s",
        adapter_name, n_train, n_val, epochs, device,
    )

    for epoch in range(epochs):
        # Train
        adapter.train()
        train_losses = []
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()

            mu, alpha, beta = adapter(inp)
            loss = l1_loss(mu, tgt) + gg_loss(mu, alpha, beta, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        adapter.eval()
        val_losses = []
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mu, alpha, beta = adapter(inp)
                loss = l1_loss(mu, tgt) + gg_loss(mu, alpha, beta, tgt)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses) if val_losses else float("inf")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "  [%s] Epoch %d/%d: train=%.4f, val=%.4f",
                adapter_name, epoch + 1, epochs, avg_train, avg_val,
            )

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            if output_path:
                adapter.save(output_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  [%s] Early stopping at epoch %d", adapter_name, epoch + 1)
                break

    # Load best if saved
    if output_path and Path(output_path).exists():
        adapter = ProbabilisticAdapter.load(output_path)
        adapter = adapter.to(device)
    else:
        adapter = adapter.cpu()

    adapter.eval()
    logger.info("  [%s] Training complete. Best val_loss=%.4f", adapter_name, best_val_loss)
    return adapter


def build_training_pairs_from_index(
    embedding_index_path: str,
    text_embedder_fn,
    modality: str = "image",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (media_embedding, text_embedding) pairs from an embedding index.

    For each media file in the index, generates a text description from
    the filename/metadata and embeds it.

    Args:
        embedding_index_path: Path to image_index.npz or audio_index.npz.
        text_embedder_fn: Function that takes text -> np.ndarray embedding.
        modality: "image" for CLIP text, "audio" for CLAP text.

    Returns:
        (media_embeddings, text_embeddings) both shape [N, 512].
    """
    data = np.load(embedding_index_path, allow_pickle=True)
    ids = data["ids"] if "ids" in data else data.get("paths", np.array([]))
    embs = data["embs"] if "embs" in data else data.get("embeddings", np.array([]))
    domains = data["domains"] if "domains" in data else np.array(["other"] * len(ids))

    media_embs = []
    text_embs = []

    for i, (file_id, domain) in enumerate(zip(ids, domains)):
        # Generate caption from filename
        name = Path(str(file_id)).stem
        # Clean up filename to make a caption
        caption = name.replace("_", " ").replace("-", " ")
        # Remove common prefixes
        for prefix in ["fs ", "wm ", "proc "]:
            if caption.lower().startswith(prefix):
                caption = caption[len(prefix):]
        # Add domain context
        if domain != "other":
            caption = f"{domain}: {caption}"

        try:
            text_emb = text_embedder_fn(caption)
            media_embs.append(embs[i])
            text_embs.append(text_emb)
        except Exception as e:
            logger.warning("Skipping %s: %s", file_id, e)

    return np.array(media_embs), np.array(text_embs)
