"""
Contrastive Training for Embedding Alignment

Trains projection heads using InfoNCE contrastive loss to align
text, image, and audio embeddings in a shared semantic space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.training.learned_projection import LearnedProjection, check_torch


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for multimodal alignment.

    Pulls positive pairs (same sample, different modalities) together
    and pushes negative pairs (different samples) apart.
    """

    def __init__(self, temperature: float = 0.07):
        check_torch()
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings [batch, dim]
            positive: Positive (aligned) embeddings [batch, dim]
            negatives: Negative embeddings [n_neg, dim] (optional, uses in-batch if None)

        Returns:
            Scalar loss
        """
        batch_size = anchor.size(0)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch]

        if negatives is None:
            # In-batch negatives: all other samples in batch
            # Similarity matrix: [batch, batch]
            sim_matrix = torch.mm(anchor, positive.t()) / self.temperature

            # Mask out positive pairs (diagonal)
            mask = torch.eye(batch_size, dtype=torch.bool, device=anchor.device)
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

            # Log-sum-exp over negatives
            neg_logsumexp = torch.logsumexp(sim_matrix, dim=-1)  # [batch]

        else:
            # Explicit negatives
            neg_sim = torch.mm(anchor, negatives.t()) / self.temperature  # [batch, n_neg]
            neg_logsumexp = torch.logsumexp(neg_sim, dim=-1)  # [batch]

        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # ≈ -pos + logsumexp([pos, neg1, neg2, ...])
        loss = -pos_sim + torch.logsumexp(
            torch.stack([pos_sim, neg_logsumexp], dim=-1), dim=-1
        )

        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet margin loss for multimodal alignment.

    Alternative to InfoNCE that explicitly uses anchor, positive, negative triplets.
    """

    def __init__(self, margin: float = 0.2):
        check_torch()
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch, dim]
            positive: Positive embeddings [batch, dim]
            negative: Negative embeddings [batch, dim]

        Returns:
            Scalar loss
        """
        pos_dist = 1 - F.cosine_similarity(anchor, positive)
        neg_dist = 1 - F.cosine_similarity(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


@dataclass
class TrainingConfig:
    """Configuration for contrastive training."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_epochs: int = 20
    temperature: float = 0.07
    warmup_steps: int = 100
    eval_every: int = 500
    save_every: int = 1000
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "temperature": self.temperature,
            "warmup_steps": self.warmup_steps,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "device": self.device,
        }


class MultimodalTripletDataset(Dataset):
    """
    Dataset of (text, image, audio) embedding triplets.

    Each sample contains embeddings from aligned multimodal content.
    """

    def __init__(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        audio_embeddings: np.ndarray,
    ):
        """
        Args:
            text_embeddings: [N, text_dim]
            image_embeddings: [N, image_dim]
            audio_embeddings: [N, audio_dim]
        """
        check_torch()
        assert len(text_embeddings) == len(image_embeddings) == len(audio_embeddings)

        self.text = torch.tensor(text_embeddings, dtype=torch.float32)
        self.image = torch.tensor(image_embeddings, dtype=torch.float32)
        self.audio = torch.tensor(audio_embeddings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "text": self.text[idx],
            "image": self.image[idx],
            "audio": self.audio[idx],
        }


class ContrastiveTrainer:
    """
    Trainer for multimodal embedding alignment.

    Trains projection heads using contrastive learning on
    aligned (text, image, audio) triplets.
    """

    def __init__(
        self,
        model: LearnedProjection,
        config: TrainingConfig,
        output_dir: Path = Path("models/projection"),
    ):
        check_torch()
        self.model = model.to(config.device)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.loss_fn = ContrastiveLoss(temperature=config.temperature)
        self.step = 0
        self.history: List[Dict[str, float]] = []

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        text = batch["text"].to(self.config.device)
        image = batch["image"].to(self.config.device)
        audio = batch["audio"].to(self.config.device)

        # Project all modalities
        projected = self.model(text_emb=text, image_emb=image, audio_emb=audio)
        p_text = projected["text"]
        p_image = projected["image"]
        p_audio = projected["audio"]

        # Compute losses for all pairs
        loss_ti = self.loss_fn(p_text, p_image)  # text-image
        loss_ta = self.loss_fn(p_text, p_audio)  # text-audio
        loss_ia = self.loss_fn(p_image, p_audio)  # image-audio

        # Combined loss (weighted)
        total_loss = 0.45 * loss_ti + 0.45 * loss_ta + 0.10 * loss_ia

        # Backward
        total_loss.backward()
        self.optimizer.step()

        self.step += 1

        return {
            "loss": total_loss.item(),
            "loss_ti": loss_ti.item(),
            "loss_ta": loss_ta.item(),
            "loss_ia": loss_ia.item(),
        }

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                text = batch["text"].to(self.config.device)
                image = batch["image"].to(self.config.device)
                audio = batch["audio"].to(self.config.device)

                projected = self.model(text_emb=text, image_emb=image, audio_emb=audio)
                p_text = projected["text"]
                p_image = projected["image"]
                p_audio = projected["audio"]

                loss_ti = self.loss_fn(p_text, p_image)
                loss_ta = self.loss_fn(p_text, p_audio)
                loss_ia = self.loss_fn(p_image, p_audio)

                total_loss += (0.45 * loss_ti + 0.45 * loss_ta + 0.10 * loss_ia).item()
                n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    def train(
        self,
        train_dataset: MultimodalTripletDataset,
        val_dataset: Optional[MultimodalTripletDataset] = None,
    ) -> LearnedProjection:
        """
        Full training loop.

        Args:
            train_dataset: Training triplets
            val_dataset: Optional validation triplets

        Returns:
            Trained model
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        print(f"Training on {len(train_dataset)} samples")
        print(f"Config: {self.config.to_dict()}")

        for epoch in range(self.config.n_epochs):
            epoch_losses = []

            for batch in train_loader:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["loss"])

                # Log
                if self.step % 100 == 0:
                    print(f"Step {self.step}: loss={metrics['loss']:.4f}")

                # Evaluate
                if val_loader and self.step % self.config.eval_every == 0:
                    val_metrics = self.evaluate(val_loader)
                    print(f"Step {self.step}: val_loss={val_metrics['val_loss']:.4f}")
                    self.history.append({"step": self.step, **val_metrics})

                # Save checkpoint
                if self.step % self.config.save_every == 0:
                    self._save_checkpoint(f"checkpoint_{self.step}")

            # Epoch summary
            mean_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}/{self.config.n_epochs}: mean_loss={mean_loss:.4f}")

        # Final save
        self._save_checkpoint("final")

        return self.model

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = self.output_dir / f"{name}.pt"
        self.model.save(path)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with history_path.open("w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Saved checkpoint: {path}")


def create_synthetic_triplets(
    embedder,
    prompts: List[str],
    image_paths: List[str],
    audio_paths: List[str],
) -> MultimodalTripletDataset:
    """
    Create triplet dataset from raw content.

    Args:
        embedder: Embedder with embed_text, embed_image, embed_audio methods
        prompts: List of text prompts
        image_paths: List of aligned image paths
        audio_paths: List of aligned audio paths

    Returns:
        MultimodalTripletDataset
    """
    check_torch()
    assert len(prompts) == len(image_paths) == len(audio_paths)

    text_embs = []
    image_embs = []
    audio_embs = []

    for i, (prompt, img, aud) in enumerate(zip(prompts, image_paths, audio_paths)):
        try:
            text_embs.append(embedder.embed_text(prompt))
            image_embs.append(embedder.embed_image(img))
            audio_embs.append(embedder.embed_audio(aud))

            if (i + 1) % 100 == 0:
                print(f"Embedded {i + 1}/{len(prompts)} samples")

        except Exception as e:
            print(f"Skipping sample {i}: {e}")
            continue

    return MultimodalTripletDataset(
        text_embeddings=np.array(text_embs),
        image_embeddings=np.array(image_embs),
        audio_embeddings=np.array(audio_embs),
    )
