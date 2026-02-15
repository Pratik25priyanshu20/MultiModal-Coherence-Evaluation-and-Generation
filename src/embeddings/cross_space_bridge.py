"""
Cross-Space Bridge: CLIP Image ↔ CLAP Audio Alignment

Problem:
    CLIP image embeddings and CLAP audio embeddings live in DIFFERENT 512-d spaces.
    Cosine similarity between them is meaningless. This is why si_a = None in
    the coherence engine.

Solution:
    Train two lightweight projection heads that map:
        CLIP image (512-d)  →  shared bridge space (256-d)
        CLAP audio (512-d)  →  shared bridge space (256-d)

    After training, cosine similarity in the bridge space gives a meaningful
    image-audio coherence score (si_a).

Architecture:
    ┌─────────────┐         ┌──────────────┐
    │ CLIP Image   │──▶ image_proj ──▶│              │
    │ Embedding    │   (512→256)     │  Shared      │──▶ cosine_sim(i, a)
    │ (512-d)      │                 │  Bridge      │
    └─────────────┘                 │  Space       │
                                     │  (256-d)     │
    ┌─────────────┐                 │              │
    │ CLAP Audio   │──▶ audio_proj ──▶│              │
    │ Embedding    │   (512→256)     └──────────────┘
    │ (512-d)      │
    └─────────────┘

Training:
    Uses paired (image, audio) data where both depict the same scene.
    InfoNCE contrastive loss pulls matched pairs together, pushes
    mismatched pairs apart. Text is NOT involved — the bridge operates
    purely between image and audio spaces.

    Critically, existing CLIP text-image and CLAP text-audio paths are
    UNCHANGED. The bridge is additive — it enables si_a without
    degrading st_i or st_a.

Integration:
    Once trained, load the bridge into CoherenceEngine via:
        engine.load_bridge("models/bridge/bridge_final.pt")
    This enables si_a computation and activates the full MSCI formula:
        MSCI = 0.45 * st_i + 0.45 * st_a + 0.10 * si_a

Status: ARCHITECTURE ONLY — not trained. Requires paired image-audio data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the cross-space bridge. "
            "Install with: pip install torch"
        )


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════


class BridgeProjectionHead(nn.Module):
    """
    Single projection head for one modality.

    Architecture: Linear(in, hidden) → GELU → Dropout → Linear(hidden, out) → L2 norm

    Uses GELU instead of ReLU following modern transformer conventions.
    L2 normalization ensures cosine similarity operates on the unit hypersphere.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 384,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        _check_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class CrossSpaceBridge(nn.Module):
    """
    Learned bridge between CLIP image space and CLAP audio space.

    Maps both to a shared 256-d space where cosine similarity is meaningful.
    Does NOT touch text embeddings — CLIP text-image and CLAP text-audio
    paths remain identity (pre-trained alignment preserved).
    """

    def __init__(
        self,
        clip_image_dim: int = 512,
        clap_audio_dim: int = 512,
        bridge_dim: int = 256,
        hidden_dim: int = 384,
        dropout: float = 0.1,
    ):
        _check_torch()
        super().__init__()

        self.image_proj = BridgeProjectionHead(
            input_dim=clip_image_dim,
            hidden_dim=hidden_dim,
            output_dim=bridge_dim,
            dropout=dropout,
        )
        self.audio_proj = BridgeProjectionHead(
            input_dim=clap_audio_dim,
            hidden_dim=hidden_dim,
            output_dim=bridge_dim,
            dropout=dropout,
        )

        self.bridge_dim = bridge_dim
        self.config = {
            "clip_image_dim": clip_image_dim,
            "clap_audio_dim": clap_audio_dim,
            "bridge_dim": bridge_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

    def forward(
        self,
        image_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Project image and/or audio embeddings into bridge space.

        Args:
            image_emb: CLIP image embeddings [batch, 512]
            audio_emb: CLAP audio embeddings [batch, 512]

        Returns:
            Dict with 'image' and/or 'audio' keys, values are [batch, bridge_dim]
        """
        result = {}
        if image_emb is not None:
            result["image"] = self.image_proj(image_emb)
        if audio_emb is not None:
            result["audio"] = self.audio_proj(audio_emb)
        return result

    def compute_similarity(
        self,
        image_emb: np.ndarray,
        audio_emb: np.ndarray,
    ) -> float:
        """
        Compute image-audio similarity through the bridge.

        This is the main inference method. Takes raw CLIP image and CLAP audio
        embeddings (numpy), projects both into bridge space, returns cosine sim.

        Args:
            image_emb: CLIP image embedding, shape (512,)
            audio_emb: CLAP audio embedding, shape (512,)

        Returns:
            Cosine similarity in bridge space (float, range [-1, 1])
        """
        _check_torch()
        self.eval()
        with torch.no_grad():
            img_t = torch.tensor(image_emb, dtype=torch.float32).unsqueeze(0)
            aud_t = torch.tensor(audio_emb, dtype=torch.float32).unsqueeze(0)
            projected = self.forward(image_emb=img_t, audio_emb=aud_t)
            sim = F.cosine_similarity(projected["image"], projected["audio"])
            return float(sim.item())

    def save(self, path: Path):
        """Save bridge weights + config."""
        _check_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        config_path = path.with_suffix(".json")
        with config_path.open("w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("Saved bridge to %s", path)

    @classmethod
    def load(cls, path: Path) -> "CrossSpaceBridge":
        """Load bridge from saved weights."""
        _check_torch()
        path = Path(path)
        config_path = path.with_suffix(".json")
        with config_path.open("r") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Loaded bridge from %s", path)
        return model


# ═══════════════════════════════════════════════════════════════
# TRAINING COMPONENTS
# ═══════════════════════════════════════════════════════════════


class ImageAudioPairDataset(Dataset):
    """
    Dataset of paired (CLIP image, CLAP audio) embeddings.

    Each pair represents the same scene — e.g., a beach photo paired
    with ocean wave audio. Text is not needed for bridge training.

    Data sources for future training:
        - RQ1/RQ2 baseline runs: each has a matched (image, audio) pair
        - Manual curation: pair images from data/wikimedia/images with
          audio from data/freesound/audio by domain
        - External datasets: AudioCaps + MSCOCO overlap, VGGSound, etc.
    """

    def __init__(
        self,
        image_embeddings: np.ndarray,
        audio_embeddings: np.ndarray,
    ):
        """
        Args:
            image_embeddings: CLIP image embeddings [N, 512]
            audio_embeddings: CLAP audio embeddings [N, 512]
        """
        _check_torch()
        assert len(image_embeddings) == len(audio_embeddings), (
            f"Mismatched pair count: {len(image_embeddings)} images, "
            f"{len(audio_embeddings)} audio"
        )
        self.images = torch.tensor(image_embeddings, dtype=torch.float32)
        self.audio = torch.tensor(audio_embeddings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"image": self.images[idx], "audio": self.audio[idx]}


class BridgeInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for image-audio bridge training.

    For a batch of N paired (image, audio) embeddings:
        - Each image should be most similar to its paired audio (and vice versa)
        - All other items in the batch are treated as negatives

    Loss = 0.5 * (image→audio NCE + audio→image NCE)

    This is the same loss structure used by CLIP and CLAP themselves,
    applied here to bridge their output spaces.
    """

    def __init__(self, temperature: float = 0.07):
        _check_torch()
        super().__init__()
        # Learnable temperature (following CLIP)
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(1.0 / temperature))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_temperature)

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute symmetric InfoNCE loss.

        Args:
            image_emb: Projected image embeddings [batch, bridge_dim], L2-normalized
            audio_emb: Projected audio embeddings [batch, bridge_dim], L2-normalized

        Returns:
            (loss, metrics_dict)
        """
        batch_size = image_emb.size(0)

        # Similarity matrix [batch, batch]
        logits = torch.mm(image_emb, audio_emb.t()) / self.temperature

        # Labels: diagonal (each image matches its own audio)
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric cross-entropy
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i2a + loss_a2i)

        # Metrics
        with torch.no_grad():
            acc_i2a = (logits.argmax(dim=1) == labels).float().mean()
            acc_a2i = (logits.t().argmax(dim=1) == labels).float().mean()

        metrics = {
            "loss": loss.item(),
            "loss_i2a": loss_i2a.item(),
            "loss_a2i": loss_a2i.item(),
            "acc_i2a": acc_i2a.item(),
            "acc_a2i": acc_a2i.item(),
            "temperature": self.temperature.item(),
        }

        return loss, metrics


class BridgeTrainer:
    """
    Training loop for the cross-space bridge.

    Minimal, focused trainer:
        - AdamW optimizer with cosine LR schedule
        - Symmetric InfoNCE loss with learnable temperature
        - Early stopping on validation loss
        - Checkpoint saving

    Usage (future, when paired data is available):
        bridge = CrossSpaceBridge()
        trainer = BridgeTrainer(bridge)
        dataset = ImageAudioPairDataset(image_embs, audio_embs)
        trainer.train(dataset)
    """

    def __init__(
        self,
        model: CrossSpaceBridge,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 50,
        patience: int = 10,
        output_dir: str = "models/bridge",
    ):
        _check_torch()
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = BridgeInfoNCELoss().to(self.device)

        # Optimize both projection heads + temperature
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.loss_fn.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.history = []

    def train(
        self,
        train_data: ImageAudioPairDataset,
        val_data: Optional[ImageAudioPairDataset] = None,
        val_split: float = 0.15,
    ) -> CrossSpaceBridge:
        """
        Train the bridge.

        Args:
            train_data: Paired image-audio embeddings
            val_data: Optional separate validation set. If None, splits from train_data.
            val_split: Fraction to hold out for validation if val_data is None.

        Returns:
            Trained CrossSpaceBridge model
        """
        # Split if no val_data provided
        if val_data is None and val_split > 0:
            n_val = max(1, int(len(train_data) * val_split))
            n_train = len(train_data) - n_val
            train_data, val_data = torch.utils.data.random_split(
                train_data, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
        ) if val_data is not None else None

        # Cosine LR schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.n_epochs,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            "Training bridge: %d train, %d val, %d epochs, batch=%d, device=%s",
            len(train_data), len(val_data) if val_data else 0,
            self.n_epochs, self.batch_size, self.device,
        )

        for epoch in range(self.n_epochs):
            # ── Train ────────────────────────────────
            self.model.train()
            self.loss_fn.train()
            epoch_metrics = []

            for batch in train_loader:
                img = batch["image"].to(self.device)
                aud = batch["audio"].to(self.device)

                self.optimizer.zero_grad()
                projected = self.model(image_emb=img, audio_emb=aud)
                loss, metrics = self.loss_fn(projected["image"], projected["audio"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_metrics.append(metrics)

            scheduler.step()

            # Epoch averages
            avg = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}

            # ── Validate ─────────────────────────────
            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)
                avg["val_loss"] = val_loss

            avg["epoch"] = epoch + 1
            avg["lr"] = scheduler.get_last_lr()[0]
            self.history.append(avg)

            logger.info(
                "Epoch %d/%d: loss=%.4f acc_i2a=%.3f acc_a2i=%.3f temp=%.3f%s",
                epoch + 1, self.n_epochs, avg["loss"],
                avg["acc_i2a"], avg["acc_a2i"], avg["temperature"],
                f" val_loss={val_loss:.4f}" if val_loss is not None else "",
            )

            # ── Early stopping ───────────────────────
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.model.save(self.output_dir / "bridge_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        # Save final
        self.model.save(self.output_dir / "bridge_final.pt")
        self._save_history()

        # Load best if we had validation
        if val_loader and (self.output_dir / "bridge_best.pt").exists():
            self.model = CrossSpaceBridge.load(self.output_dir / "bridge_best.pt")
            logger.info("Loaded best checkpoint (val_loss=%.4f)", best_val_loss)

        return self.model

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        self.loss_fn.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(self.device)
                aud = batch["audio"].to(self.device)
                projected = self.model(image_emb=img, audio_emb=aud)
                loss, _ = self.loss_fn(projected["image"], projected["audio"])
                losses.append(loss.item())
        return float(np.mean(losses))

    def _save_history(self):
        path = self.output_dir / "bridge_training_history.json"
        with path.open("w") as f:
            json.dump(self.history, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# COHERENCE ENGINE INTEGRATION (future use)
# ═══════════════════════════════════════════════════════════════


def build_paired_dataset_from_runs(
    results_json: str,
    embedder=None,
) -> ImageAudioPairDataset:
    """
    Build paired training data from existing experiment runs.

    Each baseline run in RQ1/RQ2 has a matched (image_path, audio_path) pair
    for the same prompt. These pairs can be used to train the bridge.

    Args:
        results_json: Path to rq1_results.json or rq2_results.json
        embedder: AlignedEmbedder instance (for re-embedding if needed)

    Returns:
        ImageAudioPairDataset ready for training

    Example:
        embedder = AlignedEmbedder()
        dataset = build_paired_dataset_from_runs(
            "runs/rq1/rq1_results.json",
            embedder=embedder,
        )
        bridge = CrossSpaceBridge()
        trainer = BridgeTrainer(bridge)
        trainer.train(dataset)
    """
    _check_torch()
    import json as _json

    with open(results_json) as f:
        data = _json.load(f)

    # Collect unique (image_path, audio_path) pairs from baseline runs
    pairs = {}
    for r in data["results"]:
        # Only use baseline (matched) pairs
        condition = r.get("condition", r.get("mode", ""))
        if condition not in ("baseline", "direct"):
            continue

        img = r.get("image_path")
        aud = r.get("audio_path")
        if img and aud:
            key = f"{img}||{aud}"
            if key not in pairs:
                pairs[key] = (img, aud)

    logger.info("Found %d unique image-audio pairs from %s", len(pairs), results_json)

    if embedder is None:
        from src.embeddings.aligned_embeddings import AlignedEmbedder
        embedder = AlignedEmbedder()

    image_embs = []
    audio_embs = []
    for img_path, aud_path in pairs.values():
        try:
            img_emb = embedder.embed_image(img_path)
            aud_emb = embedder.embed_audio(aud_path)
            image_embs.append(img_emb)
            audio_embs.append(aud_emb)
        except Exception as e:
            logger.warning("Skipping pair %s / %s: %s", img_path, aud_path, e)

    logger.info("Successfully embedded %d pairs", len(image_embs))

    return ImageAudioPairDataset(
        image_embeddings=np.array(image_embs),
        audio_embeddings=np.array(audio_embs),
    )
