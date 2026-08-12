#!/usr/bin/env python3
"""
Train Ex-MCR Projector: CLAP Audio -> CLIP Space.

Projects CLAP audio embeddings INTO CLIP space using InfoNCE contrastive loss
on paired (CLIP image, CLAP audio) embeddings. After training, audio embeddings
can be directly compared to CLIP text/image embeddings via cosine similarity,
enabling true tri-modal Gramian volume computation.

Architecture: MLP (512 -> 512 -> 512) with ReLU, L2-normalized output.
Uses the existing ExMCRProjector class from src/embeddings/space_alignment.py.

Data: combined_training.npz — 2,193 paired (CLIP image 512d, CLAP audio 512d)
Loss: Symmetric InfoNCE with learnable temperature
Training: AdamW lr=3e-4, wd=1e-4, cosine LR, batch=64, epochs=50, patience=10

Usage:
    python scripts/train_exmcr.py
    python scripts/train_exmcr.py --data data/bridge_training/combined_training.npz
    python scripts/train_exmcr.py --epochs 100 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)


# ─── Dataset ────────────────────────────────────────────────────

class PairedEmbeddingDataset(Dataset):
    """Dataset of paired (CLIP image, CLAP audio) embeddings."""

    def __init__(self, clip_embs: np.ndarray, clap_embs: np.ndarray):
        assert len(clip_embs) == len(clap_embs)
        self.clip = torch.tensor(clip_embs, dtype=torch.float32)
        self.clap = torch.tensor(clap_embs, dtype=torch.float32)

    def __len__(self):
        return len(self.clip)

    def __getitem__(self, idx):
        return {"clip": self.clip[idx], "clap": self.clap[idx]}


# ─── Model ──────────────────────────────────────────────────────

class ExMCRNet(nn.Module):
    """MLP projector: CLAP 512-d -> CLIP 512-d with L2 normalization."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), p=2, dim=-1)


# ─── Loss ───────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss with learnable temperature."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1.0 / temperature), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_temperature)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor):
        """
        Args:
            anchor: CLIP image embeddings [B, D], L2-normalized
            positive: Projected CLAP audio embeddings [B, D], L2-normalized
        """
        logits = torch.mm(anchor, positive.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_a2p = F.cross_entropy(logits, labels)
        loss_p2a = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_a2p + loss_p2a)

        with torch.no_grad():
            acc_a2p = (logits.argmax(dim=1) == labels).float().mean()
            acc_p2a = (logits.t().argmax(dim=1) == labels).float().mean()

        return loss, {
            "loss": loss.item(),
            "acc_a2p": acc_a2p.item(),
            "acc_p2a": acc_p2a.item(),
            "temperature": self.temperature.item(),
        }


# ─── Training ───────────────────────────────────────────────────

def train_exmcr(
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    val_split: float = 0.15,
    seed: int = 42,
):
    """Train Ex-MCR projector and save weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    # ── Load data ───────────────────────────────────────────
    logger.info("Loading data from %s", data_path)
    data = np.load(data_path)

    # Support both naming conventions
    if "clip_embeddings" in data:
        clip_embs = data["clip_embeddings"]
        clap_embs = data["clap_embeddings"]
    elif "image_embeddings" in data:
        clip_embs = data["image_embeddings"]
        clap_embs = data["audio_embeddings"]
    else:
        raise ValueError(f"Unrecognized data format. Keys: {list(data.keys())}")

    logger.info("Loaded %d pairs: CLIP %s, CLAP %s", len(clip_embs), clip_embs.shape, clap_embs.shape)

    # L2-normalize CLIP targets (anchor space)
    clip_norms = np.linalg.norm(clip_embs, axis=1, keepdims=True) + 1e-12
    clip_embs = clip_embs / clip_norms

    # ── Split ───────────────────────────────────────────────
    dataset = PairedEmbeddingDataset(clip_embs, clap_embs)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info("Train: %d, Val: %d, Batch: %d", n_train, n_val, batch_size)

    # ── Model ───────────────────────────────────────────────
    model = ExMCRNet(in_dim=512, hidden_dim=512, out_dim=512).to(device)
    loss_fn = InfoNCELoss().to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("ExMCR model: %d parameters", param_count)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Output dir ──────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Training loop ───────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        loss_fn.train()
        epoch_metrics = []

        for batch in train_loader:
            clip_b = batch["clip"].to(device)
            clap_b = batch["clap"].to(device)

            optimizer.zero_grad()
            projected = model(clap_b)  # CLAP -> CLIP space
            # Anchor = L2-normalized CLIP image embeddings
            anchor = F.normalize(clip_b, p=2, dim=-1)
            loss, metrics = loss_fn(anchor, projected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_metrics.append(metrics)

        scheduler.step()

        avg_train = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}

        # Validate
        model.eval()
        loss_fn.eval()
        val_losses = []
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                clip_b = batch["clip"].to(device)
                clap_b = batch["clap"].to(device)
                projected = model(clap_b)
                anchor = F.normalize(clip_b, p=2, dim=-1)
                loss, metrics = loss_fn(anchor, projected)
                val_losses.append(loss.item())
                val_metrics.append(metrics)

        val_loss = float(np.mean(val_losses))
        avg_val = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}

        record = {
            "epoch": epoch,
            "train_loss": avg_train["loss"],
            "val_loss": val_loss,
            "train_acc": (avg_train["acc_a2p"] + avg_train["acc_p2a"]) / 2,
            "val_acc": (avg_val["acc_a2p"] + avg_val["acc_p2a"]) / 2,
            "temperature": avg_train["temperature"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f train_acc=%.3f val_acc=%.3f temp=%.3f",
            epoch, epochs, record["train_loss"], record["val_loss"],
            record["train_acc"], record["val_acc"], record["temperature"],
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best — use Sequential format compatible with ExMCRProjector
            save_path = out_path / "ex_clap.pt"
            torch.save(model.layers.state_dict(), save_path)
            logger.info("  Saved best model (val_loss=%.4f) -> %s", val_loss, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save final
    torch.save(model.layers.state_dict(), out_path / "ex_clap_final.pt")

    # Save history
    with open(out_path / "exmcr_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)

    # ── Validation: matched > mismatched similarity ─────────
    logger.info("\n--- Validation: Matched vs Mismatched Similarity ---")
    model.eval()
    # Reload best weights
    best_state = torch.load(out_path / "ex_clap.pt", map_location=device, weights_only=True)
    model.layers.load_state_dict(best_state)

    with torch.no_grad():
        # Use validation set
        all_clip = []
        all_projected = []
        for batch in val_loader:
            clip_b = batch["clip"].to(device)
            clap_b = batch["clap"].to(device)
            projected = model(clap_b)
            anchor = F.normalize(clip_b, p=2, dim=-1)
            all_clip.append(anchor)
            all_projected.append(projected)

        all_clip = torch.cat(all_clip, dim=0)
        all_projected = torch.cat(all_projected, dim=0)

        # Matched similarities (diagonal)
        matched_sims = (all_clip * all_projected).sum(dim=1)
        mean_matched = matched_sims.mean().item()

        # Mismatched similarities (off-diagonal)
        sim_matrix = torch.mm(all_clip, all_projected.t())
        n = sim_matrix.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        mismatched_sims = sim_matrix[mask]
        mean_mismatched = mismatched_sims.mean().item()

    logger.info("  Matched similarity (mean):    %.4f", mean_matched)
    logger.info("  Mismatched similarity (mean):  %.4f", mean_mismatched)
    logger.info("  Gap (matched - mismatched):    %.4f", mean_matched - mean_mismatched)

    if mean_matched > mean_mismatched:
        logger.info("  PASS: Matched > Mismatched")
    else:
        logger.warning("  FAIL: Matched <= Mismatched — projector may not be useful")

    return out_path / "ex_clap.pt"


def main():
    parser = argparse.ArgumentParser(description="Train Ex-MCR Projector (CLAP -> CLIP)")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "data" / "bridge_training" / "combined_training.npz"),
                        help="Path to paired training data (.npz)")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "models" / "exmcr"),
                        help="Output directory for model weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Ex-MCR Training: CLAP Audio -> CLIP Space")
    print("=" * 70)

    train_exmcr(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
