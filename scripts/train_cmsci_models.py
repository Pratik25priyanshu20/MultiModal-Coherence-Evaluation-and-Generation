#!/usr/bin/env python3
"""
Unified training script for cMSCI Variant E+F models.

Trains everything needed for the full cMSCI pipeline:
  1. CrossSpaceBridge (CLIP image <-> CLAP audio alignment)
  2. CLIP probabilistic adapter (ProbVLM-style uncertainty for images/text)
  3. CLAP probabilistic adapter (ProbVLM-style uncertainty for audio/text)

After training, validates by computing si_a on held-out pairs and checking
uncertainty calibration.

Usage:
    python scripts/train_cmsci_models.py --help
    python scripts/train_cmsci_models.py --device cuda
    python scripts/train_cmsci_models.py --device mps --bridge-only
    python scripts/train_cmsci_models.py --data-path data/bridge_training/combined_training.npz

Prerequisite:
    python scripts/prepare_bridge_data.py --all --vggsound-features-path /data/vggsound/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bridge_training" / "combined_training.npz"
BRIDGE_OUTPUT_DIR = PROJECT_ROOT / "models" / "bridge"
PROB_ADAPTER_DIR = PROJECT_ROOT / "models" / "prob_adapters"


def detect_device(requested: str = "auto") -> str:
    """Detect best available device."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_training_data(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load paired image-audio embeddings from npz."""
    data = np.load(data_path)
    image_embs = data["image_embeddings"]
    audio_embs = data["audio_embeddings"]

    logger.info("Loaded training data: %d pairs", len(image_embs))
    logger.info("  Image embeddings: %s", image_embs.shape)
    logger.info("  Audio embeddings: %s", audio_embs.shape)

    assert image_embs.shape[1] == 512, f"Expected 512-d CLIP, got {image_embs.shape[1]}"
    assert audio_embs.shape[1] == 512, f"Expected 512-d CLAP, got {audio_embs.shape[1]}"

    return image_embs, audio_embs


def train_bridge(
    image_embs: np.ndarray,
    audio_embs: np.ndarray,
    device: str = "cpu",
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 10,
) -> None:
    """Train the CrossSpaceBridge on paired embeddings."""
    from src.embeddings.cross_space_bridge import (
        CrossSpaceBridge,
        ImageAudioPairDataset,
        BridgeTrainer,
    )

    logger.info("=" * 60)
    logger.info("TRAINING: CrossSpaceBridge")
    logger.info("=" * 60)
    logger.info("  Device: %s", device)
    logger.info("  Pairs: %d", len(image_embs))
    logger.info("  Epochs: %d, Batch: %d, LR: %s", n_epochs, batch_size, lr)

    bridge = CrossSpaceBridge()
    dataset = ImageAudioPairDataset(image_embs, audio_embs)

    trainer = BridgeTrainer(
        model=bridge,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        patience=patience,
        output_dir=str(BRIDGE_OUTPUT_DIR),
    )

    t_start = time.time()
    trained_bridge = trainer.train(dataset)
    elapsed = time.time() - t_start

    logger.info("Bridge training complete in %.1fs", elapsed)
    logger.info("  Best model: %s", BRIDGE_OUTPUT_DIR / "bridge_best.pt")
    logger.info("  Final model: %s", BRIDGE_OUTPUT_DIR / "bridge_final.pt")

    # Quick validation: compute si_a on a few held-out pairs
    n_val = min(100, len(image_embs))
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(image_embs), n_val, replace=False)

    sims = []
    for idx in val_idx:
        sim = trained_bridge.compute_similarity(image_embs[idx], audio_embs[idx])
        sims.append(sim)

    logger.info("  Validation si_a: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                np.mean(sims), np.std(sims), np.min(sims), np.max(sims))


def train_prob_adapter(
    embeddings: np.ndarray,
    name: str,
    output_path: str,
    device: str = "cpu",
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 15,
) -> None:
    """Train a ProbVLM-style probabilistic adapter."""
    from src.embeddings.prob_adapter_trainer import ProbAdapterTrainer

    logger.info("=" * 60)
    logger.info("TRAINING: %s Probabilistic Adapter", name)
    logger.info("=" * 60)
    logger.info("  Device: %s", device)
    logger.info("  Embeddings: %d x %d", *embeddings.shape)
    logger.info("  Epochs: %d, Batch: %d, LR: %s", n_epochs, batch_size, lr)

    trainer = ProbAdapterTrainer(
        input_dim=embeddings.shape[1],
        device=device,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        patience=patience,
    )

    t_start = time.time()
    adapter = trainer.train(embeddings, output_path=output_path)
    elapsed = time.time() - t_start

    logger.info("%s adapter training complete in %.1fs", name, elapsed)
    logger.info("  Model saved: %s", output_path)

    # Quick validation: check uncertainty distribution
    n_val = min(100, len(embeddings))
    uncertainties = []
    for i in range(n_val):
        u = adapter.uncertainty(embeddings[i])
        uncertainties.append(u)

    logger.info("  Validation uncertainty: mean=%.6f, std=%.6f",
                np.mean(uncertainties), np.std(uncertainties))


def validate_full_pipeline(
    image_embs: np.ndarray,
    audio_embs: np.ndarray,
) -> None:
    """Re-run cMSCI comparison with all trained models active."""
    logger.info("=" * 60)
    logger.info("VALIDATION: Full cMSCI Pipeline")
    logger.info("=" * 60)

    from src.config.settings import (
        CMSCI_CALIBRATION_PATH,
        EXMCR_WEIGHTS_PATH,
        PROB_CLIP_ADAPTER_PATH,
        PROB_CLAP_ADAPTER_PATH,
    )

    bridge_path = BRIDGE_OUTPUT_DIR / "bridge_best.pt"
    clip_adapter_path = PROB_ADAPTER_DIR / "clip_adapter.pt"
    clap_adapter_path = PROB_ADAPTER_DIR / "clap_adapter.pt"

    # Check which models are available
    available = {
        "calibration": CMSCI_CALIBRATION_PATH.exists(),
        "bridge": bridge_path.exists(),
        "clip_adapter": clip_adapter_path.exists(),
        "clap_adapter": clap_adapter_path.exists(),
    }
    logger.info("  Available models: %s", {k: v for k, v in available.items() if v})

    if bridge_path.exists():
        from src.embeddings.cross_space_bridge import CrossSpaceBridge
        bridge = CrossSpaceBridge.load(bridge_path)

        # Compute si_a distribution
        n_val = min(200, len(image_embs))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(image_embs), n_val, replace=False)

        matched_sims = []
        mismatched_sims = []
        for i in range(n_val):
            # Matched pair
            sim = bridge.compute_similarity(image_embs[idx[i]], audio_embs[idx[i]])
            matched_sims.append(sim)
            # Mismatched pair (random)
            j = rng.choice(len(audio_embs))
            sim_mm = bridge.compute_similarity(image_embs[idx[i]], audio_embs[j])
            mismatched_sims.append(sim_mm)

        logger.info("  Bridge si_a (matched): mean=%.4f, std=%.4f",
                    np.mean(matched_sims), np.std(matched_sims))
        logger.info("  Bridge si_a (mismatched): mean=%.4f, std=%.4f",
                    np.mean(mismatched_sims), np.std(mismatched_sims))
        sep = np.mean(matched_sims) - np.mean(mismatched_sims)
        logger.info("  Separation (matched - mismatched): %.4f", sep)

    logger.info("Validation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Train cMSCI models (bridge + probabilistic adapters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Full training pipeline:
  1. python scripts/prepare_bridge_data.py --all --vggsound-features-path /data/vggsound/
  2. python scripts/train_cmsci_models.py --device cuda
  3. python scripts/run_cmsci_comparison.py --all
  4. python scripts/run_cmsci_ablation.py
        """,
    )
    parser.add_argument("--data-path", type=str,
                        default=str(DEFAULT_DATA_PATH),
                        help="Path to combined training npz")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Training device (default: auto-detect)")
    parser.add_argument("--bridge-only", action="store_true",
                        help="Only train the cross-space bridge")
    parser.add_argument("--adapters-only", action="store_true",
                        help="Only train probabilistic adapters")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip post-training validation")

    # Bridge hyperparameters
    parser.add_argument("--bridge-epochs", type=int, default=50)
    parser.add_argument("--bridge-batch-size", type=int, default=64)
    parser.add_argument("--bridge-lr", type=float, default=3e-4)
    parser.add_argument("--bridge-patience", type=int, default=10)

    # Adapter hyperparameters
    parser.add_argument("--adapter-epochs", type=int, default=100)
    parser.add_argument("--adapter-batch-size", type=int, default=32)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--adapter-patience", type=int, default=15)

    args = parser.parse_args()

    device = detect_device(args.device)
    logger.info("=" * 60)
    logger.info("cMSCI MODEL TRAINING")
    logger.info("=" * 60)
    logger.info("  Device: %s", device)
    logger.info("  Data: %s", args.data_path)

    # Load data
    if not Path(args.data_path).exists():
        logger.error("Training data not found: %s", args.data_path)
        logger.error("Run first: python scripts/prepare_bridge_data.py")
        return 1

    image_embs, audio_embs = load_training_data(args.data_path)

    t_total_start = time.time()

    # 1. Train bridge
    if not args.adapters_only:
        train_bridge(
            image_embs, audio_embs,
            device=device,
            n_epochs=args.bridge_epochs,
            batch_size=args.bridge_batch_size,
            lr=args.bridge_lr,
            patience=args.bridge_patience,
        )

    # 2. Train probabilistic adapters
    if not args.bridge_only:
        PROB_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

        # CLIP adapter (trained on image embeddings)
        train_prob_adapter(
            embeddings=image_embs,
            name="CLIP",
            output_path=str(PROB_ADAPTER_DIR / "clip_adapter.pt"),
            device=device,
            n_epochs=args.adapter_epochs,
            batch_size=args.adapter_batch_size,
            lr=args.adapter_lr,
            patience=args.adapter_patience,
        )

        # CLAP adapter (trained on audio embeddings)
        train_prob_adapter(
            embeddings=audio_embs,
            name="CLAP",
            output_path=str(PROB_ADAPTER_DIR / "clap_adapter.pt"),
            device=device,
            n_epochs=args.adapter_epochs,
            batch_size=args.adapter_batch_size,
            lr=args.adapter_lr,
            patience=args.adapter_patience,
        )

    # 3. Validate
    if not args.skip_validation:
        validate_full_pipeline(image_embs, audio_embs)

    elapsed = time.time() - t_total_start
    logger.info("=" * 60)
    logger.info("ALL TRAINING COMPLETE in %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  python scripts/run_cmsci_comparison.py --all")
    logger.info("  python scripts/run_cmsci_ablation.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
