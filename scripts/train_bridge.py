#!/usr/bin/env python3
"""
Train the Cross-Space Bridge (CLIP Image ↔ CLAP Audio).

Builds paired training data from domain-matched embedding indexes,
optionally augmented with exact pairs from experiment runs.

Usage:
    # Basic training (domain-matched weak supervision only)
    python scripts/train_bridge.py

    # With exact pairs from generative experiment
    python scripts/train_bridge.py --add-runs runs/rq1_gen/rq1_gen_results.json

    # Custom params
    python scripts/train_bridge.py --epochs 100 --batch-size 64 --lr 1e-4

Status: READY TO RUN when needed. Not required for current experiments.
        Bridge training is future work — architecture is complete.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Cross-Space Bridge")
    parser.add_argument("--image-index", default="data/embeddings/image_index.npz")
    parser.add_argument("--audio-index", default="data/embeddings/audio_index.npz")
    parser.add_argument("--add-runs", nargs="*", default=[],
                        help="Experiment result JSONs for exact pair augmentation")
    parser.add_argument("--output-dir", default="models/bridge")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    import numpy as np
    import torch

    from src.embeddings.cross_space_bridge import (
        CrossSpaceBridge,
        BridgeTrainer,
        ImageAudioPairDataset,
        build_domain_matched_dataset,
        build_paired_dataset_from_runs,
    )

    # ── Build domain-matched dataset ──────────────────────
    print("=" * 70)
    print("CROSS-SPACE BRIDGE TRAINING")
    print("=" * 70)
    print(f"\n1. Building domain-matched dataset...")

    dataset = build_domain_matched_dataset(
        image_index_path=args.image_index,
        audio_index_path=args.audio_index,
    )
    print(f"   Domain-matched pairs: {len(dataset)}")

    # ── Optionally add exact pairs from runs ──────────────
    if args.add_runs:
        print(f"\n2. Adding exact pairs from {len(args.add_runs)} run file(s)...")
        from src.embeddings.aligned_embeddings import AlignedEmbedder
        embedder = AlignedEmbedder()

        all_image_embs = [dataset.images.numpy()]
        all_audio_embs = [dataset.audio.numpy()]

        for run_path in args.add_runs:
            run_dataset = build_paired_dataset_from_runs(run_path, embedder=embedder)
            print(f"   {run_path}: {len(run_dataset)} pairs")
            all_image_embs.append(run_dataset.images.numpy())
            all_audio_embs.append(run_dataset.audio.numpy())

        combined_images = np.concatenate(all_image_embs, axis=0)
        combined_audio = np.concatenate(all_audio_embs, axis=0)
        dataset = ImageAudioPairDataset(combined_images, combined_audio)
        print(f"   Total combined pairs: {len(dataset)}")
    else:
        print(f"\n2. No run files specified (--add-runs). Using domain-matched only.")

    # ── Train ──────────────────────────────────────────────
    print(f"\n3. Training bridge...")
    print(f"   Epochs:     {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   LR:         {args.lr}")
    print(f"   Patience:   {args.patience}")
    print(f"   Val split:  {args.val_split}")
    print(f"   Output:     {args.output_dir}")
    print()

    bridge = CrossSpaceBridge()
    trainer = BridgeTrainer(
        model=bridge,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir,
    )

    trained_bridge = trainer.train(dataset, val_split=args.val_split)

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    output_dir = Path(args.output_dir)
    print(f"  Best model:   {output_dir / 'bridge_best.pt'}")
    print(f"  Final model:  {output_dir / 'bridge_final.pt'}")
    print(f"  History:      {output_dir / 'bridge_training_history.json'}")

    # Quick validation: compute si_a for a random pair
    if len(dataset) > 0:
        idx = 0
        img_emb = dataset.images[idx].numpy()
        aud_emb = dataset.audio[idx].numpy()
        si_a = trained_bridge.compute_similarity(img_emb, aud_emb)
        print(f"\n  Sample si_a (pair 0): {si_a:.4f}")

    print(f"\nTo use: engine.load_bridge('{output_dir / 'bridge_best.pt'}')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
