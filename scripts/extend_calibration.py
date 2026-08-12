#!/usr/bin/env python3
"""
Extend cMSCI Calibration with ExMCR and Uncertainty Channels.

Adds new reference distribution channels to artifacts/cmsci_calibration.json:
  - gram_coh_ia_exmcr: gram coherence of (image_clip, ExMCR(audio_clap))
  - gram_coh_tia: 3-way gram coherence of (text_clip, image_clip, ExMCR(audio_clap))
  - uncertainty_ti: ProbVLM uncertainty for text-image (CLIP adapter)
  - uncertainty_ta: ProbVLM uncertainty for text-audio (CLAP adapter)
  - uncertainty_mean: average of uncertainty_ti and uncertainty_ta

Computed from the 90 RQ1 baseline samples using their stored embeddings.

Usage:
    python scripts/extend_calibration.py
    python scripts/extend_calibration.py --exmcr-only    # skip uncertainty
    python scripts/extend_calibration.py --dry-run        # print stats but don't save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RQ1_RESULTS_PATH = PROJECT_ROOT / "runs" / "rq1" / "rq1_results.json"
CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "cmsci_calibration.json"
EXMCR_WEIGHTS_PATH = PROJECT_ROOT / "models" / "exmcr" / "ex_clap.pt"
BRIDGE_WEIGHTS_PATH = PROJECT_ROOT / "models" / "bridge" / "bridge_best.pt"
PROB_CLIP_PATH = PROJECT_ROOT / "models" / "prob_adapters" / "clip_adapter.pt"
PROB_CLAP_PATH = PROJECT_ROOT / "models" / "prob_adapters" / "clap_adapter.pt"


def load_rq1_baseline_data():
    """Load RQ1 baseline results and re-embed for ExMCR/uncertainty channels."""
    if not RQ1_RESULTS_PATH.exists():
        print(f"ERROR: RQ1 results not found at {RQ1_RESULTS_PATH}")
        sys.exit(1)

    with open(RQ1_RESULTS_PATH) as f:
        data = json.load(f)

    baselines = [r for r in data["results"] if r.get("condition") == "baseline"]
    logger.info("Loaded %d baseline samples from RQ1", len(baselines))
    return baselines


def compute_exmcr_channels(baselines: list, exmcr_path: str | None, bridge_path: str | None):
    """Compute gram_coh_ia and gram_coh_tia using ExMCR or bridge."""
    from src.embeddings.aligned_embeddings import AlignedEmbedder
    from src.coherence.gram_volume import gram_volume_2d, gram_volume_3d, normalized_gram_coherence

    embedder = AlignedEmbedder(target_dim=512)

    # Try ExMCR first, fall back to bridge
    projector = None
    use_bridge = False

    if exmcr_path and Path(exmcr_path).exists():
        from src.embeddings.space_alignment import ExMCRProjector
        projector = ExMCRProjector(weights_path=exmcr_path)
        if projector.is_identity:
            logger.warning("ExMCR in identity mode — weights may not be trained")
            projector = None

    if projector is None and bridge_path and Path(bridge_path).exists():
        logger.info("No ExMCR weights, falling back to CrossSpaceBridge for ia channel")
        use_bridge = True

    gram_coh_ia_scores = []
    gram_coh_tia_scores = []

    for i, r in enumerate(baselines):
        img_path = r.get("image_path")
        aud_path = r.get("audio_path")
        prompt = r.get("prompt_text", r.get("text", ""))

        if not img_path or not aud_path:
            continue

        try:
            emb_image = embedder.embed_image(img_path)
            emb_audio = embedder.embed_audio(aud_path)
            emb_text_clip = embedder.embed_text(prompt) if prompt else None

            if projector is not None:
                # ExMCR: project audio into CLIP space
                audio_in_clip = projector.project_audio(emb_audio)
                gram_ia = gram_volume_2d(emb_image, audio_in_clip)
                gram_coh_ia = normalized_gram_coherence(gram_ia)
                gram_coh_ia_scores.append(gram_coh_ia)

                if emb_text_clip is not None:
                    gram_tia = gram_volume_3d(emb_text_clip, emb_image, audio_in_clip)
                    gram_coh_tia = normalized_gram_coherence(gram_tia, n_vectors=3)
                    gram_coh_tia_scores.append(gram_coh_tia)

            elif use_bridge:
                # Bridge: project both to shared 256-d space
                import torch
                from src.embeddings.cross_space_bridge import CrossSpaceBridge
                bridge = CrossSpaceBridge.load(bridge_path)
                bridge.eval()
                with torch.no_grad():
                    img_t = torch.tensor(emb_image, dtype=torch.float32).unsqueeze(0)
                    aud_t = torch.tensor(emb_audio, dtype=torch.float32).unsqueeze(0)
                    projected = bridge(image_emb=img_t, audio_emb=aud_t)
                    bridge_img = projected["image"].squeeze(0).numpy()
                    bridge_aud = projected["audio"].squeeze(0).numpy()
                gram_ia = gram_volume_2d(bridge_img, bridge_aud)
                gram_coh_ia = normalized_gram_coherence(gram_ia)
                gram_coh_ia_scores.append(gram_coh_ia)

        except Exception as e:
            logger.warning("  [%d] Error processing %s: %s", i, img_path, e)
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(baselines)}] ExMCR channels computed", end="\r")

    print()
    return gram_coh_ia_scores, gram_coh_tia_scores


def compute_uncertainty_channels(baselines: list, clip_adapter_path: str, clap_adapter_path: str):
    """Compute uncertainty channels using ProbVLM adapters."""
    from src.embeddings.aligned_embeddings import AlignedEmbedder

    embedder = AlignedEmbedder(target_dim=512)

    prob_clip = None
    prob_clap = None

    if clip_adapter_path and Path(clip_adapter_path).exists():
        from src.embeddings.probabilistic_adapter import ProbabilisticAdapter
        prob_clip = ProbabilisticAdapter.load(clip_adapter_path)
        logger.info("Loaded CLIP probabilistic adapter")

    if clap_adapter_path and Path(clap_adapter_path).exists():
        from src.embeddings.probabilistic_adapter import ProbabilisticAdapter
        prob_clap = ProbabilisticAdapter.load(clap_adapter_path)
        logger.info("Loaded CLAP probabilistic adapter")

    if prob_clip is None and prob_clap is None:
        logger.warning("No probabilistic adapters found — skipping uncertainty channels")
        return [], []

    uncertainty_ti_scores = []
    uncertainty_ta_scores = []

    for i, r in enumerate(baselines):
        img_path = r.get("image_path")
        aud_path = r.get("audio_path")
        prompt = r.get("prompt_text", r.get("text", ""))

        try:
            if prob_clip is not None and img_path and prompt:
                emb_text_clip = embedder.embed_text(prompt)
                emb_image = embedder.embed_image(img_path)
                u_text = prob_clip.uncertainty(emb_text_clip)
                u_image = prob_clip.uncertainty(emb_image)
                uncertainty_ti_scores.append((u_text + u_image) / 2.0)

            if prob_clap is not None and aud_path and prompt:
                emb_text_clap = embedder.embed_text_for_audio(prompt)
                emb_audio = embedder.embed_audio(aud_path)
                u_text = prob_clap.uncertainty(emb_text_clap)
                u_audio = prob_clap.uncertainty(emb_audio)
                uncertainty_ta_scores.append((u_text + u_audio) / 2.0)

        except Exception as e:
            logger.warning("  [%d] Uncertainty error: %s", i, e)
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(baselines)}] Uncertainty computed", end="\r")

    print()
    return uncertainty_ti_scores, uncertainty_ta_scores


def main():
    parser = argparse.ArgumentParser(description="Extend cMSCI Calibration")
    parser.add_argument("--exmcr-only", action="store_true", help="Skip uncertainty channels")
    parser.add_argument("--uncertainty-only", action="store_true", help="Skip ExMCR channels")
    parser.add_argument("--dry-run", action="store_true", help="Print stats but don't save")
    args = parser.parse_args()

    print("=" * 70)
    print("Extending cMSCI Calibration Reference")
    print("=" * 70)

    # Load existing calibration
    from src.coherence.calibration import (
        CalibrationStore,
        extend_calibration_with_exmcr,
        extend_calibration_with_uncertainty,
    )

    if CALIBRATION_PATH.exists():
        store = CalibrationStore.load(str(CALIBRATION_PATH))
        print(f"\nExisting channels: {list(store.distributions.keys())}")
    else:
        print(f"\nERROR: Calibration file not found: {CALIBRATION_PATH}")
        sys.exit(1)

    # Load RQ1 baseline data
    baselines = load_rq1_baseline_data()

    # ExMCR channels
    if not args.uncertainty_only:
        print("\n--- Computing ExMCR Channels ---")
        exmcr_path = str(EXMCR_WEIGHTS_PATH) if EXMCR_WEIGHTS_PATH.exists() else None
        bridge_path = str(BRIDGE_WEIGHTS_PATH) if BRIDGE_WEIGHTS_PATH.exists() else None

        if exmcr_path:
            print(f"  Using ExMCR: {exmcr_path}")
        elif bridge_path:
            print(f"  Using Bridge fallback: {bridge_path}")
        else:
            print("  WARNING: No ExMCR or Bridge weights found")

        gram_coh_ia, gram_coh_tia = compute_exmcr_channels(baselines, exmcr_path, bridge_path)

        if gram_coh_ia:
            print(f"  gram_coh_ia_exmcr: n={len(gram_coh_ia)}, "
                  f"mean={np.mean(gram_coh_ia):.6f}, std={np.std(gram_coh_ia, ddof=1):.6f}")
            extend_calibration_with_exmcr(store, gram_coh_ia, gram_coh_tia if gram_coh_tia else None)
        if gram_coh_tia:
            print(f"  gram_coh_tia:      n={len(gram_coh_tia)}, "
                  f"mean={np.mean(gram_coh_tia):.6f}, std={np.std(gram_coh_tia, ddof=1):.6f}")

    # Uncertainty channels
    if not args.exmcr_only:
        print("\n--- Computing Uncertainty Channels ---")
        clip_path = str(PROB_CLIP_PATH) if PROB_CLIP_PATH.exists() else None
        clap_path = str(PROB_CLAP_PATH) if PROB_CLAP_PATH.exists() else None

        if clip_path:
            print(f"  CLIP adapter: {clip_path}")
        if clap_path:
            print(f"  CLAP adapter: {clap_path}")

        unc_ti, unc_ta = compute_uncertainty_channels(baselines, clip_path, clap_path)

        if unc_ti:
            print(f"  uncertainty_ti: n={len(unc_ti)}, "
                  f"mean={np.mean(unc_ti):.6f}, std={np.std(unc_ti, ddof=1):.6f}")
        if unc_ta:
            print(f"  uncertainty_ta: n={len(unc_ta)}, "
                  f"mean={np.mean(unc_ta):.6f}, std={np.std(unc_ta, ddof=1):.6f}")

        if unc_ti or unc_ta:
            extend_calibration_with_uncertainty(store, unc_ti, unc_ta if unc_ta else None)

    # Summary
    print(f"\n--- Final Calibration Channels ({len(store.distributions)}) ---")
    for name, ref in store.distributions.items():
        print(f"  {name:25s}: mean={ref.mean:.6f}, std={ref.std:.6f}, n={ref.n}")

    # Save
    if not args.dry_run:
        store.save(str(CALIBRATION_PATH))
        print(f"\nSaved to {CALIBRATION_PATH}")
    else:
        print("\n[DRY RUN] Not saving.")

    print("=" * 70)


if __name__ == "__main__":
    main()
