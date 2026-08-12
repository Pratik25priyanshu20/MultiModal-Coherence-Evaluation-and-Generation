#!/usr/bin/env python3
"""
Build Gemini Calibration Reference Distributions.

Embeds RQ1 baseline samples (matched condition) via Gemini and fits
CalibrationStore with Gemini-specific channels:
    - gram_coh_ti_gemini: text-image gram coherence
    - gram_coh_ta_gemini: text-audio gram coherence
    - gram_coh_ia_gemini: image-audio gram coherence (NEW!)
    - gram_coh_tia_gemini: exact 3D gram coherence (NEW!)

These reference distributions enable z-score normalization in cMSCI v2.

Usage:
    python scripts/build_gemini_calibration.py

API cost estimate: ~90 API calls, ~$0.74
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coherence.calibration import CalibrationStore
from src.coherence.gram_volume import (
    gram_volume_2d,
    gram_volume_3d,
    normalized_gram_coherence,
)
from src.coherence.multiscale_gramian import MultiscaleGramian
from src.config.settings import GEMINI_CALIBRATION_PATH, MATRYOSHKA_DIMS

RQ1_RESULTS_PATH = PROJECT_ROOT / "runs" / "rq1" / "rq1_results.json"


def main():
    print("=" * 60)
    print("Building Gemini Calibration Reference Distributions")
    print("=" * 60)

    # Load RQ1 baseline results
    if not RQ1_RESULTS_PATH.exists():
        print(f"ERROR: RQ1 results not found at {RQ1_RESULTS_PATH}")
        sys.exit(1)

    with open(RQ1_RESULTS_PATH) as f:
        rq1_data = json.load(f)

    # Filter to baseline condition only (matched image + audio)
    baseline_samples = [
        r for r in rq1_data["results"]
        if r.get("condition") == "baseline"
    ]
    print(f"  Found {len(baseline_samples)} baseline samples from RQ1")

    if not baseline_samples:
        print("ERROR: No baseline samples found")
        sys.exit(1)

    # Initialize Gemini embedder
    from src.embeddings.gemini_embedder import GeminiEmbedder
    embedder = GeminiEmbedder(enable_cache=True)

    # Embed all baseline samples and compute gram coherences
    gram_coh_ti_list = []
    gram_coh_ta_list = []
    gram_coh_ia_list = []
    gram_coh_tia_list = []
    cos_ti_list = []
    cos_ta_list = []
    cos_ia_list = []

    t0 = time.time()
    for idx, sample in enumerate(baseline_samples):
        text = sample.get("prompt_text", sample.get("text", ""))
        image_path = sample.get("image_path")
        audio_path = sample.get("audio_path")

        if not text:
            continue

        try:
            emb_t = embedder.embed_text(text)
            emb_i = embedder.embed_image(image_path) if image_path and Path(image_path).exists() else None
            emb_a = embedder.embed_audio(audio_path) if audio_path and Path(audio_path).exists() else None

            # Cosine similarities
            if emb_i is not None:
                cos_ti = float(np.dot(emb_t, emb_i))
                cos_ti_list.append(cos_ti)
                vol_ti = gram_volume_2d(emb_t, emb_i)
                gram_coh_ti_list.append(normalized_gram_coherence(vol_ti))

            if emb_a is not None:
                cos_ta = float(np.dot(emb_t, emb_a))
                cos_ta_list.append(cos_ta)
                vol_ta = gram_volume_2d(emb_t, emb_a)
                gram_coh_ta_list.append(normalized_gram_coherence(vol_ta))

            if emb_i is not None and emb_a is not None:
                cos_ia = float(np.dot(emb_i, emb_a))
                cos_ia_list.append(cos_ia)
                vol_ia = gram_volume_2d(emb_i, emb_a)
                gram_coh_ia_list.append(normalized_gram_coherence(vol_ia))

                vol_tia = gram_volume_3d(emb_t, emb_i, emb_a)
                gram_coh_tia_list.append(normalized_gram_coherence(vol_tia, n_vectors=3))

            print(f"  [{idx+1}/{len(baseline_samples)}] {text[:50]}...", end="\r")

        except Exception as e:
            print(f"  Skipped sample {idx}: {e}")

    elapsed = time.time() - t0
    print(f"\n  Embedding completed in {elapsed:.1f}s")

    # Build calibration store
    store = CalibrationStore()

    if gram_coh_ti_list:
        store.add("gram_coh_ti_gemini", gram_coh_ti_list)
    if gram_coh_ta_list:
        store.add("gram_coh_ta_gemini", gram_coh_ta_list)
    if gram_coh_ia_list:
        store.add("gram_coh_ia_gemini", gram_coh_ia_list)
    if gram_coh_tia_list:
        store.add("gram_coh_tia_gemini", gram_coh_tia_list)

    # Also store cosine distributions for reference
    if cos_ti_list:
        store.add("cos_ti_gemini", cos_ti_list)
    if cos_ta_list:
        store.add("cos_ta_gemini", cos_ta_list)
    if cos_ia_list:
        store.add("cos_ia_gemini", cos_ia_list)

    # ── Multi-scale calibration (Matryoshka truncation) ──
    print("\n--- Building Multi-Scale Calibration ---")
    ms = MultiscaleGramian()

    # Collect multi-scale coherences per sample
    ms_channels: dict[str, list] = {}  # e.g. "gram_coh_ti_768" -> [values]

    for idx, sample in enumerate(baseline_samples):
        text = sample.get("prompt_text", sample.get("text", ""))
        image_path = sample.get("image_path")
        audio_path = sample.get("audio_path")
        if not text:
            continue

        try:
            emb_t = embedder.embed_text(text)
            emb_i = embedder.embed_image(image_path) if image_path and Path(image_path).exists() else None
            emb_a = embedder.embed_audio(audio_path) if audio_path and Path(audio_path).exists() else None

            ms_result = ms.compute(emb_t, emb_i, emb_a)
            per_scale = ms_result["per_scale"]

            for dim in MATRYOSHKA_DIMS:
                for ch in ["ti", "ta", "ia", "tia"]:
                    if ch in per_scale.get(dim, {}):
                        key = f"gram_coh_{ch}_{dim}"
                        ms_channels.setdefault(key, []).append(per_scale[dim][ch])

            # Fused channels
            for ch in ["ti", "ta", "ia", "tia"]:
                val = ms_result.get(f"fused_{ch}")
                if val is not None:
                    ms_channels.setdefault(f"gram_coh_{ch}_fused", []).append(val)

        except Exception as e:
            print(f"  Multi-scale skip {idx}: {e}")

    for key, values in sorted(ms_channels.items()):
        if values:
            store.add(key, values)

    print(f"  Added {len(ms_channels)} multi-scale calibration channels")

    # Save
    GEMINI_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(str(GEMINI_CALIBRATION_PATH))

    print(f"\n--- Calibration Summary ---")
    for name, ref in store.distributions.items():
        print(f"  {name}: mean={ref.mean:.6f}, std={ref.std:.6f}, n={ref.n}")

    print(f"\nCalibration saved to {GEMINI_CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
