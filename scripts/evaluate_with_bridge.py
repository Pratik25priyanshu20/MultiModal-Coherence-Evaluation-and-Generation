#!/usr/bin/env python3
"""
Re-evaluate experiment results with the trained cross-space bridge.

Loads a trained bridge, re-computes si_a for each result, and reports
MSCI with vs without the bridge contribution.

Usage:
    python scripts/evaluate_with_bridge.py runs/rq1/rq1_results.json
    python scripts/evaluate_with_bridge.py runs/rq1_gen/rq1_gen_results.json
    python scripts/evaluate_with_bridge.py runs/rq1/rq1_results.json --bridge models/bridge/bridge_best.pt

Status: READY TO RUN after bridge training. Not required for current experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate with cross-space bridge")
    parser.add_argument("results_json", help="Path to experiment results JSON")
    parser.add_argument("--bridge", default="models/bridge/bridge_best.pt",
                        help="Path to trained bridge weights")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <input>_with_bridge.json)")
    args = parser.parse_args()

    bridge_path = Path(args.bridge)
    if not bridge_path.exists():
        print(f"ERROR: Bridge not found at {bridge_path}")
        print("Train the bridge first: python scripts/train_bridge.py")
        return 1

    # Load results
    with open(args.results_json) as f:
        data = json.load(f)
    results = data["results"]

    print("=" * 70)
    print("RE-EVALUATION WITH CROSS-SPACE BRIDGE")
    print("=" * 70)
    print(f"  Results:  {args.results_json} ({len(results)} entries)")
    print(f"  Bridge:   {bridge_path}")

    # Load bridge + embedder
    from src.embeddings.cross_space_bridge import CrossSpaceBridge
    from src.embeddings.aligned_embeddings import AlignedEmbedder

    bridge = CrossSpaceBridge.load(bridge_path)
    embedder = AlignedEmbedder()

    # Re-evaluate
    comparisons: List[Dict[str, Any]] = []
    n_updated = 0

    for i, r in enumerate(results):
        if "error" in r:
            continue

        img_path = r.get("image_path")
        aud_path = r.get("audio_path")
        if not img_path or not aud_path:
            continue

        try:
            img_emb = embedder.embed_image(img_path)
            aud_emb = embedder.embed_audio(aud_path)
            si_a = bridge.compute_similarity(img_emb, aud_emb)

            # Recompute MSCI with si_a
            st_i = r.get("st_i")
            st_a = r.get("st_a")
            old_msci = r.get("msci")

            if st_i is not None and st_a is not None:
                new_msci = 0.45 * st_i + 0.45 * st_a + 0.10 * si_a
            else:
                new_msci = old_msci

            comparisons.append({
                "prompt_id": r.get("prompt_id"),
                "condition": r.get("condition", r.get("mode")),
                "old_msci": old_msci,
                "new_msci": round(new_msci, 4) if new_msci else None,
                "si_a": round(si_a, 4),
                "delta_msci": round(new_msci - old_msci, 4) if old_msci and new_msci else None,
            })

            # Update result in-place
            r["si_a"] = round(si_a, 4)
            r["msci_with_bridge"] = round(new_msci, 4) if new_msci else None
            n_updated += 1

        except Exception as e:
            print(f"  Warning: Failed for {r.get('prompt_id')}: {e}")

    # Summary
    print(f"\n  Updated: {n_updated} / {len(results)} entries")

    if comparisons:
        deltas = [c["delta_msci"] for c in comparisons if c["delta_msci"] is not None]
        si_a_vals = [c["si_a"] for c in comparisons]
        print(f"\n  si_a stats:   mean={np.mean(si_a_vals):.4f}  std={np.std(si_a_vals):.4f}")
        print(f"  MSCI delta:   mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")
        print(f"  MSCI range:   [{np.min(deltas):+.4f}, {np.max(deltas):+.4f}]")

    # Save
    output_path = args.output or args.results_json.replace(".json", "_with_bridge.json")
    data["bridge_evaluation"] = {
        "bridge_path": str(bridge_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_updated": n_updated,
        "comparisons_summary": {
            "mean_si_a": round(float(np.mean(si_a_vals)), 4) if si_a_vals else None,
            "mean_delta_msci": round(float(np.mean(deltas)), 4) if deltas else None,
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n  Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
