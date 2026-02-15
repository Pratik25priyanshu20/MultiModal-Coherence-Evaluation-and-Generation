"""
PHASE 1A — Diagnose Retrieval Quality

Runs test prompts against the image and audio indexes.
For each prompt: logs top-5 candidates with similarity scores.
Answers: Is retrieval semantically reasonable? Are scores near zero?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity


TEST_PROMPTS = [
    "A peaceful forest at dawn with birdsong",
    "A bustling city street at night with neon lights",
    "Ocean waves crashing on a sandy beach at sunset",
    "A rainy day in a European city with cobblestone streets",
    "A snowy mountain peak under a clear blue sky",
    "Children playing in a sunlit park with green trees",
    "A quiet library with wooden shelves and warm lighting",
    "A desert landscape with sand dunes and a hot sun",
    "A tropical jungle with exotic birds and a waterfall",
    "A foggy morning in a rural countryside with fields",
]


def diagnose_image_retrieval(index_path: str = "data/embeddings/image_index.npz") -> dict:
    """Diagnose image retrieval for all test prompts."""
    path = Path(index_path)
    if not path.exists():
        print(f"ERROR: Image index not found at {path}")
        return {}

    data = np.load(path, allow_pickle=True)
    ids = data["ids"].tolist()
    embs = data["embs"].astype("float32")

    embedder = AlignedEmbedder(target_dim=512, enable_cache=False)

    print("=" * 80)
    print("IMAGE RETRIEVAL DIAGNOSIS")
    print(f"Index size: {len(ids)} images")
    print("=" * 80)

    all_scores = []
    results = {}

    for prompt in TEST_PROMPTS:
        query_emb = embedder.embed_text(prompt)
        scored = []
        for img_path, img_emb in zip(ids, embs):
            sim = cosine_similarity(query_emb, img_emb)
            scored.append((Path(img_path).name, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        print(f"\nPrompt: \"{prompt}\"")
        print(f"  {'Rank':<6} {'Image':<35} {'Similarity':>10}")
        print(f"  {'─'*6} {'─'*35} {'─'*10}")
        for rank, (name, sim) in enumerate(scored[:5], 1):
            marker = " <<<" if rank == 1 and sim < 0.15 else ""
            print(f"  {rank:<6} {name:<35} {sim:>10.4f}{marker}")
            all_scores.append(sim)

        top_name, top_score = scored[0]
        results[prompt] = {
            "top_match": top_name,
            "top_score": top_score,
            "top_5": scored[:5],
        }

    # Distribution summary
    all_scores_arr = np.array(all_scores)
    print("\n" + "=" * 80)
    print("SIMILARITY DISTRIBUTION (all top-5 scores)")
    print(f"  Mean:   {all_scores_arr.mean():.4f}")
    print(f"  Std:    {all_scores_arr.std():.4f}")
    print(f"  Min:    {all_scores_arr.min():.4f}")
    print(f"  Max:    {all_scores_arr.max():.4f}")
    print(f"  Median: {np.median(all_scores_arr):.4f}")
    print(f"  < 0.10: {(all_scores_arr < 0.10).sum()} / {len(all_scores_arr)}")
    print(f"  < 0.15: {(all_scores_arr < 0.15).sum()} / {len(all_scores_arr)}")
    print(f"  < 0.20: {(all_scores_arr < 0.20).sum()} / {len(all_scores_arr)}")
    print(f"  ≥ 0.25: {(all_scores_arr >= 0.25).sum()} / {len(all_scores_arr)}")

    # Obvious mismatch detection
    print("\n" + "=" * 80)
    print("MISMATCH ANALYSIS")
    mismatches = []
    for prompt, res in results.items():
        top = res["top_match"].lower()
        prompt_lower = prompt.lower()
        # Check domain mismatches
        if any(k in prompt_lower for k in ["forest", "jungle", "tree", "rural", "countryside"]):
            if any(k in top for k in ["city", "neon", "urban"]):
                mismatches.append((prompt, res["top_match"], res["top_score"]))
        elif any(k in prompt_lower for k in ["city", "street", "urban", "neon"]):
            if any(k in top for k in ["beach", "ocean", "forest", "mountain"]):
                mismatches.append((prompt, res["top_match"], res["top_score"]))
        elif any(k in prompt_lower for k in ["beach", "ocean", "wave", "sea"]):
            if any(k in top for k in ["city", "neon", "forest"]):
                mismatches.append((prompt, res["top_match"], res["top_score"]))

    if mismatches:
        print(f"  FOUND {len(mismatches)} OBVIOUS MISMATCHES:")
        for prompt, match, score in mismatches:
            print(f"    \"{prompt[:50]}\" → {match} (sim={score:.4f})")
    else:
        print("  No obvious domain mismatches in top-1 results.")

    return results


def diagnose_audio_retrieval(index_path: str = "data/embeddings/audio_index.npz") -> dict:
    """Diagnose audio index coverage."""
    path = Path(index_path)
    if not path.exists():
        print(f"\nERROR: Audio index not found at {path}")
        return {}

    data = np.load(path, allow_pickle=True)
    ids = data["ids"].tolist()
    embs = data["embs"].astype("float32")

    print("\n" + "=" * 80)
    print("AUDIO INDEX DIAGNOSIS")
    print(f"Index size: {len(ids)} audio files")
    print("=" * 80)
    for i, audio_id in enumerate(ids):
        print(f"  {i+1}. {Path(audio_id).name}")

    # Check available audio beyond index
    audio_dirs = [
        Path("data/processed/audio"),
        Path("data/audiocaps/audio"),
        Path("data/wikimedia/audio"),
    ]
    print("\nAvailable audio files (not all in index):")
    for d in audio_dirs:
        if d.exists():
            files = list(d.glob("*.wav")) + list(d.glob("*.mp3"))
            print(f"  {d}: {len(files)} files")

    return {"ids": ids, "count": len(ids)}


def diagnose_data_coverage():
    """Report overall data coverage."""
    print("\n" + "=" * 80)
    print("DATA COVERAGE SUMMARY")
    print("=" * 80)

    img_dirs = {
        "data/processed/images": "Indexed (in image_index.npz)",
        "data/wikimedia/images": "NOT indexed",
    }
    for d, status in img_dirs.items():
        p = Path(d)
        if p.exists():
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            files = [f for f in p.iterdir() if f.suffix.lower() in exts]
            print(f"  {d}: {len(files)} images — {status}")
        else:
            print(f"  {d}: MISSING")

    aud_dirs = {
        "data/processed/audio": "Indexed (in audio_index.npz)",
        "data/audiocaps/audio": "NOT indexed",
        "data/wikimedia/audio": "NOT indexed",
    }
    for d, status in aud_dirs.items():
        p = Path(d)
        if p.exists():
            exts = {".wav", ".mp3", ".flac", ".ogg"}
            files = [f for f in p.iterdir() if f.suffix.lower() in exts]
            print(f"  {d}: {len(files)} audio files — {status}")
        else:
            print(f"  {d}: MISSING")

    print("\n  VERDICT:")
    print("  Image index covers ONLY data/processed/images (9 files)")
    print("  Audio index covers ONLY data/processed/audio (5 files)")
    print("  Wikimedia images (18) and AudioCaps audio (50) are NOT indexed")
    print("  → Rebuild indexes to include all available data")


if __name__ == "__main__":
    diagnose_data_coverage()
    img_results = diagnose_image_retrieval()
    aud_results = diagnose_audio_retrieval()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("  1. Rebuild image index to include wikimedia images (9 → ~27)")
    print("  2. Rebuild audio index to include audiocaps (5 → ~55)")
    print("  3. Add domain tags to enable domain gating")
    print("  4. Raise min_similarity threshold based on distribution")
    print("  5. Add retrieval_failed metadata when no good match found")
