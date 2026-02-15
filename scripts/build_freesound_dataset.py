"""
build_freesound_dataset.py

Freesound audio dataset builder for multimodal coherence evaluation.

- Uses Freesound API (requires FREESOUND_API_KEY in .env)
- Downloads audio across domain-specific search queries
- Normalizes to WAV 48kHz mono
- Saves metadata with captions + domain tags

Usage:
    python scripts/build_freesound_dataset.py
    python scripts/build_freesound_dataset.py --target 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =========================
# CONFIG
# =========================

API_BASE = "https://freesound.org/apiv2"

OUT_DIR = Path("data/freesound")
AUD_DIR = OUT_DIR / "audio"
META_FILE = OUT_DIR / "samples.json"

DEFAULT_TARGET = 100
BASE_SLEEP = 1.0
MAX_RETRIES = 3

# Domain-specific search queries
DOMAIN_QUERIES = {
    "nature": [
        "forest ambience",
        "birds singing nature",
        "wind through trees",
        "rain forest",
        "crickets night",
        "thunder storm nature",
        "river stream flowing",
    ],
    "urban": [
        "city traffic ambience",
        "crowd people talking",
        "construction site",
        "car horn city",
        "subway train metro",
        "cafe restaurant interior",
        "street market",
    ],
    "water": [
        "ocean waves beach",
        "rain on roof",
        "waterfall flowing",
        "lake calm water",
        "rain puddle dripping",
        "underwater ambience",
        "stream babbling brook",
        "harbor boats water",
    ],
    "weather": [
        "thunder lightning storm",
        "heavy rain",
        "wind howling",
        "hailstorm",
    ],
    "animals": [
        "dog barking",
        "cat purring",
        "birds dawn chorus",
        "frogs pond night",
        "insects buzzing",
    ],
    "indoor": [
        "office ambience typing",
        "kitchen cooking",
        "fireplace crackling",
        "clock ticking room",
    ],
}

# Remap auxiliary domains to the 3 pipeline domains
DOMAIN_REMAP = {
    "weather": "nature",
    "animals": "nature",
    "indoor": "urban",
}


def get_api_key() -> str:
    """Get Freesound API key from environment."""
    key = os.environ.get("FREESOUND_API_KEY", "")
    if not key:
        print("ERROR: FREESOUND_API_KEY not set in environment or .env file.")
        print("Get a free key at: https://freesound.org/apiv2/apply/")
        print("Then add to .env: FREESOUND_API_KEY=your_key_here")
        sys.exit(1)
    return key


def search_sounds(query: str, api_key: str, page_size: int = 15) -> list[dict]:
    """Search Freesound for sounds matching a query."""
    params = {
        "query": query,
        "token": api_key,
        "page_size": page_size,
        "fields": "id,name,description,duration,previews,tags,avg_rating,num_ratings",
        "sort": "rating_desc",
        "filter": "duration:[5 TO 30] type:wav OR type:flac OR type:mp3",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(f"{API_BASE}/search/text/", params=params, timeout=30)
            if r.status_code == 429:
                wait = BASE_SLEEP * attempt * 2
                print(f"    Rate limited. Sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            if r.status_code == 401:
                print("ERROR: Invalid API key. Check FREESOUND_API_KEY.")
                sys.exit(1)
            r.raise_for_status()
            data = r.json()
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"    Search attempt {attempt} failed: {e}")
            time.sleep(BASE_SLEEP * attempt)

    return []


def download_preview(sound: dict, out_path: Path) -> bool:
    """Download the HQ preview of a sound (no OAuth needed for previews)."""
    previews = sound.get("previews", {})
    # Prefer HQ OGG or MP3 preview
    url = (
        previews.get("preview-hq-mp3")
        or previews.get("preview-hq-ogg")
        or previews.get("preview-lq-mp3")
    )
    if not url:
        return False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                time.sleep(BASE_SLEEP * attempt * 2)
                continue
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except Exception as e:
            print(f"      Download attempt {attempt} failed: {e}")
            time.sleep(BASE_SLEEP * attempt)

    return False


def convert_to_wav(in_path: Path, out_path: Path, target_sr: int = 48000) -> bool:
    """Convert audio to WAV 48kHz mono using librosa/soundfile."""
    try:
        import librosa
        import soundfile as sf
        import numpy as np

        y, sr = librosa.load(str(in_path), sr=target_sr, mono=True)
        # Normalize amplitude
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.95
        sf.write(str(out_path), y, target_sr)
        return True
    except Exception as e:
        print(f"      Conversion failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build Freesound audio dataset")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Target number of audio files (default: {DEFAULT_TARGET})")
    args = parser.parse_args()

    api_key = get_api_key()

    AUD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Freesound Audio Dataset Builder")
    print(f"Target: {args.target} audio files across {len(DOMAIN_QUERIES)} domains")
    print("=" * 70)

    # Distribute target across domains
    total_queries = sum(len(qs) for qs in DOMAIN_QUERIES.values())
    per_query = max(args.target // total_queries, 3)

    all_sounds = []  # (sound_meta, domain, query)
    seen_ids = set()

    for domain, queries in DOMAIN_QUERIES.items():
        print(f"\nDomain: {domain}")
        for query in queries:
            if len(all_sounds) >= args.target:
                break

            print(f"  Searching: \"{query}\" (target={per_query})")
            results = search_sounds(query, api_key, page_size=per_query + 5)

            added = 0
            for sound in results:
                if len(all_sounds) >= args.target:
                    break
                if added >= per_query:
                    break

                sid = sound["id"]
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)

                all_sounds.append((sound, domain, query))
                added += 1

            print(f"    Found {added} new sounds")
            time.sleep(BASE_SLEEP)

    print(f"\nTotal candidates: {len(all_sounds)}")

    # Download and convert
    samples = []
    for idx, (sound, domain, query) in enumerate(all_sounds):
        # Remap auxiliary domains to pipeline's 3 domains
        pipeline_domain = DOMAIN_REMAP.get(domain, domain)
        sid = sound["id"]
        name = sound.get("name", f"sound_{sid}").replace("/", "_").replace(" ", "_")
        safe_name = f"fs_{idx+1:03d}_{pipeline_domain}_{name}"[:80]

        wav_path = AUD_DIR / f"{safe_name}.wav"
        tmp_path = AUD_DIR / f"{safe_name}.tmp.mp3"

        # Skip already downloaded
        if wav_path.exists():
            samples.append({
                "id": f"fs_{idx+1:03d}",
                "freesound_id": sid,
                "caption": sound.get("description", name)[:200],
                "query": query,
                "domain": pipeline_domain,
                "audio": f"audio/{wav_path.name}",
                "duration": sound.get("duration"),
                "tags": sound.get("tags", [])[:10],
            })
            continue

        print(f"  [{idx+1}/{len(all_sounds)}] {safe_name[:50]}...")

        # Download preview
        if not download_preview(sound, tmp_path):
            print(f"    Skipping (download failed)")
            continue

        # Convert to WAV 48kHz
        if not convert_to_wav(tmp_path, wav_path):
            tmp_path.unlink(missing_ok=True)
            continue

        # Clean up temp
        tmp_path.unlink(missing_ok=True)

        samples.append({
            "id": f"fs_{idx+1:03d}",
            "freesound_id": sid,
            "caption": sound.get("description", name)[:200],
            "query": query,
            "domain": pipeline_domain,
            "audio": f"audio/{wav_path.name}",
            "duration": sound.get("duration"),
            "tags": sound.get("tags", [])[:10],
        })
        time.sleep(BASE_SLEEP)

    # Save metadata
    META_FILE.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    # Summary
    domain_counts = {}
    for s in samples:
        d = s["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print(f"\nDONE: {len(samples)} audio files saved")
    print(f"Metadata: {META_FILE}")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
