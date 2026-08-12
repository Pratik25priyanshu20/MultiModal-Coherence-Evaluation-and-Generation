#!/usr/bin/env python3
"""
Benchmark Data Download (Task 1.3).

Downloads and prepares external benchmark datasets:
1. AudioCaps test set (~5K audio-caption pairs)
2. VGGSound subset (~5K video-audio pairs with category labels)

Preferred mode (--huggingface): loads from HuggingFace Hub directly.
No yt-dlp needed — audio bytes come with the dataset.

Fallback mode: uses yt-dlp to download from YouTube (slow, unreliable).

Usage:
    python scripts/download_benchmarks.py --audiocaps --huggingface
    python scripts/download_benchmarks.py --audiocaps --huggingface --max-samples 500
    python scripts/download_benchmarks.py --vggsound --huggingface
    python scripts/download_benchmarks.py --all --huggingface
    python scripts/download_benchmarks.py --audiocaps              # yt-dlp fallback
    python scripts/download_benchmarks.py --audiocaps --validate
    python scripts/download_benchmarks.py --audiocaps --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"
AUDIOCAPS_DIR = BENCHMARK_DIR / "audiocaps"
VGGSOUND_DIR = BENCHMARK_DIR / "vggsound"

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY_SEC = 3


def download_audiocaps_hf(max_samples: int = 5000) -> None:
    """Download AudioCaps from HuggingFace Hub (no yt-dlp needed).

    Tries sources in order of preference:
    1. OpenSound/AudioCaps — full AudioCaps with embedded audio (43.9 GB)
    2. CLAPv2/Clotho — smaller audio captioning benchmark (12.5 GB)
    """
    print("\n--- Downloading AudioCaps via HuggingFace ---")
    AUDIOCAPS_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = AUDIOCAPS_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: pip install datasets  (required for HuggingFace mode)")
        return

    # Try HuggingFace sources that include actual audio bytes
    # Use streaming mode to avoid downloading the full dataset
    ds = None
    source_name = None
    id_col = "audiocap_id"
    caption_col = "caption"
    sources = [
        ("OpenSound/AudioCaps", "test", "audiocap_id", "caption"),
        ("CLAPv2/Clotho", "test", "index", "text"),
    ]
    for dataset_id, split, _id_col, _cap_col in sources:
        try:
            print(f"  Trying: {dataset_id} ({split} split, streaming)...")
            ds = load_dataset(dataset_id, split=split, streaming=True)
            # Peek at first row to confirm audio column exists
            first = next(iter(ds))
            if "audio" not in first:
                print(f"  {dataset_id}: no 'audio' column (keys: {list(first.keys())})")
                ds = None
                continue
            print(f"  Connected to {dataset_id} (streaming)")
            source_name = dataset_id
            id_col = _id_col
            caption_col = _cap_col
            # Re-create iterator (peeking consumed first element)
            ds = load_dataset(dataset_id, split=split, streaming=True)
            break
        except Exception as e:
            print(f"  {dataset_id} failed: {e}")

    if ds is None:
        print("  ERROR: Could not load AudioCaps from HuggingFace.")
        print("  Fallback: use --audiocaps without --huggingface (requires yt-dlp)")
        return

    # Extract audio files and build manifest (streaming — only downloads what we need)
    import soundfile as sf
    import numpy as np
    manifest = []
    errors = 0

    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        try:
            audio_data = row.get("audio")
            caption = row.get(caption_col, row.get("caption", row.get("text", "")))

            if audio_data is None:
                errors += 1
                continue

            entry_id = row.get(id_col, row.get("audiocap_id", row.get("id", str(i))))
            safe_id = str(entry_id).replace("/", "_").replace(" ", "_")[:80]
            output_path = audio_dir / f"ac_{safe_id}.wav"

            if not output_path.exists():
                written = False
                # Try dict-style indexing (AudioDecoder or dict with 'array')
                try:
                    array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data["sampling_rate"]
                    sf.write(str(output_path), array, sr)
                    written = True
                except (TypeError, KeyError, IndexError):
                    pass
                if not written and isinstance(audio_data, dict) and "array" in audio_data:
                    array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data.get("sampling_rate", 16000)
                    sf.write(str(output_path), array, sr)
                    written = True
                if not written and isinstance(audio_data, dict) and "bytes" in audio_data:
                    output_path.write_bytes(audio_data["bytes"])
                    written = True
                if not written and isinstance(audio_data, bytes):
                    output_path.write_bytes(audio_data)
                    written = True
                if not written:
                    errors += 1
                    if errors <= 5:
                        print(f"  Unknown audio format at sample {i}: {type(audio_data)}")
                    continue

            manifest.append({
                "audiocap_id": str(entry_id),
                "caption": str(caption),
                "audio_path": str(output_path),
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("Error on sample %d: %s", i, e)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{max_samples}] extracted, {errors} errors")

    manifest_path = AUDIOCAPS_DIR / "manifest.json"
    dataset_label = source_name.replace("/", "_")
    with open(manifest_path, "w") as f:
        json.dump({"dataset": f"{dataset_label}_test_hf", "entries": manifest}, f, indent=2)

    print(f"  Done: {len(manifest)} entries saved ({errors} errors)")
    print(f"  Manifest: {manifest_path}")


def download_vggsound_hf(max_samples: int = 5000) -> None:
    """Download VGGSound from HuggingFace Hub (no yt-dlp needed).

    Uses 'Loie/VGGSound' or similar HF-hosted version.
    """
    print("\n--- Downloading VGGSound via HuggingFace ---")
    VGGSOUND_DIR.mkdir(parents=True, exist_ok=True)
    audio_dir = VGGSOUND_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: pip install datasets  (required for HuggingFace mode)")
        return

    ds = None
    source_name = None
    vgs_sources = [
        ("Loie/VGGSound", "test"),
        ("Loie/VGGSound", "train"),
    ]
    for dataset_id, split in vgs_sources:
        try:
            print(f"  Trying: {dataset_id} ({split} split, streaming)...")
            ds = load_dataset(dataset_id, split=split, streaming=True)
            first = next(iter(ds))
            if "audio" not in first:
                print(f"  {dataset_id}: no 'audio' column (keys: {list(first.keys())})")
                ds = None
                continue
            print(f"  Connected to {dataset_id} (streaming)")
            source_name = dataset_id
            ds = load_dataset(dataset_id, split=split, streaming=True)
            break
        except Exception as e:
            print(f"  {dataset_id} ({split}) failed: {e}")

    if ds is None:
        print("  ERROR: Could not load VGGSound from HuggingFace.")
        print("  Fallback: use --vggsound without --huggingface (requires yt-dlp)")
        return

    import soundfile as sf
    import numpy as np
    manifest = []
    errors = 0

    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        try:
            audio_data = row.get("audio")
            caption = row.get("label", row.get("text", row.get("caption", "")))
            vid_id = row.get("video_id", row.get("id", str(i)))
            output_path = audio_dir / f"vgs_{vid_id}.wav"

            if audio_data is None:
                errors += 1
                continue

            if not output_path.exists():
                written = False
                try:
                    array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data["sampling_rate"]
                    sf.write(str(output_path), array, sr)
                    written = True
                except (TypeError, KeyError, IndexError):
                    pass
                if not written and isinstance(audio_data, dict) and "array" in audio_data:
                    array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data.get("sampling_rate", 16000)
                    sf.write(str(output_path), array, sr)
                    written = True
                if not written and isinstance(audio_data, dict) and "bytes" in audio_data:
                    output_path.write_bytes(audio_data["bytes"])
                    written = True
                if not written and isinstance(audio_data, bytes):
                    output_path.write_bytes(audio_data)
                    written = True
                if not written:
                    errors += 1
                    continue

            manifest.append({
                "video_id": str(vid_id),
                "caption": str(caption),
                "audio_path": str(output_path),
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("Error on sample %d: %s", i, e)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(ds)}] extracted, {errors} errors")

    manifest_path = VGGSOUND_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"dataset": "VGGSound_test_hf", "entries": manifest}, f, indent=2)

    print(f"  Done: {len(manifest)} entries saved ({errors} errors)")
    print(f"  Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# yt-dlp fallback (for when HuggingFace datasets are unavailable)
# ---------------------------------------------------------------------------


def _run_ytdlp_with_retries(
    cmd: list[str],
    timeout: int = 60,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """Run a yt-dlp command with retry logic.

    Args:
        cmd: The yt-dlp command and arguments.
        timeout: Seconds before the subprocess is killed.
        max_retries: Maximum number of retry attempts on failure (0 = no retries).

    Returns:
        True if the command succeeded, False otherwise.
    """
    for attempt in range(1 + max_retries):
        try:
            subprocess.run(cmd, timeout=timeout, capture_output=True, check=True)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            if attempt < max_retries:
                logger.debug(
                    "yt-dlp attempt %d/%d failed (%s), retrying in %ds...",
                    attempt + 1, 1 + max_retries, type(e).__name__, RETRY_DELAY_SEC,
                )
                time.sleep(RETRY_DELAY_SEC)
            else:
                logger.debug(
                    "yt-dlp failed after %d attempt(s): %s", 1 + max_retries, e,
                )
    return False


def _validate_audio_file(path: Path) -> bool:
    """Validate that an audio file is loadable and has positive duration.

    Uses librosa to attempt a load and checks that the duration > 0.

    Args:
        path: Path to a .wav audio file.

    Returns:
        True if the file is valid, False otherwise.
    """
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, duration=15)
        duration = len(y) / sr if sr > 0 else 0.0
        if duration <= 0:
            logger.warning("  Validation failed (zero duration): %s", path.name)
            return False
        return True
    except Exception as e:
        logger.warning("  Validation failed (%s): %s", e, path.name)
        return False


def _load_existing_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load an existing manifest and return a lookup keyed by audio_path.

    Returns:
        dict mapping audio_path -> entry dict.  Empty dict if file missing.
    """
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        return {e["audio_path"]: e for e in data.get("entries", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def download_audiocaps(max_samples: int = 5000, validate: bool = False, resume: bool = False):
    """Download AudioCaps test set.

    AudioCaps provides human-written captions for 10-second audio clips
    from YouTube. We use the test split for evaluation.

    Args:
        max_samples: Maximum number of entries to process.
        validate: If True, verify each downloaded audio file with librosa.
        resume: If True, load the existing manifest and skip entries already present.
    """
    print("\n--- Downloading AudioCaps ---")
    AUDIOCAPS_DIR.mkdir(parents=True, exist_ok=True)

    # Download the CSV metadata
    csv_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv"
    csv_path = AUDIOCAPS_DIR / "test.csv"

    if not csv_path.exists():
        print(f"  Downloading metadata from {csv_url}")
        try:
            import requests
            resp = requests.get(csv_url, timeout=30)
            resp.raise_for_status()
            csv_path.write_text(resp.text)
            print(f"  Saved: {csv_path}")
        except Exception as e:
            print(f"  ERROR downloading metadata: {e}")
            print(f"  Please manually download test.csv from AudioCaps GitHub")
            return
    else:
        print(f"  Metadata already exists: {csv_path}")

    # Parse CSV
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "audiocap_id": row.get("audiocap_id", ""),
                "youtube_id": row.get("youtube_id", ""),
                "start_time": int(row.get("start_time", 0)),
                "caption": row.get("caption", ""),
            })

    print(f"  {len(entries)} AudioCaps test entries")

    # Limit samples
    entries = entries[:max_samples]

    audio_dir = AUDIOCAPS_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Resume support: load existing manifest to skip already-processed entries
    manifest_path = AUDIOCAPS_DIR / "manifest.json"
    existing_manifest = _load_existing_manifest(manifest_path) if resume else {}
    if resume and existing_manifest:
        print(f"  Resume mode: {len(existing_manifest)} entries already in manifest")

    # Download audio via yt-dlp
    downloaded = 0
    skipped = 0
    failed = 0
    resumed = 0

    for i, entry in enumerate(entries):
        ytid = entry["youtube_id"]
        start = entry["start_time"]
        output_path = audio_dir / f"{ytid}_{start}.wav"

        # Resume: skip if already in the manifest
        if resume and str(output_path) in existing_manifest:
            resumed += 1
            continue

        if output_path.exists():
            skipped += 1
            continue

        url = f"https://www.youtube.com/watch?v={ytid}"
        cmd = [
            "yt-dlp", "-x", "--audio-format", "wav",
            "--postprocessor-args", f"-ss {start} -t 10",
            "-o", str(output_path),
            "--quiet", url,
        ]

        success = _run_ytdlp_with_retries(cmd, timeout=60)
        if success:
            downloaded += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(entries)} (downloaded={downloaded}, skipped={skipped}, failed={failed}, resumed={resumed})")

    print(f"  Done: downloaded={downloaded}, skipped={skipped}, failed={failed}, resumed={resumed}")

    # Validation pass (optional)
    validated = 0
    invalid = 0
    if validate:
        print("  Validating audio files...")
        for entry in entries:
            ytid = entry["youtube_id"]
            start = entry["start_time"]
            audio_path = audio_dir / f"{ytid}_{start}.wav"
            if audio_path.exists():
                if _validate_audio_file(audio_path):
                    validated += 1
                else:
                    invalid += 1
                    # Remove invalid files so they are excluded from the manifest
                    audio_path.unlink(missing_ok=True)
        print(f"  Validation: {validated} valid, {invalid} invalid (removed)")

    # Save processed manifest (merge with existing if resuming)
    manifest_entries: dict[str, dict] = dict(existing_manifest) if resume else {}
    for entry in entries:
        ytid = entry["youtube_id"]
        start = entry["start_time"]
        audio_path = audio_dir / f"{ytid}_{start}.wav"
        if audio_path.exists():
            manifest_entries[str(audio_path)] = {
                **entry,
                "audio_path": str(audio_path),
            }

    manifest = list(manifest_entries.values())
    with open(manifest_path, "w") as f:
        json.dump({"dataset": "AudioCaps_test", "entries": manifest}, f, indent=2)

    print(f"  Manifest: {len(manifest)} entries saved to {manifest_path}")


def download_vggsound(max_samples: int = 5000, validate: bool = False, resume: bool = False):
    """Download VGGSound subset.

    VGGSound provides 10-second video clips with audio and category labels.
    We extract audio and use category text as captions.

    Args:
        max_samples: Maximum number of entries to process.
        validate: If True, verify each downloaded audio file with librosa.
        resume: If True, load the existing manifest and skip entries already present.
    """
    print("\n--- Downloading VGGSound ---")
    VGGSOUND_DIR.mkdir(parents=True, exist_ok=True)

    # VGGSound metadata is available from the official repository
    csv_url = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv"
    csv_path = VGGSOUND_DIR / "vggsound.csv"

    if not csv_path.exists():
        print(f"  Downloading metadata from {csv_url}")
        try:
            import requests
            resp = requests.get(csv_url, timeout=30)
            resp.raise_for_status()
            csv_path.write_text(resp.text)
        except Exception as e:
            print(f"  ERROR downloading metadata: {e}")
            print(f"  Please manually download vggsound.csv")
            return
    else:
        print(f"  Metadata already exists: {csv_path}")

    # Parse CSV (format: youtube_id, start_time, label, split)
    entries = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[3].strip() == "test":
                entries.append({
                    "youtube_id": row[0].strip(),
                    "start_time": int(row[1].strip()),
                    "label": row[2].strip(),
                    "split": row[3].strip(),
                })

    print(f"  {len(entries)} VGGSound test entries")
    entries = entries[:max_samples]

    audio_dir = VGGSOUND_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Resume support: load existing manifest to skip already-processed entries
    manifest_path = VGGSOUND_DIR / "manifest.json"
    existing_manifest = _load_existing_manifest(manifest_path) if resume else {}
    if resume and existing_manifest:
        print(f"  Resume mode: {len(existing_manifest)} entries already in manifest")

    # Download audio via yt-dlp with retry logic
    downloaded = 0
    skipped = 0
    failed = 0
    resumed = 0

    for i, entry in enumerate(entries):
        ytid = entry["youtube_id"]
        start = entry["start_time"]
        output_path = audio_dir / f"{ytid}_{start}.wav"

        # Resume: skip if already in the manifest
        if resume and str(output_path) in existing_manifest:
            resumed += 1
            continue

        if output_path.exists():
            skipped += 1
            continue

        url = f"https://www.youtube.com/watch?v={ytid}"
        cmd = [
            "yt-dlp", "-x", "--audio-format", "wav",
            "--postprocessor-args", f"-ss {start} -t 10",
            "-o", str(output_path),
            "--quiet", url,
        ]

        success = _run_ytdlp_with_retries(cmd, timeout=60)
        if success:
            downloaded += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(entries)} (downloaded={downloaded}, skipped={skipped}, failed={failed}, resumed={resumed})")

    print(f"  Done: downloaded={downloaded}, skipped={skipped}, failed={failed}, resumed={resumed}")

    # Validation pass (optional)
    validated = 0
    invalid = 0
    if validate:
        print("  Validating audio files...")
        for entry in entries:
            ytid = entry["youtube_id"]
            start = entry["start_time"]
            audio_path = audio_dir / f"{ytid}_{start}.wav"
            if audio_path.exists():
                if _validate_audio_file(audio_path):
                    validated += 1
                else:
                    invalid += 1
                    audio_path.unlink(missing_ok=True)
        print(f"  Validation: {validated} valid, {invalid} invalid (removed)")

    # Save manifest (merge with existing if resuming)
    manifest_entries: dict[str, dict] = dict(existing_manifest) if resume else {}
    for entry in entries:
        ytid = entry["youtube_id"]
        start = entry["start_time"]
        audio_path = audio_dir / f"{ytid}_{start}.wav"
        if audio_path.exists():
            manifest_entries[str(audio_path)] = {
                **entry,
                "audio_path": str(audio_path),
                "caption": entry["label"],  # Use category label as caption
            }

    manifest = list(manifest_entries.values())
    with open(manifest_path, "w") as f:
        json.dump({"dataset": "VGGSound_test", "entries": manifest}, f, indent=2)

    print(f"  Manifest: {len(manifest)} entries saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Benchmark Data")
    parser.add_argument("--audiocaps", action="store_true", help="Download AudioCaps")
    parser.add_argument("--vggsound", action="store_true", help="Download VGGSound")
    parser.add_argument("--all", action="store_true", help="Download all benchmarks")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples per dataset")
    parser.add_argument(
        "--huggingface", action="store_true",
        help="Download from HuggingFace Hub (recommended, no yt-dlp needed)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After downloading, verify audio files are valid (requires librosa)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing manifest and only process entries not yet present in it",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if "--debug" in sys.argv else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not (args.audiocaps or args.vggsound or args.all):
        print("No dataset specified. Use --audiocaps, --vggsound, or --all")
        parser.print_help()
        return

    print("=" * 70)
    print("Benchmark Data Download")
    if args.huggingface:
        print("  Mode: HuggingFace Hub (recommended)")
    else:
        print("  Mode: yt-dlp YouTube download (slow, requires yt-dlp)")
    print(f"  Max samples: {args.max_samples}")
    print("=" * 70)

    if args.huggingface:
        # HuggingFace mode — no yt-dlp needed
        if args.audiocaps or args.all:
            download_audiocaps_hf(args.max_samples)
        if args.vggsound or args.all:
            download_vggsound_hf(args.max_samples)
    else:
        # yt-dlp fallback
        if args.audiocaps or args.all:
            download_audiocaps(args.max_samples, validate=args.validate, resume=args.resume)
        if args.vggsound or args.all:
            download_vggsound(args.max_samples, validate=args.validate, resume=args.resume)

    print("\nDone!")


if __name__ == "__main__":
    main()
