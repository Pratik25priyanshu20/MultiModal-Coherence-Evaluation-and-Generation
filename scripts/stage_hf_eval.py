#!/usr/bin/env python3
"""
Stage files for Hugging Face Spaces deployment.

Copies the self-contained app, rewrites media paths in sample JSONs to
relative form, and copies only the referenced media files into a flat
deploy directory.

Usage:
    python scripts/stage_hf_eval.py [--output /tmp/hf_eval_deploy]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path("/tmp/hf_eval_deploy")

# Directories where media files may live (searched in order)
IMAGE_SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "wikimedia" / "images",
    PROJECT_ROOT / "data" / "generated" / "eval_images",
    PROJECT_ROOT / "data" / "processed" / "images",
]
AUDIO_SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "freesound" / "audio",
    PROJECT_ROOT / "data" / "benchmarks" / "audiocaps" / "audio",
    PROJECT_ROOT / "data" / "processed" / "audio",
    PROJECT_ROOT / "data" / "audiocaps" / "audio",
]

SAMPLE_FILES = [
    PROJECT_ROOT / "runs" / "rq3" / "rq3_samples.json",
    PROJECT_ROOT / "runs" / "rq3" / "rq3_samples_extended.json",
]


def find_media_file(filename: str, search_dirs: list[Path]) -> Path | None:
    """Search for a media file across multiple directories."""
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


def extract_filename(path_str: str) -> str:
    """Get the filename from an absolute or relative path string."""
    return Path(path_str).name


def rewrite_and_copy_samples(
    src_path: Path,
    out_samples_dir: Path,
    out_media_dir: Path,
    copied_images: dict[str, Path],
    copied_audio: dict[str, Path],
    missing: list[str],
):
    """Rewrite a sample JSON: convert paths to relative, copy referenced media."""
    if not src_path.exists():
        print(f"  SKIP (not found): {src_path.name}")
        return

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images_dir = out_media_dir / "images"
    audio_dir = out_media_dir / "audio"
    images_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    for sample in data["samples"]:
        # --- Image ---
        img_name = extract_filename(sample["image_path"])
        if img_name not in copied_images:
            # Try to resolve from absolute path first
            abs_path = Path(sample["image_path"])
            if abs_path.is_absolute() and abs_path.exists():
                src_img = abs_path
            else:
                # Try relative to PROJECT_ROOT
                rel_path = PROJECT_ROOT / sample["image_path"]
                if rel_path.exists():
                    src_img = rel_path
                else:
                    src_img = find_media_file(img_name, IMAGE_SEARCH_DIRS)

            if src_img:
                shutil.copy2(src_img, images_dir / img_name)
                copied_images[img_name] = src_img
            else:
                missing.append(f"IMAGE: {img_name} (sample {sample['sample_id']})")
        sample["image_path"] = f"images/{img_name}"

        # --- Audio ---
        aud_name = extract_filename(sample["audio_path"])
        if aud_name not in copied_audio:
            abs_path = Path(sample["audio_path"])
            if abs_path.is_absolute() and abs_path.exists():
                src_aud = abs_path
            else:
                rel_path = PROJECT_ROOT / sample["audio_path"]
                if rel_path.exists():
                    src_aud = rel_path
                else:
                    src_aud = find_media_file(aud_name, AUDIO_SEARCH_DIRS)

            if src_aud:
                shutil.copy2(src_aud, audio_dir / aud_name)
                copied_audio[aud_name] = src_aud
            else:
                missing.append(f"AUDIO: {aud_name} (sample {sample['sample_id']})")
        sample["audio_path"] = f"audio/{aud_name}"

    # Write rewritten JSON
    out_path = out_samples_dir / src_path.name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {out_path.name} ({len(data['samples'])} samples)")


def main():
    parser = argparse.ArgumentParser(description="Stage HF eval deployment")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory (default: /tmp/hf_eval_deploy)")
    args = parser.parse_args()

    out = args.output
    if out.exists():
        shutil.rmtree(out)

    print(f"Staging HF eval deploy to: {out}")

    # Ensure output dirs exist
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "media").mkdir(parents=True, exist_ok=True)
    (out / "sessions").mkdir(parents=True, exist_ok=True)

    # 1. Copy app files
    hf_eval_dir = PROJECT_ROOT / "deploy" / "hf-eval"
    for item in ["app.py", "requirements.txt", "README.md"]:
        src = hf_eval_dir / item
        if src.exists():
            shutil.copy2(src, out / item)

    # Copy .streamlit directory
    streamlit_src = hf_eval_dir / ".streamlit"
    if streamlit_src.exists():
        shutil.copytree(streamlit_src, out / ".streamlit")

    # 2. Rewrite sample JSONs and copy media
    copied_images: dict[str, Path] = {}
    copied_audio: dict[str, Path] = {}
    missing: list[str] = []

    print("\nProcessing sample files:")
    for sf in SAMPLE_FILES:
        rewrite_and_copy_samples(
            sf, out / "samples", out / "media",
            copied_images, copied_audio, missing,
        )

    # 3. Report
    print(f"\n--- Staging Summary ---")
    print(f"Images copied: {len(copied_images)}")
    print(f"Audio copied:  {len(copied_audio)}")
    print(f"Total media:   {len(copied_images) + len(copied_audio)} files")

    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    else:
        print("All media files found.")

    # Calculate total size
    total_bytes = 0
    for d in [out / "media" / "images", out / "media" / "audio"]:
        if d.exists():
            for f in d.iterdir():
                total_bytes += f.stat().st_size
    print(f"Media size:    {total_bytes / (1024*1024):.1f} MB")
    print(f"\nDeploy directory: {out}")
    print("Test locally:  cd {} && streamlit run app.py".format(out))


if __name__ == "__main__":
    main()
