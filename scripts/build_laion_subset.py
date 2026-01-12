import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/laion")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "samples.json"

N = 500  # number of prompts we want
DATASET_NAME = "conceptual_captions"
DATASET_CONFIG = "labeled"
SPLIT = "train"

def extract_caption(row):
    """
    Try all common LAION caption fields.
    Returns first non-empty string.
    """
    for key in [
        "caption",
        "text",
        "TEXT",
        "original_caption",
        "laion_caption",
        "url_caption",
        "alt_text",
    ]:
        if key in row and row[key]:
            return str(row[key]).strip()
    return None

def main():
    print("Loading captioned dataset (streaming)…")
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=SPLIT,
        streaming=True,
    )

    print("Dataset columns:", ds.column_names)

    samples = []
    for row in ds:
        caption = extract_caption(row)
        if not caption:
            continue

        samples.append({
            "id": f"la_{len(samples)+1:03d}",
            "caption": caption,
            "image": None,
            "audio": None
        })

        if len(samples) >= N:
            break

    OUT_FILE.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nPrompt subset ready")
    print("Total samples:", len(samples))
    print("Saved to:", OUT_FILE)
    print("First sample:", samples[0] if samples else "NONE")

if __name__ == "__main__":
    main()
