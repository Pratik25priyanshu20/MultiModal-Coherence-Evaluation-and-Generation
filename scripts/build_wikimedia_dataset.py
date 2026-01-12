"""
build_wikimedia_dataset.py

SAFE Wikimedia dataset builder (evaluation only)

- Handles rate limits (429)
- Skips non-image files
- Uses polite delays + retry
- Produces samples.json
"""

import json
import time
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote

# =========================
# CONFIG
# =========================

API_URL = "https://commons.wikimedia.org/w/api.php"

OUT_DIR = Path("data/wikimedia")
IMG_DIR = OUT_DIR / "images"
OUT_FILE = OUT_DIR / "samples.json"

TARGET_SAMPLES = 50            # increase later if needed
BASE_SLEEP = 1.5               # base delay (seconds)
MAX_RETRIES = 5

HEADERS = {
    "User-Agent": "Multimodal-Coherence-AI/1.0 (academic project; polite crawler)"
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

IMG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# API PARAMS
# =========================

params = {
    "action": "query",
    "generator": "categorymembers",
    "gcmtitle": "Category:Featured_pictures_on_Wikimedia_Commons",
    "gcmtype": "file",
    "gcmlimit": 50,
    "prop": "imageinfo",
    "iiprop": "url|extmetadata",
    "format": "json",
    "origin": "*"
}

print("Fetching metadata from Wikimedia API...")

samples = []
count = 0
continue_token = {}

# =========================
# MAIN LOOP
# =========================

while count < TARGET_SAMPLES:
    merged = params | continue_token
    r = requests.get(API_URL, params=merged, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "query" not in data:
        print("No query results returned, stopping.")
        break

    pages = data["query"]["pages"]

    for page in pages.values():
        if count >= TARGET_SAMPLES:
            break

        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue

        url = imageinfo[0]["url"]
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()

        # Skip non-images
        if ext not in ALLOWED_EXTENSIONS:
            print(f"Skipping non-image: {parsed.path}")
            continue

        metadata = imageinfo[0].get("extmetadata", {})
        caption = (
            metadata.get("ImageDescription", {}).get("value")
            or page["title"]
        )

        filename = unquote(Path(parsed.path).name)
        local_name = f"wm_{count+1:03d}_{filename}"
        img_path = IMG_DIR / local_name

        print(f"Downloading {local_name}")

        # ---------- SAFE DOWNLOAD ----------
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code == 429:
                    wait = BASE_SLEEP * attempt * 2
                    print(f"Rate limited (429). Sleeping {wait:.1f}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                img_path.write_bytes(resp.content)
                success = True
                break

            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(BASE_SLEEP * attempt)

        if not success:
            print(f"Skipping {local_name} after retries")
            continue

        samples.append({
            "id": f"wm_{count+1:03d}",
            "caption": caption,
            "image": f"images/{local_name}",
            "audio": None
        })

        count += 1
        time.sleep(BASE_SLEEP)

    if "continue" in data:
        continue_token = data["continue"]
    else:
        break

# =========================
# SAVE
# =========================

OUT_FILE.write_text(json.dumps(samples, indent=2), encoding="utf-8")

print("\nDONE")
print(f"Saved {len(samples)} samples")
print(f"Dataset: {OUT_FILE}")