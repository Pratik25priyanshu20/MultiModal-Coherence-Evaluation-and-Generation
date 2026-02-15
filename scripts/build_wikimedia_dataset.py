"""
build_wikimedia_dataset.py

Scene-focused Wikimedia dataset builder for multimodal coherence evaluation.

- Fetches from scene-specific categories (landscapes, cityscapes, seascapes, etc.)
- Recurses one level into subcategories to find more files
- Filters out non-photographic content (paintings, diagrams, logos, artworks, etc.)
- Validates image dimensions (min 200K pixels, aspect ratio 0.4-4.0)
- Domain-tags images at download time with per-domain targets
- Handles rate limits (429) with polite delays + retry
- Produces samples.json with domain metadata
"""

import json
import re
import time
import requests
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, unquote

# =========================
# CONFIG
# =========================

API_URL = "https://commons.wikimedia.org/w/api.php"

OUT_DIR = Path("data/wikimedia")
IMG_DIR = OUT_DIR / "images"
OUT_FILE = OUT_DIR / "samples.json"

# Per-domain targets for balanced coverage
DOMAIN_TARGETS = {
    "nature": 25,
    "urban": 25,
    "water": 25,
}

BASE_SLEEP = 2.0
MAX_RETRIES = 8

HEADERS = {
    "User-Agent": "Multimodal-Coherence-AI/1.0 (academic project; polite crawler)"
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Min pixel count (width * height) to reject tiny icons
MIN_PIXELS = 200_000
# Aspect ratio bounds to reject extreme ratios
MIN_ASPECT = 0.4
MAX_ASPECT = 4.0

# Non-photographic content patterns to filter out
SKIP_PATTERNS = re.compile(
    r"(painting|diagram|logo|icon|map|stamp|coat.of.arms|flag|cartoon|"
    r"illustration|drawing|sketch|engraving|lithograph|woodcut|fresco|"
    r"sculpture|statue|medal|coin|heraldic|emblem|sigil|crest|svg|"
    r"restored|NARA|Wellcome|artwork|"
    r"insect|butterfly|moth|beetle|specimen|"
    r"manuscript|document|poster|"
    r"portrait|microscop|"
    r"\.webm|\.ogv)",
    re.IGNORECASE,
)

# Scene-specific categories with domain tags (no catch-all categories)
CATEGORIES = [
    # Nature
    ("Category:Featured_pictures_of_landscapes", "nature"),
    ("Category:Featured_pictures_of_mountains", "nature"),
    ("Category:Featured_pictures_of_forests", "nature"),
    ("Category:Featured_pictures_of_sunsets", "nature"),
    ("Category:Featured_pictures_of_fields", "nature"),
    # Urban
    ("Category:Featured_pictures_of_cities", "urban"),
    ("Category:Featured_pictures_of_buildings", "urban"),
    ("Category:Featured_pictures_of_bridges", "urban"),
    ("Category:Featured_pictures_of_streets", "urban"),
    # Water
    ("Category:Featured_pictures_of_beaches", "water"),
    ("Category:Featured_pictures_of_oceans_and_seas", "water"),
    ("Category:Featured_pictures_of_rivers", "water"),
    ("Category:Featured_pictures_of_lakes", "water"),
    ("Category:Featured_pictures_of_waterfalls", "water"),
]

IMG_DIR.mkdir(parents=True, exist_ok=True)


def _is_photographic(title: str, caption: str) -> bool:
    """Filter out non-photographic content based on title and caption."""
    text = f"{title} {caption}"
    return not SKIP_PATTERNS.search(text)


def _validate_image_dimensions(img_bytes: bytes) -> bool:
    """Check that image meets minimum size and aspect ratio requirements."""
    try:
        from PIL import Image
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size
        pixels = w * h
        if pixels < MIN_PIXELS:
            return False
        aspect = w / h
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            return False
        return True
    except Exception:
        return False


def fetch_subcategories(category: str) -> list[str]:
    """Fetch one level of subcategories for a given category."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "subcat",
        "cmlimit": 50,
        "format": "json",
        "origin": "*",
    }
    subcats = []
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(BASE_SLEEP * 4)
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        for member in data.get("query", {}).get("categorymembers", []):
            subcats.append(member["title"])
        time.sleep(BASE_SLEEP)
    except requests.RequestException as e:
        print(f"  Subcategory fetch failed for {category}: {e}")
    return subcats


def fetch_category(category: str, domain: str, existing_names: set, target: int) -> list:
    """Fetch images from a single Wikimedia category."""
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|thumburl",
        "iiurlwidth": 1024,  # request 1024px thumbnails to avoid rate limits
        "format": "json",
        "origin": "*",
    }

    results = []
    continue_token = {}

    while len(results) < target:
        merged = params | continue_token
        try:
            r = requests.get(API_URL, params=merged, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"  Request failed: {e}")
            break

        if r.status_code == 429:
            wait = BASE_SLEEP * 4
            print(f"  Rate limited (429). Sleeping {wait:.1f}s...")
            time.sleep(wait)
            continue

        r.raise_for_status()
        data = r.json()

        if "query" not in data:
            break

        pages = data["query"]["pages"]

        for page in pages.values():
            if len(results) >= target:
                break

            imageinfo = page.get("imageinfo", [])
            if not imageinfo:
                continue

            url = imageinfo[0]["url"]
            parsed = urlparse(url)
            ext = Path(parsed.path).suffix.lower()

            if ext not in ALLOWED_EXTENSIONS:
                continue

            metadata = imageinfo[0].get("extmetadata", {})
            caption = (
                metadata.get("ImageDescription", {}).get("value")
                or page["title"]
            )
            title = page.get("title", "")

            # Filter non-photographic content
            if not _is_photographic(title, caption):
                continue

            filename = unquote(Path(parsed.path).name)

            # Deduplicate across categories
            if filename in existing_names:
                continue

            # Prefer thumbnail URL to avoid rate limiting on full-res images
            thumb_url = imageinfo[0].get("thumburl", url)

            results.append({
                "url": thumb_url,
                "filename": filename,
                "caption": caption,
                "domain": domain,
            })
            existing_names.add(filename)

        time.sleep(BASE_SLEEP)

        if "continue" in data:
            continue_token = data["continue"]
        else:
            break

    return results


def fetch_category_with_subcategories(
    category: str, domain: str, existing_names: set, target: int
) -> list:
    """Fetch images from a category and its subcategories (one level deep)."""
    # First, fetch directly from the category
    results = fetch_category(category, domain, existing_names, target)

    if len(results) >= target:
        return results

    # If we didn't hit the target, recurse into subcategories
    subcats = fetch_subcategories(category)
    for subcat in subcats:
        if len(results) >= target:
            break
        remaining = target - len(results)
        print(f"    Subcategory: {subcat} (need {remaining} more)")
        sub_results = fetch_category(subcat, domain, existing_names, remaining)
        results.extend(sub_results)

    return results


def download_images(items: list) -> list:
    """Download images, validate dimensions, and return samples list."""
    samples = []
    for idx, item in enumerate(items):
        local_name = f"wm_{idx+1:03d}_{item['domain']}_{item['filename']}"
        img_path = IMG_DIR / local_name

        # Skip already downloaded
        if img_path.exists():
            samples.append({
                "id": f"wm_{idx+1:03d}",
                "caption": item["caption"],
                "image": f"images/{local_name}",
                "domain": item["domain"],
                "audio": None,
            })
            continue

        print(f"  [{idx+1}/{len(items)}] Downloading {local_name[:60]}...")

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(item["url"], headers=HEADERS, timeout=60)
                if resp.status_code == 429:
                    wait = BASE_SLEEP * (2 ** attempt)  # exponential: 4, 8, 16, 32...
                    print(f"    Rate limited (429). Sleeping {wait:.0f}s... (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()

                # Validate image dimensions before saving
                if not _validate_image_dimensions(resp.content):
                    print(f"    Rejected (bad dimensions): {local_name[:50]}")
                    break

                img_path.write_bytes(resp.content)
                success = True
                break
            except Exception as e:
                print(f"    Attempt {attempt} failed: {e}")
                time.sleep(BASE_SLEEP * attempt)

        if not success:
            print(f"    Skipping {local_name} after retries")
            continue

        samples.append({
            "id": f"wm_{idx+1:03d}",
            "caption": item["caption"],
            "image": f"images/{local_name}",
            "domain": item["domain"],
            "audio": None,
        })
        time.sleep(BASE_SLEEP * 2)  # extra polite between downloads

    return samples


def main():
    total_target = sum(DOMAIN_TARGETS.values())
    print("=" * 70)
    print("Wikimedia Scene-Focused Dataset Builder")
    print(f"Per-domain targets: {DOMAIN_TARGETS}")
    print(f"Total target: {total_target} images across {len(CATEGORIES)} categories")
    print("=" * 70)

    # Group categories by domain
    domain_categories: dict[str, list[str]] = {}
    for category, domain in CATEGORIES:
        domain_categories.setdefault(domain, []).append(category)

    all_items = []
    existing_names = set()

    for domain, target in DOMAIN_TARGETS.items():
        categories = domain_categories.get(domain, [])
        if not categories:
            print(f"\nNo categories for domain '{domain}', skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Domain: {domain} (target={target})")
        print(f"{'='*50}")

        domain_items = []
        per_category = max(target // len(categories), 5)

        for category in categories:
            if len(domain_items) >= target:
                break

            remaining = target - len(domain_items)
            cat_target = min(per_category + 5, remaining)  # slight overshoot per cat
            print(f"\n  Category: {category} (target={cat_target})")

            items = fetch_category_with_subcategories(
                category, domain, existing_names, cat_target
            )
            print(f"    Found {len(items)} images")
            domain_items.extend(items)

        all_items.extend(domain_items[:target])
        print(f"  Domain '{domain}': {min(len(domain_items), target)} candidates")

    print(f"\nTotal candidates: {len(all_items)}")

    # Download
    print("\nDownloading images...")
    samples = download_images(all_items)

    # Save metadata
    OUT_FILE.write_text(json.dumps(samples, indent=2), encoding="utf-8")

    # Summary
    domain_counts: dict[str, int] = {}
    for s in samples:
        d = s.get("domain", "other")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print(f"\nDONE: {len(samples)} images saved")
    print(f"Metadata: {OUT_FILE}")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
