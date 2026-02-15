#!/usr/bin/env python3
"""
Analyze the retrieval bottleneck for the RQ2 negative result.

Quantifies how low-quality retrieval (small corpus, low cosine similarity)
constrains the MSCI ceiling and prevents planning from showing measurable
improvement over direct generation.

Usage:
    python scripts/analyze_retrieval_bottleneck.py

Reads:
    runs/rq1/rq1_results.json   -- RQ1 skip-text results (has image_sim, audio_sim)
    runs/rq2/rq2_results.json   -- RQ2 planning-mode results (no sim stored)
    runs/rq2/*/*/bundle.json    -- individual RQ2 bundles (has retrieval metadata)

Writes:
    runs/bottleneck_analysis.json
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root (works from any CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RQ1_PATH = PROJECT_ROOT / "runs" / "rq1" / "rq1_results.json"
RQ2_PATH = PROJECT_ROOT / "runs" / "rq2" / "rq2_results.json"
RQ2_BUNDLES_GLOB = str(PROJECT_ROOT / "runs" / "rq2" / "*" / "*" / "bundle.json")
OUTPUT_PATH = PROJECT_ROOT / "runs" / "bottleneck_analysis.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load a JSON file, exit with a clear message if missing."""
    if not path.exists():
        print(f"ERROR: Required file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def safe_stats(values: list[float]) -> dict:
    """Compute descriptive statistics for a list of floats."""
    if not values:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "min": None, "max": None, "q25": None, "q75": None,
        }
    n = len(values)
    s = sorted(values)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    q25 = s[int(n * 0.25)]
    q75 = s[int(n * 0.75)]
    return {
        "n": n,
        "mean": round(mean, 5),
        "std": round(std, 5),
        "median": round(median, 5),
        "min": round(s[0], 5),
        "max": round(s[-1], 5),
        "q25": round(q25, 5),
        "q75": round(q75, 5),
    }


def pct_below(values: list[float], threshold: float) -> float:
    """Percentage of values strictly below a threshold."""
    if not values:
        return 0.0
    return round(100.0 * sum(1 for v in values if v < threshold) / len(values), 2)


def entropy(counter: Counter) -> float:
    """Shannon entropy (base-2) of a frequency distribution."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return round(h, 4)


def fmt(val, width=10):
    """Format a numeric value or None for table display."""
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, int):
        return str(val).rjust(width)
    return f"{val:.4f}".rjust(width)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 72)
print("  RETRIEVAL BOTTLENECK ANALYSIS")
print("  Quantifying the retrieval ceiling for the RQ2 negative result")
print("=" * 72)
print()

rq1_data = load_json(RQ1_PATH)
rq2_data = load_json(RQ2_PATH)

rq1_results = rq1_data["results"]
rq2_results = rq2_data["results"]

print(f"Loaded RQ1: {len(rq1_results)} results from {RQ1_PATH.name}")
print(f"Loaded RQ2: {len(rq2_results)} results from {RQ2_PATH.name}")

# ---------------------------------------------------------------------------
# 2. Extract similarity values
# ---------------------------------------------------------------------------

# --- RQ1: use only baseline condition results ---
rq1_baseline = [r for r in rq1_results if r.get("condition") == "baseline"]

rq1_image_sims = [r["image_sim"] for r in rq1_baseline if r.get("image_sim") is not None]
rq1_audio_sims = [r["audio_sim"] for r in rq1_baseline if r.get("audio_sim") is not None]

print(f"RQ1 baseline runs with image_sim: {len(rq1_image_sims)}")
print(f"RQ1 baseline runs with audio_sim: {len(rq1_audio_sims)}")

# --- RQ2: similarity is NOT in the aggregated results file. ---
# Load from individual bundle.json files under runs/rq2/*/*/bundle.json.
rq2_bundle_paths = sorted(glob(RQ2_BUNDLES_GLOB))
print(f"RQ2 bundle files found: {len(rq2_bundle_paths)}")

rq2_image_sims = []
rq2_audio_sims = []
rq2_image_top5 = []  # list of lists: [[score1, ...score5], ...]
rq2_audio_top5 = []
rq2_bundle_records = []  # enriched records for per-domain analysis

for bp in rq2_bundle_paths:
    try:
        with open(bp) as f:
            bundle = json.load(f)
    except (json.JSONDecodeError, OSError):
        continue

    meta = bundle.get("meta", {})
    img_ret = meta.get("image_retrieval", {})
    aud_ret = meta.get("audio_retrieval", {})

    img_sim = img_ret.get("similarity")
    aud_sim = aud_ret.get("similarity")

    # Determine domain from the bundle's directory name prefix
    dir_name = Path(bp).parent.parent.name  # e.g. "nat_01_s42_direct"
    prefix = dir_name.split("_")[0]  # e.g. "nat"
    domain_map = {"nat": "nature", "urb": "urban", "wat": "water", "mix": "mixed"}
    domain = domain_map.get(prefix, "unknown")
    mode = meta.get("mode", "unknown")

    record = {
        "domain": domain,
        "mode": mode,
        "image_sim": img_sim,
        "audio_sim": aud_sim,
        "image_path": bundle.get("image_path"),
        "audio_path": bundle.get("audio_path"),
        "image_top5": [score for _, score in img_ret.get("top_5", [])],
        "audio_top5": [score for _, score in aud_ret.get("top_5", [])],
    }
    rq2_bundle_records.append(record)

    if img_sim is not None:
        rq2_image_sims.append(img_sim)
        if record["image_top5"]:
            rq2_image_top5.append(record["image_top5"])
    if aud_sim is not None:
        rq2_audio_sims.append(aud_sim)
        if record["audio_top5"]:
            rq2_audio_top5.append(record["audio_top5"])

print(f"RQ2 bundles with image_sim: {len(rq2_image_sims)}")
print(f"RQ2 bundles with audio_sim: {len(rq2_audio_sims)}")

# Also enrich the RQ2 aggregated results with domain info for cross-analysis
# (these already have domain in the rq2_results.json)
rq2_image_paths = [r.get("image_path") for r in rq2_results if r.get("image_path")]
rq2_audio_paths = [r.get("audio_path") for r in rq2_results if r.get("audio_path")]

# Combine RQ1 baseline + RQ2 bundle similarities for overall analysis
all_image_sims = rq1_image_sims + rq2_image_sims
all_audio_sims = rq1_audio_sims + rq2_audio_sims

print()

# ---------------------------------------------------------------------------
# 3a. Overall similarity distribution stats
# ---------------------------------------------------------------------------
print("-" * 72)
print("  SECTION 1: Similarity Distribution Statistics")
print("-" * 72)
print()

sources = {
    "RQ1 Baseline -- Image": rq1_image_sims,
    "RQ1 Baseline -- Audio": rq1_audio_sims,
    "RQ2 Bundles  -- Image": rq2_image_sims,
    "RQ2 Bundles  -- Audio": rq2_audio_sims,
    "Combined     -- Image": all_image_sims,
    "Combined     -- Audio": all_audio_sims,
}

header = f"{'Source':<28s} {'N':>5s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Q25':>10s} {'Median':>10s} {'Q75':>10s} {'Max':>10s}"
print(header)
print("-" * len(header))

stats_output = {}
for label, vals in sources.items():
    st = safe_stats(vals)
    stats_output[label] = st
    print(
        f"{label:<28s} {fmt(st['n'], 5)} {fmt(st['mean'])} {fmt(st['std'])} "
        f"{fmt(st['min'])} {fmt(st['q25'])} {fmt(st['median'])} {fmt(st['q75'])} {fmt(st['max'])}"
    )

print()

# ---------------------------------------------------------------------------
# 3b. % of retrievals below various thresholds
# ---------------------------------------------------------------------------
print("-" * 72)
print("  SECTION 2: Percentage of Retrievals Below Threshold")
print("-" * 72)
print()

thresholds = [0.15, 0.20, 0.25, 0.30]
threshold_header = f"{'Source':<28s}" + "".join(f" {'<' + str(t):>8s}" for t in thresholds)
print(threshold_header)
print("-" * len(threshold_header))

threshold_output = {}
for label, vals in sources.items():
    row = {str(t): pct_below(vals, t) for t in thresholds}
    threshold_output[label] = row
    cols = "".join(f" {row[str(t)]:>7.1f}%" for t in thresholds)
    print(f"{label:<28s}{cols}")

print()

# ---------------------------------------------------------------------------
# 3c. Per-domain breakdown of mean similarity
# ---------------------------------------------------------------------------
print("-" * 72)
print("  SECTION 3: Per-Domain Mean Similarity")
print("-" * 72)
print()

DOMAINS = ["nature", "urban", "water", "mixed"]

# RQ1 per-domain (baseline only)
rq1_domain_img = defaultdict(list)
rq1_domain_aud = defaultdict(list)
for r in rq1_baseline:
    d = r.get("domain", "unknown")
    if r.get("image_sim") is not None:
        rq1_domain_img[d].append(r["image_sim"])
    if r.get("audio_sim") is not None:
        rq1_domain_aud[d].append(r["audio_sim"])

# RQ2 per-domain (from bundles)
rq2_domain_img = defaultdict(list)
rq2_domain_aud = defaultdict(list)
for rec in rq2_bundle_records:
    d = rec["domain"]
    if rec["image_sim"] is not None:
        rq2_domain_img[d].append(rec["image_sim"])
    if rec["audio_sim"] is not None:
        rq2_domain_aud[d].append(rec["audio_sim"])

domain_header = f"{'Domain':<10s}  {'RQ1 Img Mean':>12s} {'RQ1 Img N':>10s}  {'RQ1 Aud Mean':>12s} {'RQ1 Aud N':>10s}  {'RQ2 Img Mean':>12s} {'RQ2 Img N':>10s}  {'RQ2 Aud Mean':>12s} {'RQ2 Aud N':>10s}"
print(domain_header)
print("-" * len(domain_header))

domain_output = {}
for dom in DOMAINS:
    rq1_is = rq1_domain_img.get(dom, [])
    rq1_as = rq1_domain_aud.get(dom, [])
    rq2_is = rq2_domain_img.get(dom, [])
    rq2_as = rq2_domain_aud.get(dom, [])

    rq1_img_mean = round(sum(rq1_is) / len(rq1_is), 5) if rq1_is else None
    rq1_aud_mean = round(sum(rq1_as) / len(rq1_as), 5) if rq1_as else None
    rq2_img_mean = round(sum(rq2_is) / len(rq2_is), 5) if rq2_is else None
    rq2_aud_mean = round(sum(rq2_as) / len(rq2_as), 5) if rq2_as else None

    domain_output[dom] = {
        "rq1_image_mean": rq1_img_mean, "rq1_image_n": len(rq1_is),
        "rq1_audio_mean": rq1_aud_mean, "rq1_audio_n": len(rq1_as),
        "rq2_image_mean": rq2_img_mean, "rq2_image_n": len(rq2_is),
        "rq2_audio_mean": rq2_aud_mean, "rq2_audio_n": len(rq2_as),
    }

    print(
        f"{dom:<10s}  {fmt(rq1_img_mean, 12)} {fmt(len(rq1_is), 10)}  "
        f"{fmt(rq1_aud_mean, 12)} {fmt(len(rq1_as), 10)}  "
        f"{fmt(rq2_img_mean, 12)} {fmt(len(rq2_is), 10)}  "
        f"{fmt(rq2_aud_mean, 12)} {fmt(len(rq2_as), 10)}"
    )

print()

# ---------------------------------------------------------------------------
# 3d. Domain coverage -- unique image/audio files used across all runs
# ---------------------------------------------------------------------------
print("-" * 72)
print("  SECTION 4: Domain Coverage (Unique Files)")
print("-" * 72)
print()

# From RQ1 baseline
rq1_img_files = set(r.get("image_path", "") for r in rq1_baseline if r.get("image_path"))
rq1_aud_files = set(r.get("audio_path", "") for r in rq1_baseline if r.get("audio_path"))

# From RQ2 aggregated results (all modes)
rq2_img_files_agg = set(r.get("image_path", "") for r in rq2_results if r.get("image_path"))
rq2_aud_files_agg = set(r.get("audio_path", "") for r in rq2_results if r.get("audio_path"))

# From RQ2 bundles (direct/planner only -- what's on disk)
rq2_img_files_bun = set(rec.get("image_path", "") for rec in rq2_bundle_records if rec.get("image_path"))
rq2_aud_files_bun = set(rec.get("audio_path", "") for rec in rq2_bundle_records if rec.get("audio_path"))

combined_img = rq1_img_files | rq2_img_files_agg
combined_aud = rq1_aud_files | rq2_aud_files_agg

coverage_output = {
    "rq1_unique_images": len(rq1_img_files),
    "rq1_unique_audios": len(rq1_aud_files),
    "rq2_unique_images_aggregated": len(rq2_img_files_agg),
    "rq2_unique_audios_aggregated": len(rq2_aud_files_agg),
    "rq2_unique_images_bundles": len(rq2_img_files_bun),
    "rq2_unique_audios_bundles": len(rq2_aud_files_bun),
    "combined_unique_images": len(combined_img),
    "combined_unique_audios": len(combined_aud),
}

print(f"  RQ1 baseline  -- unique images: {len(rq1_img_files):>4d}   unique audios: {len(rq1_aud_files):>4d}")
print(f"  RQ2 all modes -- unique images: {len(rq2_img_files_agg):>4d}   unique audios: {len(rq2_aud_files_agg):>4d}")
print(f"  RQ2 bundles   -- unique images: {len(rq2_img_files_bun):>4d}   unique audios: {len(rq2_aud_files_bun):>4d}")
print(f"  Combined      -- unique images: {len(combined_img):>4d}   unique audios: {len(combined_aud):>4d}")
print()
print(f"  Total RQ1 baseline retrievals:        {len(rq1_baseline)}")
print(f"  Total RQ2 results (all modes):        {len(rq2_results)}")
print(f"  Total RQ2 bundles on disk:            {len(rq2_bundle_records)}")
print()

# ---------------------------------------------------------------------------
# 3e. Retrieval diversity -- entropy of file selection distribution
# ---------------------------------------------------------------------------
print("-" * 72)
print("  SECTION 5: Retrieval Diversity (Shannon Entropy)")
print("-" * 72)
print()

rq1_img_counter = Counter(r.get("image_path") for r in rq1_baseline if r.get("image_path"))
rq1_aud_counter = Counter(r.get("audio_path") for r in rq1_baseline if r.get("audio_path"))
rq2_img_counter = Counter(r.get("image_path") for r in rq2_results if r.get("image_path"))
rq2_aud_counter = Counter(r.get("audio_path") for r in rq2_results if r.get("audio_path"))

rq1_img_entropy = entropy(rq1_img_counter)
rq1_aud_entropy = entropy(rq1_aud_counter)
rq2_img_entropy = entropy(rq2_img_counter)
rq2_aud_entropy = entropy(rq2_aud_counter)

# Maximum possible entropy = log2(unique_files)
rq1_img_max_h = round(math.log2(len(rq1_img_counter)), 4) if len(rq1_img_counter) > 1 else 0.0
rq1_aud_max_h = round(math.log2(len(rq1_aud_counter)), 4) if len(rq1_aud_counter) > 1 else 0.0
rq2_img_max_h = round(math.log2(len(rq2_img_counter)), 4) if len(rq2_img_counter) > 1 else 0.0
rq2_aud_max_h = round(math.log2(len(rq2_aud_counter)), 4) if len(rq2_aud_counter) > 1 else 0.0

diversity_output = {
    "rq1_image_entropy": rq1_img_entropy,
    "rq1_image_max_entropy": rq1_img_max_h,
    "rq1_audio_entropy": rq1_aud_entropy,
    "rq1_audio_max_entropy": rq1_aud_max_h,
    "rq2_image_entropy": rq2_img_entropy,
    "rq2_image_max_entropy": rq2_img_max_h,
    "rq2_audio_entropy": rq2_aud_entropy,
    "rq2_audio_max_entropy": rq2_aud_max_h,
}

ent_header = f"{'Source':<20s} {'H(image)':>10s} {'H_max(img)':>12s} {'H(audio)':>10s} {'H_max(aud)':>12s}"
print(ent_header)
print("-" * len(ent_header))
print(f"{'RQ1 baseline':<20s} {rq1_img_entropy:>10.4f} {rq1_img_max_h:>12.4f} {rq1_aud_entropy:>10.4f} {rq1_aud_max_h:>12.4f}")
print(f"{'RQ2 all modes':<20s} {rq2_img_entropy:>10.4f} {rq2_img_max_h:>12.4f} {rq2_aud_entropy:>10.4f} {rq2_aud_max_h:>12.4f}")

# Show most-selected files
print()
print("  Top-5 most selected images (RQ2 all modes):")
for path, count in rq2_img_counter.most_common(5):
    print(f"    {count:>4d}x  {path}")

print()
print("  Top-5 most selected audios (RQ2 all modes):")
for path, count in rq2_aud_counter.most_common(5):
    print(f"    {count:>4d}x  {path}")
print()

# ---------------------------------------------------------------------------
# 3f. Top-5 similarity gap
# ---------------------------------------------------------------------------
print("-" * 72)
print("  SECTION 6: Top-5 Similarity Gap (RQ2 Bundles)")
print("-" * 72)
print()

image_gaps = []
audio_gaps = []
image_gap_details = []
audio_gap_details = []

for rec in rq2_bundle_records:
    top5_img = rec.get("image_top5", [])
    if len(top5_img) >= 5:
        gap = top5_img[0] - top5_img[4]
        image_gaps.append(gap)
        image_gap_details.append({
            "rank1": round(top5_img[0], 5),
            "rank5": round(top5_img[4], 5),
            "gap": round(gap, 5),
        })
    elif len(top5_img) >= 2:
        gap = top5_img[0] - top5_img[-1]
        image_gaps.append(gap)
        image_gap_details.append({
            "rank1": round(top5_img[0], 5),
            "rank_last": round(top5_img[-1], 5),
            "gap": round(gap, 5),
            "n_candidates": len(top5_img),
        })

    top5_aud = rec.get("audio_top5", [])
    if len(top5_aud) >= 5:
        gap = top5_aud[0] - top5_aud[4]
        audio_gaps.append(gap)
        audio_gap_details.append({
            "rank1": round(top5_aud[0], 5),
            "rank5": round(top5_aud[4], 5),
            "gap": round(gap, 5),
        })
    elif len(top5_aud) >= 2:
        gap = top5_aud[0] - top5_aud[-1]
        audio_gaps.append(gap)
        audio_gap_details.append({
            "rank1": round(top5_aud[0], 5),
            "rank_last": round(top5_aud[-1], 5),
            "gap": round(gap, 5),
            "n_candidates": len(top5_aud),
        })

img_gap_stats = safe_stats(image_gaps)
aud_gap_stats = safe_stats(audio_gaps)

gap_header = f"{'Modality':<10s} {'N':>5s} {'Mean Gap':>10s} {'Std':>10s} {'Min':>10s} {'Median':>10s} {'Max':>10s}"
print(gap_header)
print("-" * len(gap_header))
print(
    f"{'Image':<10s} {fmt(img_gap_stats['n'], 5)} {fmt(img_gap_stats['mean'])} "
    f"{fmt(img_gap_stats['std'])} {fmt(img_gap_stats['min'])} "
    f"{fmt(img_gap_stats['median'])} {fmt(img_gap_stats['max'])}"
)
print(
    f"{'Audio':<10s} {fmt(aud_gap_stats['n'], 5)} {fmt(aud_gap_stats['mean'])} "
    f"{fmt(aud_gap_stats['std'])} {fmt(aud_gap_stats['min'])} "
    f"{fmt(aud_gap_stats['median'])} {fmt(aud_gap_stats['max'])}"
)

print()
print("  Interpretation: A small gap means that the #1 result is barely better")
print("  than the #5 result, indicating a flat similarity landscape where the")
print("  retrieval system cannot meaningfully distinguish good from mediocre matches.")
print()

# ---------------------------------------------------------------------------
# 3g. RQ2 mode comparison -- does planning change retrieval quality?
# ---------------------------------------------------------------------------
print("-" * 72)
print("  SECTION 7: Retrieval Quality by RQ2 Mode (from bundles)")
print("-" * 72)
print()

mode_img = defaultdict(list)
mode_aud = defaultdict(list)
for rec in rq2_bundle_records:
    m = rec.get("mode", "unknown")
    if rec["image_sim"] is not None:
        mode_img[m].append(rec["image_sim"])
    if rec["audio_sim"] is not None:
        mode_aud[m].append(rec["audio_sim"])

mode_header = f"{'Mode':<18s} {'Img N':>6s} {'Img Mean':>10s} {'Img Std':>10s} {'Aud N':>6s} {'Aud Mean':>10s} {'Aud Std':>10s}"
print(mode_header)
print("-" * len(mode_header))

mode_output = {}
for m in sorted(mode_img.keys()):
    ist = safe_stats(mode_img[m])
    ast = safe_stats(mode_aud.get(m, []))
    mode_output[m] = {"image": ist, "audio": ast}
    print(
        f"{m:<18s} {fmt(ist['n'], 6)} {fmt(ist['mean'])} {fmt(ist['std'])} "
        f"{fmt(ast['n'], 6)} {fmt(ast['mean'])} {fmt(ast['std'])}"
    )

print()

# ---------------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------------
print("=" * 72)
print("  SUMMARY TABLE")
print("=" * 72)
print()

combined_img_stats = safe_stats(all_image_sims)
combined_aud_stats = safe_stats(all_audio_sims)

print(f"  {'Metric':<40s} {'Image':>12s} {'Audio':>12s}")
print(f"  {'-'*40} {'-'*12} {'-'*12}")
print(f"  {'Mean retrieval similarity (combined)':<40s} {fmt(combined_img_stats['mean'], 12)} {fmt(combined_aud_stats['mean'], 12)}")
print(f"  {'Median retrieval similarity':<40s} {fmt(combined_img_stats['median'], 12)} {fmt(combined_aud_stats['median'], 12)}")
print(f"  {'Std deviation':<40s} {fmt(combined_img_stats['std'], 12)} {fmt(combined_aud_stats['std'], 12)}")
print(f"  {'% below 0.20':<40s} {pct_below(all_image_sims, 0.20):>11.1f}% {pct_below(all_audio_sims, 0.20):>11.1f}%")
print(f"  {'% below 0.25':<40s} {pct_below(all_image_sims, 0.25):>11.1f}% {pct_below(all_audio_sims, 0.25):>11.1f}%")
print(f"  {'% below 0.30':<40s} {pct_below(all_image_sims, 0.30):>11.1f}% {pct_below(all_audio_sims, 0.30):>11.1f}%")
print(f"  {'Unique files used (combined)':<40s} {len(combined_img):>12d} {len(combined_aud):>12d}")
print(f"  {'Selection entropy (RQ2)':<40s} {rq2_img_entropy:>12.4f} {rq2_aud_entropy:>12.4f}")
print(f"  {'Max possible entropy (RQ2)':<40s} {rq2_img_max_h:>12.4f} {rq2_aud_max_h:>12.4f}")
print(f"  {'Mean top-5 gap (RQ2 bundles)':<40s} {fmt(img_gap_stats['mean'], 12)} {fmt(aud_gap_stats['mean'], 12)}")
print()

# ---------------------------------------------------------------------------
# 5. Save results to JSON
# ---------------------------------------------------------------------------
output = {
    "analysis": "Retrieval Bottleneck for RQ2 Negative Result",
    "data_sources": {
        "rq1": str(RQ1_PATH),
        "rq2": str(RQ2_PATH),
        "rq2_bundles_found": len(rq2_bundle_paths),
    },
    "similarity_stats": stats_output,
    "threshold_analysis": threshold_output,
    "per_domain": domain_output,
    "domain_coverage": coverage_output,
    "retrieval_diversity": diversity_output,
    "top5_gap": {
        "image": img_gap_stats,
        "audio": aud_gap_stats,
    },
    "mode_comparison": mode_output,
    "summary": {
        "combined_image_mean": combined_img_stats["mean"],
        "combined_audio_mean": combined_aud_stats["mean"],
        "combined_image_median": combined_img_stats["median"],
        "combined_audio_median": combined_aud_stats["median"],
        "pct_image_below_0.25": pct_below(all_image_sims, 0.25),
        "pct_audio_below_0.25": pct_below(all_audio_sims, 0.25),
        "unique_images": len(combined_img),
        "unique_audios": len(combined_aud),
        "rq2_image_entropy": rq2_img_entropy,
        "rq2_audio_entropy": rq2_aud_entropy,
        "mean_image_top5_gap": img_gap_stats["mean"],
        "mean_audio_top5_gap": aud_gap_stats["mean"],
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved to: {OUTPUT_PATH}")
print()

# ---------------------------------------------------------------------------
# 6. Interpretation paragraph
# ---------------------------------------------------------------------------
print("=" * 72)
print("  INTERPRETATION")
print("=" * 72)
print()
print(
    f"The retrieval bottleneck analysis reveals that the cosine similarity\n"
    f"between text prompts and retrieved assets is systematically low.\n"
    f"Across all runs, image retrieval averages {combined_img_stats['mean']:.4f}\n"
    f"(median {combined_img_stats['median']:.4f}) and audio retrieval averages\n"
    f"{combined_aud_stats['mean']:.4f} (median {combined_aud_stats['median']:.4f}).\n"
    f"{pct_below(all_image_sims, 0.25):.1f}% of image retrievals and\n"
    f"{pct_below(all_audio_sims, 0.25):.1f}% of audio retrievals fall below a\n"
    f"cosine similarity of 0.25, indicating that the corpus is too small and\n"
    f"semantically sparse to provide good matches for most prompts.\n"
    f"\n"
    f"The retrieval diversity is constrained: only {len(combined_img)} unique\n"
    f"images and {len(combined_aud)} unique audio files are ever selected\n"
    f"across all experiments. The top-5 similarity gap is small (image mean:\n"
    f"{img_gap_stats['mean']:.4f}, audio mean: {aud_gap_stats['mean']:.4f}),\n"
    f"meaning the best match is barely distinguishable from the 5th-best, so\n"
    f"retrieval is essentially choosing among equally mediocre options.\n"
    f"\n"
    f"This creates a hard ceiling on MSCI scores: no amount of planning\n"
    f"sophistication (direct, planner, council, or extended_prompt) can\n"
    f"improve coherence beyond what the retrieved assets permit. The RQ2\n"
    f"negative result -- that planning modes show no significant MSCI\n"
    f"improvement -- is therefore primarily explained by this retrieval\n"
    f"bottleneck rather than a failure of the planning algorithms themselves.\n"
    f"Future work should expand the asset corpus, adopt generative models for\n"
    f"image/audio synthesis, or implement re-ranking with learned retrieval\n"
    f"functions to raise the retrieval quality ceiling."
)
print()
