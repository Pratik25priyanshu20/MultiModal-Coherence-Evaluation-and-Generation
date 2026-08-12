"""
Dev/Test Split Infrastructure for cMSCI Evaluation.

Creates a stratified, reproducible dev/test split from human-rated
RQ3 samples. Stratification preserves the distribution of:
  - Domain (nature, urban, water, mixed)
  - Condition (baseline, wrong_image, wrong_audio)

The test set is LOCKED — all future hyperparameter optimization must use
only the dev set. The test set is for final reporting only.

Default split ratios:
  - 30 samples:  20 dev / 10 test  (test_fraction=1/3)
  - 100 samples: 70 dev / 30 test  (test_fraction=0.30)

Usage:
    from src.experiments.data_splits import get_dev_test_split
    dev_ids, test_ids = get_dev_test_split()
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.config.settings import (
    RQ3_SAMPLES_PATH, RQ3_SAMPLES_EXTENDED_PATH,
)

# Use extended samples if available, otherwise fall back to original
SAMPLES_PATH = RQ3_SAMPLES_EXTENDED_PATH if RQ3_SAMPLES_EXTENDED_PATH.exists() else RQ3_SAMPLES_PATH
SPLIT_PATH = PROJECT_ROOT / "artifacts" / "dev_test_split.json"

# Locked random seed for reproducibility — DO NOT CHANGE
_SPLIT_SEED = 2024


def _stratification_key(sample: dict) -> str:
    """Create stratification key from domain x condition."""
    domain = sample.get("domain", "unknown")
    condition = sample.get("condition", "unknown")
    return f"{domain}_{condition}"


def create_stratified_split(
    samples: list,
    test_fraction: float = 1 / 3,
    seed: int = _SPLIT_SEED,
) -> Tuple[List[str], List[str]]:
    """Create stratified dev/test split from sample list.

    Stratifies by domain x condition to preserve distribution.
    For strata with fewer than 3 samples, assigns 1 to test.

    Args:
        samples: List of sample dicts with 'sample_id', 'domain', 'condition'.
        test_fraction: Fraction of samples for test set (default 1/3 → 10/30).
        seed: Random seed for reproducibility.

    Returns:
        (dev_ids, test_ids): Lists of sample IDs for each split.
    """
    rng = np.random.default_rng(seed)

    # Group samples by stratification key
    strata: Dict[str, list] = {}
    for s in samples:
        key = _stratification_key(s)
        strata.setdefault(key, []).append(s["sample_id"])

    dev_ids = []
    test_ids = []

    for key in sorted(strata.keys()):
        ids = sorted(strata[key])
        rng.shuffle(ids)

        if len(ids) == 1:
            # Single-sample strata: assign to dev (larger set needs more data)
            dev_ids.extend(ids)
            continue

        n_test = max(1, round(len(ids) * test_fraction))
        # Cap test allocation for small strata
        if n_test >= len(ids):
            n_test = 1

        test_ids.extend(ids[:n_test])
        dev_ids.extend(ids[n_test:])

    # Sort for consistency
    dev_ids.sort()
    test_ids.sort()

    return dev_ids, test_ids


def save_split(dev_ids: List[str], test_ids: List[str], path: Path = SPLIT_PATH) -> None:
    """Save the split to a JSON file with a content hash for integrity."""
    content = {
        "description": "Stratified dev/test split for cMSCI optimization",
        "seed": _SPLIT_SEED,
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
        "dev_ids": dev_ids,
        "test_ids": test_ids,
    }
    # Add integrity hash
    hash_input = json.dumps({"dev": dev_ids, "test": test_ids}, sort_keys=True)
    content["sha256"] = hashlib.sha256(hash_input.encode()).hexdigest()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f, indent=2)


def load_split(path: Path = SPLIT_PATH) -> Tuple[List[str], List[str]]:
    """Load a previously saved split, verifying integrity."""
    with open(path) as f:
        data = json.load(f)

    dev_ids = data["dev_ids"]
    test_ids = data["test_ids"]

    # Verify integrity
    hash_input = json.dumps({"dev": dev_ids, "test": test_ids}, sort_keys=True)
    expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    if data.get("sha256") and data["sha256"] != expected_hash:
        raise ValueError("Split file integrity check failed — file may have been tampered with")

    return dev_ids, test_ids


def get_dev_test_split(force_recreate: bool = False) -> Tuple[List[str], List[str]]:
    """Get the canonical dev/test split, creating if needed.

    This is the main entry point. It loads from cache if available,
    otherwise creates and saves the split.

    Auto-detects sample count and uses appropriate test_fraction:
      - 30 samples:  test_fraction=1/3  → 20 dev / 10 test
      - 100 samples: test_fraction=0.30 → 70 dev / 30 test

    Args:
        force_recreate: If True, recreate even if cached split exists.

    Returns:
        (dev_ids, test_ids): Lists of sample IDs.
    """
    if SPLIT_PATH.exists() and not force_recreate:
        dev_ids, test_ids = load_split()
        # Check if split matches current samples file
        if SAMPLES_PATH.exists():
            with open(SAMPLES_PATH) as f:
                data = json.load(f)
            n_samples = len(data["samples"])
            n_split = len(dev_ids) + len(test_ids)
            if n_split != n_samples:
                # Sample count changed — recreate split
                import logging
                logging.getLogger(__name__).warning(
                    "Split has %d samples but samples file has %d — recreating",
                    n_split, n_samples,
                )
                return get_dev_test_split(force_recreate=True)
        return dev_ids, test_ids

    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"RQ3 samples not found: {SAMPLES_PATH}")

    with open(SAMPLES_PATH) as f:
        data = json.load(f)

    samples = data["samples"]
    # Choose test_fraction based on sample count
    if len(samples) >= 80:
        test_fraction = 0.30  # 70/30 for 100 samples
    else:
        test_fraction = 1 / 3  # 20/10 for 30 samples

    dev_ids, test_ids = create_stratified_split(samples, test_fraction=test_fraction)
    save_split(dev_ids, test_ids)
    return dev_ids, test_ids


def get_split_samples(samples: list) -> Tuple[list, list]:
    """Split a list of sample dicts into dev and test sets.

    Args:
        samples: List of sample dicts with 'sample_id' key.

    Returns:
        (dev_samples, test_samples): Lists of sample dicts.
    """
    dev_ids, test_ids = get_dev_test_split()
    dev_set = set(dev_ids)
    test_set = set(test_ids)

    dev_samples = [s for s in samples if s["sample_id"] in dev_set]
    test_samples = [s for s in samples if s["sample_id"] in test_set]

    return dev_samples, test_samples


def print_split_summary(samples: list) -> None:
    """Print detailed summary of the dev/test split."""
    dev_ids, test_ids = get_dev_test_split()
    dev_set = set(dev_ids)
    test_set = set(test_ids)

    print(f"Dev/Test Split Summary")
    print(f"  Seed: {_SPLIT_SEED}")
    print(f"  Dev:  {len(dev_ids)} samples")
    print(f"  Test: {len(test_ids)} samples")

    # Show stratification
    from collections import Counter
    dev_strata = Counter()
    test_strata = Counter()

    for s in samples:
        key = _stratification_key(s)
        if s["sample_id"] in dev_set:
            dev_strata[key] += 1
        elif s["sample_id"] in test_set:
            test_strata[key] += 1

    all_keys = sorted(set(list(dev_strata.keys()) + list(test_strata.keys())))
    print(f"\n  {'Stratum':<25s}  {'Dev':>4s}  {'Test':>4s}  {'Total':>5s}")
    print(f"  {'-'*25}  {'----':>4s}  {'----':>4s}  {'-----':>5s}")
    for key in all_keys:
        d = dev_strata.get(key, 0)
        t = test_strata.get(key, 0)
        print(f"  {key:<25s}  {d:4d}  {t:4d}  {d+t:5d}")

    print(f"  {'-'*25}  {'----':>4s}  {'----':>4s}  {'-----':>5s}")
    print(f"  {'TOTAL':<25s}  {len(dev_ids):4d}  {len(test_ids):4d}  {len(dev_ids)+len(test_ids):5d}")

    print(f"\n  Dev IDs:  {dev_ids}")
    print(f"  Test IDs: {test_ids}")
