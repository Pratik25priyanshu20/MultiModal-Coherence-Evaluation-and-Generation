"""
Centralized configuration for MultiModal Coherence AI.

All magic numbers, model names, paths, and thresholds live here.
Import from this module instead of hardcoding values in source files.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_INDEX_PATH = DATA_DIR / "embeddings" / "image_index.npz"
AUDIO_INDEX_PATH = DATA_DIR / "embeddings" / "audio_index.npz"
COHERENCE_STATS_PATH = PROJECT_ROOT / "artifacts" / "coherence_stats.json"

IMAGE_DIRS = [
    DATA_DIR / "processed" / "images",
    DATA_DIR / "wikimedia" / "images",
]
AUDIO_DIRS = [
    DATA_DIR / "processed" / "audio",
    DATA_DIR / "freesound" / "audio",
]

# Embedding cache
CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"

# Experiment output
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

CLIP_MODEL = "openai/clip-vit-base-patch32"
CLAP_MODEL = "laion/clap-htsat-unfused"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
HF_FALLBACK_MODEL = "gpt2"

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------

CLIP_DIM = 512
CLAP_DIM = 512
TARGET_DIM = 512

# ---------------------------------------------------------------------------
# MSCI weights
# ---------------------------------------------------------------------------

# These weights are hypothesized, not empirically derived.
# Text-image and text-audio are weighted equally; image-audio is down-weighted
# because CLIP and CLAP are different embedding spaces.
MSCI_WEIGHTS = {
    "st_i": 0.45,   # text-image (CLIP shared space)
    "st_a": 0.45,   # text-audio (CLAP shared space)
    "si_a": 0.10,   # image-audio (cross-space — usually omitted)
}

# ---------------------------------------------------------------------------
# Retrieval thresholds
# ---------------------------------------------------------------------------

IMAGE_MIN_SIMILARITY = 0.20
AUDIO_MIN_SIMILARITY = 0.10
IMAGE_LOW_SIMILARITY_WARN = 0.25

# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

TEXT_MAX_TOKENS = 160
TEXT_TEMPERATURE_DETERMINISTIC = 0.0
TEXT_TEMPERATURE_STOCHASTIC = 0.7
TEXT_TOP_P_DETERMINISTIC = 1.0
TEXT_TOP_P_STOCHASTIC = 0.9

# ---------------------------------------------------------------------------
# Audio generation (fallback ambient)
# ---------------------------------------------------------------------------

AUDIO_DURATION_SEC = 6.0
AUDIO_SAMPLE_RATE = 48000

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

DRIFT_ASYMMETRY_THRESHOLD = 0.15  # |st_i - st_a| gap to flag drift

# ---------------------------------------------------------------------------
# Human evaluation
# ---------------------------------------------------------------------------

RERATING_FRACTION = 0.20
KAPPA_ACCEPTABLE_THRESHOLD = 0.70
ALPHA_ACCEPTABLE_THRESHOLD = 0.667
