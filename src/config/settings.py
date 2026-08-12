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

# RQ3 sample paths (single source of truth for all scripts)
RQ3_SAMPLES_PATH = RUNS_DIR / "rq3" / "rq3_samples.json"
RQ3_SAMPLES_EXTENDED_PATH = RUNS_DIR / "rq3" / "rq3_samples_extended.json"
RQ3_HUMAN_SCORES_PATH = RUNS_DIR / "rq3" / "rq3_human_scores.json"
RQ3_SESSIONS_DIR = RUNS_DIR / "rq3" / "sessions"

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

# ---------------------------------------------------------------------------
# cMSCI (Calibrated Multimodal Semantic Coherence Index)
# ---------------------------------------------------------------------------

# Calibration store (fitted from RQ1 baseline data)
CMSCI_CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "cmsci_calibration.json"

# Ex-MCR cross-space alignment (CLAP → CLIP projection)
EXMCR_WEIGHTS_PATH = PROJECT_ROOT / "models" / "exmcr" / "ex_clap.pt"

# Cross-Space Bridge (CLIP image + CLAP audio → shared 256-d bridge space)
BRIDGE_WEIGHTS_PATH = PROJECT_ROOT / "models" / "bridge" / "bridge_best.pt"

# Probabilistic adapters (ProbVLM-style uncertainty)
PROB_CLIP_ADAPTER_PATH = PROJECT_ROOT / "models" / "prob_adapters" / "clip_adapter.pt"
PROB_CLAP_ADAPTER_PATH = PROJECT_ROOT / "models" / "prob_adapters" / "clap_adapter.pt"

# Full pipeline optimized parameters (via LOO-CV on RQ3 human ratings)
# 100-sample config (5 raters): rho=0.789 (p<1e-6), LOO-CV rho=0.749 (p<1e-6)
# Previous 30-sample config: rho=0.519 (p=0.003), LOO-CV rho=0.425 (p=0.019)
CMSCI_MARGIN_ALPHA = 1             # Margin scaling factor (re-optimized on corrected 100-sample 8/5-rater ground truth, 2026-07)
CMSCI_CHANNEL_WEIGHT_TI = 0.15    # Text-image channel weight (1 - w for text-audio)
CMSCI_CALIBRATION_MODE = "cosine"  # "cosine" (z-norm cosine sims) or "gram" (z-norm gram coherences)

# Variant E: ExMCR cross-modal complementarity (w_3d=0 recovers D exactly)
# ExMCR projects CLAP audio → CLIP space; complementarity = Gramian dispersion
# High complementarity = image and audio contribute unique perspectives (rewarded)
CMSCI_W_3D = 0.15                  # Weight for z-normalized IA complementarity
# Variant F: ProbVLM adaptive channel weighting (gamma=0 recovers E exactly)
CMSCI_GAMMA = 0.6                 # Mixing ratio: w_final = (1-gamma)*base_w + gamma*adaptive_w

# Contrastive negative bank
CMSCI_NEGATIVE_K = 5                # Number of hard negatives per modality
CMSCI_NEGATIVE_BANK_ENABLED = True  # Enable/disable contrastive calibration

# MC sampling for uncertainty estimation
CMSCI_MC_SAMPLES = 100  # Number of Monte Carlo samples for Variant F

# Probabilistic adapter training
PROB_ADAPTER_EPOCHS = 100
PROB_ADAPTER_LR = 1e-4
PROB_ADAPTER_BATCH_SIZE = 32
PROB_ADAPTER_PATIENCE = 15

# ---------------------------------------------------------------------------
# Audio Quality / Silence Detection
# ---------------------------------------------------------------------------

AUDIO_FILTER_SILENCE = True             # Run silence detection before embedding
AUDIO_SILENCE_RMS_DB = -40.0            # RMS threshold (dB) for silence
AUDIO_SILENCE_FRACTION_THRESHOLD = 0.7  # Fraction of silent frames to flag
AUDIO_LOG_QUALITY_WARNINGS = True       # Log warnings for quality issues

# ---------------------------------------------------------------------------
# Windowed CLAP Embedding
# ---------------------------------------------------------------------------

AUDIO_USE_WINDOWED = False    # Paper default: single-clip CLAP (rho=0.519)
AUDIO_WINDOW_SEC = 2.0       # Window duration in seconds
AUDIO_HOP_SEC = 1.0          # Hop between windows in seconds
AUDIO_AGGREGATION = "max"    # Aggregation: "max" (element-wise) or "mean"

# ---------------------------------------------------------------------------
# Gemini Embedding 2 (unified text/image/audio space)
# ---------------------------------------------------------------------------

GEMINI_MODEL_ID = "gemini-embedding-2-preview"
GEMINI_OUTPUT_DIM = 3072
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Cache and index paths
GEMINI_CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings_gemini"
GEMINI_IMAGE_INDEX_PATH = DATA_DIR / "embeddings" / "gemini_image_index.npz"
GEMINI_AUDIO_INDEX_PATH = DATA_DIR / "embeddings" / "gemini_audio_index.npz"
GEMINI_CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "cmsci_v2_calibration.json"

# Matryoshka Representation Learning (MRL) truncation dimensions
MATRYOSHKA_DIMS = [768, 1536, 3072]

# ---------------------------------------------------------------------------
# cMSCI v2 (Gemini-backed calibrated coherence)
# ---------------------------------------------------------------------------

# Optimized via LOO-CV on dev set (20 samples)
# Full-sample rho=0.455 (p=0.044), LOO gap=0.001, 11/20 folds chose this config
CMSCI_V2_ALPHA = 20            # Margin scaling (contrastive useful with 5 raters)
CMSCI_V2_W_TI = 0.85          # Text-image channel weight (re-optimized 100-sample corrected ground truth, 2026-07)
CMSCI_V2_W_COMPL = 0.00       # Complementarity weight (not needed in unified space)
CMSCI_V2_GAMMA_MRL = 0.7      # Matryoshka adaptive mixing (re-optimized 100-sample corrected ground truth, 2026-07)
CMSCI_V2_CAL_MODE = "gram_2d" # "gram_2d" (2-way avg) or "gram_3d" (exact 3D)
CMSCI_V2_W_IA = 0.0           # IA channel weight (default 0: IA adds noise, std=0.011)
CMSCI_V2_NEGATIVE_K = 5       # Number of hard negatives per channel
CMSCI_V2_USE_MULTISCALE = False  # Multi-scale tested, no improvement (scales too similar)

# ---------------------------------------------------------------------------
# Gemini task type (affects embedding geometry)
# ---------------------------------------------------------------------------

GEMINI_TASK_TYPE = "SEMANTIC_SIMILARITY"  # or RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, CLASSIFICATION, CLUSTERING

# ---------------------------------------------------------------------------
# Ensemble (v1 + v2)
# ---------------------------------------------------------------------------

CMSCI_ENSEMBLE_W_V1 = 0.4     # v1 weight in ensemble (optimized via LOO-CV, 30/30 folds)
