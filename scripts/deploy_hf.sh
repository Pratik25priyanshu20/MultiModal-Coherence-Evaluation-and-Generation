#!/usr/bin/env bash
# ============================================================================
# Deploy Multimodal Coherence AI to Hugging Face Spaces
#
# Usage:
#   ./scripts/deploy_hf.sh <hf_username>
#
# Example:
#   ./scripts/deploy_hf.sh pratik-250620
#
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login  (authenticate with your HF token)
# ============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hf_username>"
    echo "Example: $0 pratik-250620"
    exit 1
fi

HF_USER="$1"
SPACE_NAME="MultiModal-Coherence-AI"
REPO_ID="${HF_USER}/${SPACE_NAME}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="/tmp/hf_deploy_${SPACE_NAME}"

echo "============================================"
echo "  Deploying to: ${REPO_ID}"
echo "  Project dir:  ${PROJECT_DIR}"
echo "  Staging dir:  ${DEPLOY_DIR}"
echo "============================================"

# ── Step 1: Stage files locally ──
echo ""
echo "[1/3] Staging project files..."

rm -rf "${DEPLOY_DIR}"
mkdir -p "${DEPLOY_DIR}"

# Config files from deploy/hf/
cp "${PROJECT_DIR}/deploy/hf/app.py"           "${DEPLOY_DIR}/app.py"
cp "${PROJECT_DIR}/deploy/hf/requirements.txt"  "${DEPLOY_DIR}/requirements.txt"
cp "${PROJECT_DIR}/deploy/hf/packages.txt"      "${DEPLOY_DIR}/packages.txt"
cp "${PROJECT_DIR}/deploy/hf/README.md"         "${DEPLOY_DIR}/README.md"

# Source code
cp -r "${PROJECT_DIR}/src" "${DEPLOY_DIR}/src"

# Data: embeddings (small, <1MB)
mkdir -p "${DEPLOY_DIR}/data/embeddings"
cp "${PROJECT_DIR}/data/embeddings/image_index.npz" "${DEPLOY_DIR}/data/embeddings/"
cp "${PROJECT_DIR}/data/embeddings/audio_index.npz" "${DEPLOY_DIR}/data/embeddings/"

# Data: images
mkdir -p "${DEPLOY_DIR}/data/processed/images"
cp "${PROJECT_DIR}"/data/processed/images/*.{png,jpg,jpeg,webp} "${DEPLOY_DIR}/data/processed/images/" 2>/dev/null || true

mkdir -p "${DEPLOY_DIR}/data/wikimedia/images"
cp "${PROJECT_DIR}"/data/wikimedia/images/*.{png,jpg,jpeg,webp} "${DEPLOY_DIR}/data/wikimedia/images/" 2>/dev/null || true

# Data: audio
mkdir -p "${DEPLOY_DIR}/data/processed/audio"
cp "${PROJECT_DIR}"/data/processed/audio/*.{wav,mp3,flac,ogg} "${DEPLOY_DIR}/data/processed/audio/" 2>/dev/null || true

mkdir -p "${DEPLOY_DIR}/data/freesound/audio"
cp "${PROJECT_DIR}"/data/freesound/audio/*.{wav,mp3,flac,ogg} "${DEPLOY_DIR}/data/freesound/audio/" 2>/dev/null || true

# Freesound metadata (for domain tagging)
if [ -f "${PROJECT_DIR}/data/freesound/samples.json" ]; then
    cp "${PROJECT_DIR}/data/freesound/samples.json" "${DEPLOY_DIR}/data/freesound/"
fi

# Artifacts (coherence_stats.json needed by AdaptiveThresholds)
mkdir -p "${DEPLOY_DIR}/artifacts"
cp "${PROJECT_DIR}/artifacts/coherence_stats.json" "${DEPLOY_DIR}/artifacts/"

# Remove __pycache__ and .cache
find "${DEPLOY_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${DEPLOY_DIR}" -type d -name ".cache" -exec rm -rf {} + 2>/dev/null || true

echo "  Files staged."
echo ""
echo "  Size summary:"
du -sh "${DEPLOY_DIR}/src" "${DEPLOY_DIR}/data" "${DEPLOY_DIR}/app.py" 2>/dev/null || true

# ── Step 2: Upload using huggingface-cli (handles Xet storage for binaries) ──
echo ""
echo "[2/3] Uploading to HF Space (this may take a few minutes)..."

huggingface-cli upload "${REPO_ID}" "${DEPLOY_DIR}" . --repo-type space

# ── Step 3: Done ──
echo ""
echo "============================================"
echo "  Deployed successfully!"
echo "  URL: https://huggingface.co/spaces/${REPO_ID}"
echo "============================================"
echo ""
echo "  The Space will take a few minutes to build."
echo "  First run will download CLIP + CLAP models (~1GB)."
echo ""
