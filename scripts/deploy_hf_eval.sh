#!/usr/bin/env bash
# Deploy the human evaluation app to Hugging Face Spaces.
#
# Usage:
#   ./scripts/deploy_hf_eval.sh <hf_username> [space_name]
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
#
# Example:
#   ./scripts/deploy_hf_eval.sh myuser multimodal-coherence-eval

set -euo pipefail

HF_USER="${1:?Usage: $0 <hf_username> [space_name]}"
SPACE_NAME="${2:-multimodal-coherence-eval}"
REPO_ID="${HF_USER}/${SPACE_NAME}"
DEPLOY_DIR="/tmp/hf_eval_deploy"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Stage files ==="
python3 "$PROJECT_ROOT/scripts/stage_hf_eval.py" --output "$DEPLOY_DIR"

echo ""
echo "=== Upload to HF Spaces ==="
echo "Repo: ${REPO_ID}"
echo "Type: space"
echo ""

huggingface-cli upload "$REPO_ID" "$DEPLOY_DIR" . --repo-type space

echo ""
echo "=== Done ==="
echo "Live URL: https://huggingface.co/spaces/${REPO_ID}"
echo ""
echo "Note: First build may take 1-2 minutes. Check build logs at:"
echo "  https://huggingface.co/spaces/${REPO_ID}/logs"
