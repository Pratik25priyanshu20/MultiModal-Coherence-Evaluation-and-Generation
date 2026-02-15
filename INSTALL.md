# Installation Guide

## Prerequisites

- **Python 3.10+**
- **Ollama** (local LLM backend): https://ollama.ai/
- ~4 GB disk space for models (CLIP, CLAP, Ollama)
- ~2 GB additional for AudioLDM (optional, for audio generation)
- ~6 GB additional for Stable Diffusion (optional, for image generation)

## Step-by-Step Setup

### 1. Clone and enter the repository

```bash
git clone <repo-url>
cd MultiModal-Coherence-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install the package

```bash
# Core install (editable mode)
pip install -e .

# With dev tools (pytest, linting)
pip install -e ".[dev]"

# With human evaluation UI (Streamlit)
pip install -e ".[eval]"
```

Or install from requirements.txt directly:

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env if you need to change defaults
```

**Optional API keys** (in `.env`):
- `FREESOUND_API_KEY` — Required only for downloading audio from Freesound (dataset expansion)

### 5. Install and start Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull qwen2:7b

# Start the server (if not auto-started)
ollama serve
```

### 6. Build embedding indexes

This embeds all images and audio files using CLIP and CLAP. Takes ~2 minutes on first run (models are downloaded automatically by HuggingFace).

```bash
python scripts/build_embedding_indexes.py
```

Expected output (after dataset expansion):
```
Image index: 57 items (5 nature, 23 urban, 9 water, 20 other)
Audio index: 104 items (22 nature, 28 urban, 33 water, 21 other)
```

The dataset includes images from Wikimedia Commons and audio from Freesound, expanded from the original core dataset (21 images, 60 audio).

### 7. Verify the installation

```bash
# Run the sanity check (skips text generation if Ollama is not running)
python scripts/sanity_check.py --skip-text

# Full sanity check (requires Ollama running)
python scripts/sanity_check.py
```

All 4 checks should pass:
- Retrieval quality
- Perturbation quality
- Audio differentiation
- MSCI sensitivity

### 8. Run tests

```bash
pytest tests/ -v
```

All 84 tests should pass.

---

## Running Experiments

```bash
# Single prompt test
python scripts/run_unified.py

# RQ1: MSCI sensitivity (270 runs)
python scripts/run_rq1.py

# RQ2: Planning ablation (360 runs, requires Ollama)
python scripts/run_rq2.py

# Parallel execution (faster on multi-core machines)
python scripts/run_rq1.py --parallel 4
python scripts/run_rq2.py --parallel 4

# Analyze results (with bootstrap confidence intervals)
python scripts/analyze_results.py runs/rq1/rq1_results.json
python scripts/analyze_results.py runs/rq2/rq2_results.json
```

## Human Evaluation (RQ3)

```bash
# Select 30 stratified samples
python scripts/select_rq3_samples.py

# Launch the evaluation interface (3+ raters recommended)
streamlit run app/human_eval_app.py

# After collecting evaluations, analyze results
python scripts/analyze_rq3.py
```

## Results Dashboard

```bash
# Interactive results visualization
streamlit run app/dashboard.py
```

The dashboard provides:
- RQ1/RQ2 result visualization with box plots and effect sizes
- Dataset explorer (image/audio browser with embeddings)
- LaTeX table export for papers
- CSV data download

---

## Optional: Dataset Expansion

### Wikimedia Images

```bash
python scripts/build_wikimedia_dataset.py
```

Downloads ~200 scene-specific images from Wikimedia Commons across nature, urban, water, and other domains.

### Freesound Audio

```bash
# Requires FREESOUND_API_KEY in .env
python scripts/build_freesound_dataset.py
```

Downloads ~200 audio files from Freesound across nature, urban, water, weather, animals, and indoor domains.

### Rebuild indexes after expansion

```bash
python scripts/build_embedding_indexes.py
```

---

## Optional: Generative Models

### AudioLDM (Audio Generation)

AudioLDM downloads automatically on first use (~2 GB). To pre-download:

```bash
python -c "from src.generators.audio.generator import AudioGenerator; AudioGenerator(force_audioldm=True)"
```

### Stable Diffusion (Image Generation)

Stable Diffusion is available as a hybrid option (SDXL with SD1.5 and retrieval fallback). To use:

```bash
python scripts/run_unified.py --prompt "A forest at dawn" --use-stable-diffusion
```

First run downloads the model (~6 GB for SDXL, ~4 GB for SD1.5 fallback).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Ollama call failed` | Make sure Ollama is running: `ollama serve` |
| `Missing image index` | Run `python scripts/build_embedding_indexes.py` |
| `Missing audio index` | Same as above |
| `No module named 'src'` | Install the package: `pip install -e .` |
| MSCI scores are all 0 | Clear cache: `rm -rf .cache/embeddings` then rebuild indexes |
| Stale embeddings after regenerating audio | Clear cache: `rm -rf .cache/embeddings` |
| `Coherence stats file not found` | Run `python scripts/fit_coherence_stats.py` first |
| AudioLDM download fails | Check internet connection; model is ~2 GB from HuggingFace |
| Stable Diffusion OOM | Use `--device cpu` or reduce batch size; SDXL needs ~8 GB VRAM |

## Data Directories

```
data/
  processed/images/     # Core image dataset (included)
  processed/audio/      # Core audio dataset (included)
  wikimedia/images/     # Expanded images (from build_wikimedia_dataset.py)
  freesound/audio/      # Expanded audio (from build_freesound_dataset.py)
  embeddings/           # Built indexes (generated by step 6)
  prompts/              # Experiment prompt sets

runs/
  rq1/                  # RQ1 experiment results + analysis JSON
  rq1_full/             # RQ1 full-pipeline robustness check
  rq2/                  # RQ2 experiment results + analysis JSON
  rq3/                  # RQ3 samples, human evaluation sessions, analysis JSON

figures/                # Publication-quality PDF figures (12 figures)
paper/                  # Research paper (paper.md)
```
