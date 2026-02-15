# Multimodal Coherence Evaluation Framework

An empirical framework for measuring and validating semantic coherence across text, image, and audio modalities. This project introduces **MSCI (Multimodal Semantic Coherence Index)**, a composite metric aggregating embedding-based similarities, and provides the experimental infrastructure to test whether it captures meaningful cross-modal alignment.

---

## Research Questions & Status

| RQ | Question | Status | Verdict |
|----|----------|--------|---------|
| **RQ1** | Is MSCI sensitive to controlled semantic perturbations? | Complete | **Supported** (p < 10^-13, d > 2.2) |
| **RQ2** | Does structured semantic planning improve cross-modal alignment? | Complete | **Not Supported** (retrieval ceiling) |
| **RQ3** | Does MSCI correlate with human judgments of coherence? | Complete | **Supported** (ρ = 0.379, p = 0.039) |

See [docs/RESULTS.md](docs/RESULTS.md) for detailed results and [docs/results_summary.md](docs/results_summary.md) for the full statistical report.

---

## Key Findings

**RQ1 — MSCI Sensitivity**: MSCI reliably drops under controlled perturbations. Replacing a matched image reduces MSCI by 13% (d = 2.34); replacing matched audio reduces it by 44% (d = 2.21). Sub-metrics are modality-specific: wrong_image degrades only st_i, wrong_audio degrades only st_a.

**RQ2 — Planning Effect**: None of the three planning modes (single planner, council, extended prompt) improved MSCI over the direct baseline. The primary explanation is a **retrieval pool ceiling** — with 21 images and 60 audio files, better prompts cannot improve retrieval when no better match exists. This is an informative null result, not a framework failure.

**RQ3 — Human Correlation**: MSCI shows statistically significant correlation with human coherence judgments. Three independent raters evaluated 30 blind samples. Spearman's ρ = 0.379 (p = 0.039), confirmed by Kendall's τ = 0.293 (p = 0.038). Inter-rater reliability: ICC(3,k) = 0.873, Krippendorff's α = 0.684.

---

## What MSCI Is (and Isn't)

**MSCI (Multimodal Semantic Coherence Index)** aggregates embedding-based similarities:

```
MSCI = 0.45 x sim(text, image) + 0.45 x sim(text, audio)
```

Image-audio similarity (si_a) is currently omitted because CLIP and CLAP occupy separate embedding spaces. A `CrossSpaceBridge` architecture exists in the codebase (`src/embeddings/cross_space_bridge.py`) that, once trained, would enable si_a and activate the full formula: `MSCI = 0.45 * st_i + 0.45 * st_a + 0.10 * si_a`.

### Important Caveats

- **MSCI measures alignment between model representations, not alignment with meaning.** It captures how similarly CLIP and CLAP encode content — not whether that content is semantically coherent to a human.
- CLIP and CLAP were trained independently on different data; cross-space comparisons are not meaningful without alignment.
- **RQ3 validates** that MSCI correlates with human judgments (ρ = 0.379, p = 0.039) — a moderate but statistically significant relationship.
- The fixed weights (0.45, 0.45) are hypothesized, not empirically derived.

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for a complete discussion.

---

## System Architecture

```
User Prompt
    |
    v
+-----------------------------------------------+
|         Semantic Planning (Optional)           |
|  +-------------------------------------------+|
|  | Direct: No planning (baseline)            ||
|  | Single Planner: 1 LLM call                ||
|  | Council: 3 LLM calls + merge              ||
|  | Extended Prompt: 1 LLM, 3x tokens         ||
|  +-------------------------------------------+|
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
|         Multimodal Generation                  |
|  Text          | Image           | Audio       |
|  Ollama/Qwen2  | SD / Retrieval  | AudioLDM /  |
|                | (CLIP-ranked)   | Retrieval   |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
|         Coherence Evaluation                   |
|  CLIP (text <-> image)  |  CLAP (text <-> audio)|
|                    |                            |
|               MSCI Score                        |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
|  Statistical Analysis & Visualization          |
|  Bootstrap CIs | Effect Sizes | Dashboard      |
+-----------------------------------------------+
```

**Embedding spaces**:
- **CLIP** (openai/clip-vit-base-patch32, 512-d): text and image in shared space
- **CLAP** (laion/clap-htsat-unfused, 512-d): text and audio in shared space
- CLIP and CLAP are **separate** spaces — a `CrossSpaceBridge` architecture is built for future image-audio alignment

---

## Quick Start

### 1. Installation

```bash
pip install -e ".[dev]"    # Core + dev tools
```

Requires [Ollama](https://ollama.ai/) for local LLM planning. See [INSTALL.md](INSTALL.md) for detailed setup.

### 2. Build Embedding Indexes

```bash
python scripts/build_embedding_indexes.py
```

### 3. Run a Single Generation

```bash
python scripts/run_unified.py --prompt "A peaceful forest at dawn with birdsong"
```

### 4. Run Experiments

```bash
# RQ1: MSCI sensitivity (270 runs)
python scripts/run_rq1.py

# RQ2: Planning ablation (360 runs)
python scripts/run_rq2.py

# Parallel execution (4 threads)
python scripts/run_rq1.py --parallel 4
```

### 5. Analyze Results

```bash
python scripts/analyze_results.py runs/rq1/rq1_results.json
python scripts/analyze_results.py runs/rq2/rq2_results.json
```

### 6. Human Evaluation (RQ3)

```bash
streamlit run app/human_eval_app.py
```

### 7. Results Dashboard

```bash
streamlit run app/dashboard.py
```

---

## Features

| Feature | Description |
|---------|-------------|
| **MSCI Computation** | Embedding-based coherence scoring with CLIP + CLAP |
| **Controlled Experiments** | RQ1 (sensitivity), RQ2 (planning), RQ3 (human validation) |
| **4 Planning Modes** | Direct, single planner, council (3-agent), extended prompt |
| **Hybrid Image Generation** | Stable Diffusion (SDXL/SD1.5) with retrieval fallback |
| **Audio Generation** | AudioLDM with CLAP-based retrieval fallback |
| **Bootstrap Confidence Intervals** | BCa (bias-corrected accelerated) non-parametric CIs |
| **Thread-Safe Parallel Execution** | Shared embedder singleton, `--parallel N` flag |
| **Human Evaluation App** | Streamlit UI with blind evaluation, re-rating, session management |
| **Results Dashboard** | Interactive Plotly charts, effect sizes, forest plots, data export |
| **CI/CD** | GitHub Actions running 84 unit tests on push/PR |
| **Dataset Expansion** | Wikimedia (images) + Freesound (audio) downloaders |

---

## Project Structure

```
src/
  coherence/          # MSCI computation, coherence classification
  embeddings/         # CLIP, CLAP, projection heads, cross-space bridge, shared singleton
  evaluation/         # Human evaluation schema and analysis
  experiments/        # Statistical framework (bootstrap CIs, effect sizes, Shapiro-Wilk, Wilcoxon)
  generators/
    text/             # Ollama/Qwen2 text generation
    image/            # CLIP retrieval + Stable Diffusion hybrid
    audio/            # CLAP retrieval + AudioLDM generation
  planner/            # Semantic planning (single, council, extended)
  pipeline/           # End-to-end orchestration
  validation/         # MSCI sensitivity, correlation, calibration
  utils/              # Seeding, caching, performance monitoring

scripts/
  run_rq1.py                  # RQ1 experiment (sensitivity)
  run_rq2.py                  # RQ2 experiment (planning)
  run_unified.py              # Single generation
  analyze_results.py          # Statistical analysis with bootstrap CIs
  analyze_rq3.py              # RQ3 human evaluation analysis (ICC, Spearman, Kendall)
  generate_paper_figures.py   # 12 publication-quality figures
  build_embedding_indexes.py  # CLIP/CLAP index builder
  build_wikimedia_dataset.py  # Wikimedia image downloader
  build_freesound_dataset.py  # Freesound audio downloader
  sanity_check.py             # Pipeline validation (4 checks)
  select_rq3_samples.py       # Stratified sample selection for human eval

app/
  human_eval_app.py   # Streamlit human evaluation interface
  dashboard.py        # Streamlit results dashboard

paper/                # Research paper (paper.md)
figures/              # 12 publication-quality PDF figures
tests/                # 84 unit tests
docs/                 # Results, limitations, roadmap
```

---

## Key Files

| Purpose | File |
|---------|------|
| Pipeline orchestration | `src/pipeline/generate_and_evaluate.py` |
| Coherence evaluation | `src/coherence/coherence_engine.py` |
| Aligned embeddings | `src/embeddings/aligned_embeddings.py` |
| Cross-space bridge (CLIP-CLAP) | `src/embeddings/cross_space_bridge.py` |
| Image retrieval | `src/generators/image/generator_improved.py` |
| Hybrid image (SD + retrieval) | `src/generators/image/generator_hybrid.py` |
| Audio retrieval | `src/generators/audio/retrieval.py` |
| Audio generation (AudioLDM) | `src/generators/audio/generator.py` |
| Statistical analysis | `src/experiments/statistical_analysis.py` |
| Human eval schema | `src/evaluation/human_eval_schema.py` |
| Paper figures | `scripts/generate_paper_figures.py` |

---

## Experimental Framework

### Statistical Methods

- **Paired t-tests** for within-subject comparisons
- **Shapiro-Wilk normality tests** with Wilcoxon signed-rank non-parametric backup
- **Cohen's d** effect sizes with approximate confidence intervals
- **Bootstrap BCa confidence intervals** (bias-corrected accelerated, 10,000 resamples)
- **Holm-Bonferroni correction** for multiple comparisons
- **Power analysis** for sample size adequacy (non-central t-distribution)
- **ICC** (intraclass correlation) and **Krippendorff's alpha** for inter-rater reliability
- **Spearman's rho** and **Kendall's tau** for MSCI-human correlation

### Experiment Design

| Experiment | Prompts | Seeds | Conditions | Total Runs |
|------------|---------|-------|------------|------------|
| RQ1 | 30 | 3 | 3 (baseline, wrong_image, wrong_audio) | 270 |
| RQ2 | 30 | 3 | 4 (direct, planner, council, extended) | 360 |
| RQ3 | 30 stratified samples | — | 3 conditions, 3 raters | 90 human ratings |

### Negative Results Policy

All results are reported honestly, including null findings (RQ2), unexpected directions, and effect sizes too small to be meaningful. See [docs/results_summary.md](docs/results_summary.md).

---

## Human Evaluation Protocol (RQ3)

- **3 independent raters**: Blind evaluation, condition labels hidden
- **Randomized order**: Per-evaluator deterministic shuffle
- **Structured rubric**: 5-point Likert scale for text-image, text-audio, image-audio, and overall coherence
- **Inter-rater reliability**: ICC(3,k) = 0.873 (good), Krippendorff's alpha = 0.684 (acceptable)
- **MSCI-human correlation**: Spearman's rho = 0.379 (p = 0.039), Kendall's tau = 0.293 (p = 0.038)
- **Session persistence**: Save/resume, multiple evaluators supported

---

## Limitations

Key limitations (see [docs/LIMITATIONS.md](docs/LIMITATIONS.md)):

- **MSCI measures model alignment, not semantic truth** — results are specific to CLIP/CLAP
- **Embedding space separation** — CLIP and CLAP are independent spaces; cross-space bridge architecture built but untrained
- **Fixed MSCI weights** — not empirically derived from human data
- **Moderate human correlation** — ρ = 0.379 indicates MSCI captures some but not all aspects of perceived coherence
- **Dataset size** — expanded to 57 images and 104 audio files, but still limited domain coverage

---

## Future Work

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed descriptions:

- **Cross-space bridge training** — architecture built (`src/embeddings/cross_space_bridge.py`), awaiting paired training data to enable image-audio similarity (si_a)
- MSCI weight optimization from human evaluation data
- Cross-dataset generalization (AudioCaps, COCO, VGGSound)

---

## Citation

```bibtex
@software{multimodal_coherence_2026,
  title = {Multimodal Coherence Evaluation Framework},
  year = {2026},
  note = {An empirical study of semantic coherence measurement in multimodal AI systems}
}
```

---

## License

MIT License — see LICENSE file.
