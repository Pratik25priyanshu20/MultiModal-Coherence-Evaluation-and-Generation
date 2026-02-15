# Roadmap: Future Enhancements

This document describes the status of implemented features and aspirational future work. Items marked as implemented are available in the codebase; items marked as not started represent potential directions for improvement.

---

## Status Legend

| Status | Meaning |
|--------|---------|
| Not Started | Conceptual, no code exists |
| In Progress | Partial implementation |
| Implemented | Available in codebase |
| Validated | Implemented and empirically tested |

---

## Implemented Features

### Generative Image Pipeline — Implemented

**Problem**: Retrieval-based images are constrained to existing datasets.

**Solution**: `src/generators/image/generator_hybrid.py` implements a hybrid generator with SDXL -> SD1.5 -> retrieval fallback chain. Follows the same pattern as AudioLDM in the audio generator.

**Usage**:
```python
from src.generators.image.generator_hybrid import HybridImageGenerator
gen = HybridImageGenerator(force_sd=True)
result = gen.generate("a forest at dawn", "/tmp/test.png", seed=42)
# result.backend: "sdxl", "sd1.5", or "retrieval"
```

Backward compatible: `force_sd=False` (default) uses pure retrieval.

---

### Audio Generation (AudioLDM) — Implemented

**Problem**: CLAP retrieval limited to small audio pool (60 files, only 7 unique files ever selected).

**Solution**: `src/generators/audio/generator.py` integrates AudioLDM for text-to-audio generation with retrieval fallback. Used to generate unique audio for RQ3 human evaluation samples.

**Usage**:
```python
from src.generators.audio.generator import AudioGenerator
gen = AudioGenerator(force_audioldm=True)
result = gen.generate("forest birds at dawn", "/tmp/test.wav", seed=42)
```

---

### Results Dashboard — Implemented

**Problem**: Results required manual inspection of JSON files.

**Solution**: `app/dashboard.py` — Streamlit dashboard with interactive Plotly charts.

**Features**:
- RQ1/RQ2 box plots, effect sizes, forest plots
- Paired difference visualizations
- Dataset explorer (image/audio browser)
- LaTeX table export for papers
- CSV data download

**Launch**: `streamlit run app/dashboard.py`

---

### Bootstrap Confidence Intervals — Implemented

**Problem**: Parametric CIs assume normality, which may not hold for small samples.

**Solution**: BCa (bias-corrected accelerated) bootstrap with jackknife acceleration in `src/experiments/statistical_analysis.py`.

**Features**:
- `bootstrap_ci()` for single-group CIs
- `bootstrap_ci_diff()` for difference CIs (paired or independent)
- 10,000 resamples by default
- Integrated into `analyze_results.py` output

---

### Thread-Safe Parallel Execution — Implemented

**Problem**: Experiments (270-360 runs) execute sequentially.

**Solution**: `--parallel N` flag on `scripts/run_rq1.py` and `scripts/run_rq2.py`. Thread-safe shared embedder singleton (`src/embeddings/shared_embedder.py`) avoids loading CLIP+CLAP per thread.

**Usage**: `python scripts/run_rq1.py --parallel 4`

---

### Multi-Rater Human Evaluation — Implemented

**Problem**: Single-rater evaluation cannot compute inter-rater reliability.

**Solution**: 3 independent raters (Dr Nikhil, Fariha, Pratik) rated all 30 RQ3 samples.

**Results**:
- ICC(3,k) = 0.873 (good agreement)
- Krippendorff's alpha = 0.684 (acceptable)
- MSCI-human correlation: Spearman's rho = 0.379 (p = 0.039)

---

### Publication-Quality Figures — Implemented

**Problem**: Results required manual visualization.

**Solution**: `scripts/generate_paper_figures.py` generates 12 PDF figures:

1. RQ1 raincloud plot (violin + box + jittered points)
2. Paired slope plot (per-prompt trajectories)
3. Gardner-Altman estimation plot (RQ2)
4. Forest plot (all effect sizes with CIs)
5. Channel decomposition (st_i vs st_a stacked bars)
6. Domain x Condition heatmap
7. Power curve (RQ2)
8. Bootstrap distributions
9. Skip-text vs full-pipeline robustness
10. Seed stability histograms
11. RQ3 scatter plot (MSCI vs human ratings)
12. RQ3 condition box plots

---

### Advanced Statistical Refinements — Implemented

**Solution**: Added to `src/experiments/statistical_analysis.py` and `scripts/analyze_results.py`:

- Shapiro-Wilk normality tests for all comparisons
- Wilcoxon signed-rank non-parametric backups
- Cohen's d confidence intervals (approximate)
- Holm-Bonferroni correction for RQ1 and RQ2
- Post-hoc power analysis (minimum detectable effect sizes)

---

### CI/CD Pipeline — Implemented

**Solution**: `.github/workflows/test.yml` runs 84 unit tests on push/PR to main.

---

### Dataset Expansion Scripts — Implemented

**Problem**: Core dataset (21 images, 60 audio) limits retrieval quality.

**Solution**:
- `scripts/build_wikimedia_dataset.py` — Downloads ~200 scene-specific images from Wikimedia Commons
- `scripts/build_freesound_dataset.py` — Downloads ~200 audio files from Freesound API

Both scripts include domain tagging and are integrated into the embedding index builder.

---

## Architecture Built (Training Pending)

### Cross-Space Bridge — Architecture Complete

**Problem**: CLIP (text-image) and CLAP (text-audio) occupy separate embedding spaces. Image-audio similarity is meaningless without a trained bridge. Currently si_a is omitted from MSCI entirely.

**Solution Built**: `src/embeddings/cross_space_bridge.py` — Two projection heads mapping CLIP image (512-d) and CLAP audio (512-d) into a shared 256-d bridge space.

**Architecture**:
- `BridgeProjectionHead`: Linear(512, 384) -> GELU -> Dropout -> Linear(384, 256) -> L2 norm
- `CrossSpaceBridge`: Two projection heads (image_proj, audio_proj)
- `BridgeInfoNCELoss`: Symmetric contrastive loss with learnable temperature
- `BridgeTrainer`: AdamW + cosine LR schedule + early stopping
- `ImageAudioPairDataset` + `build_paired_dataset_from_runs()` data helper

**Integration**: `CoherenceEngine.load_bridge(path)` enables si_a computation. Full MSCI formula activates: `0.45*st_i + 0.45*st_a + 0.10*si_a`.

**What's needed to train**:
- Paired (image, audio) embeddings depicting the same scenes
- Can bootstrap from existing experiment runs via `build_paired_dataset_from_runs()`
- Stronger training would use curated datasets (VGGSound, AudioCaps + MSCOCO overlap)

**Risk**: May degrade within-space alignment. Requires sufficient aligned training data.

---

## Future Work (Not Yet Implemented)

---

### MSCI Weight Optimization — Not Started

**Problem**: Fixed weights (0.45, 0.45, 0.10) are hypothesized, not learned.

**Proposed Solution**: Use human evaluation data (from RQ3) as supervision to find optimal weights via grid search or optimization.

**Requirements**:
- N >= 100 human-rated samples with reliable ratings
- Intra-rater reliability kappa >= 0.70

**Expected benefit**: MSCI better reflects human perception of coherence.

**Estimated effort**: 1 week (after human data collected)

---

### Cross-Dataset Generalization — Not Started

**Problem**: Results may not generalize across datasets/domains.

**Proposed Solution**: Evaluate MSCI on established benchmarks.

**Candidate datasets**:
- AudioCaps (audio-centric)
- COCO (image-centric)
- VGGSound (video-audio)

**Expected benefit**: Understanding of domain-specific MSCI performance.

**Estimated effort**: 2-3 weeks

---

### Temporal Audio Analysis — Not Started

**Problem**: CLAP embeddings capture audio content but not temporal structure.

**Proposed Solution**: Add temporal alignment scoring for scene dynamics (segmented audio windows matched to scene progression).

**Expected benefit**: Better evaluation for action scenes and narratives.

**Estimated effort**: 3-4 weeks

---

## Priority Order

Based on research value and current state:

1. **Cross-Space Bridge Training** — architecture built, needs paired data; enables full 3-component MSCI
2. **MSCI Weight Optimization** — RQ3 human data now available; low effort, high impact
3. **Cross-Dataset Generalization** — improves external validity
4. **Temporal Audio Analysis** — niche improvement for specific content types

---

## Non-Goals

The following are explicitly **out of scope**:

- **Video generation**: Focus remains on static multimodal content
- **Real-time streaming**: Batch evaluation is sufficient
- **Production deployment**: This is a research framework
- **Commercial features**: No monetization planned

---

## Contributing

If implementing any roadmap item:

1. Create a branch named `feature/roadmap-<item>`
2. Add appropriate tests
3. Document in code and update this file
4. Ensure `pytest tests/ -v` passes
