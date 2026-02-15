# Known Limitations

This document provides an honest assessment of the limitations in this multimodal coherence evaluation framework. Understanding these limitations is essential for appropriate use and interpretation of results.

---

## 1. MSCI as a Proxy Metric

### The Problem

MSCI (Multimodal Semantic Coherence Index) is a **proposed metric under validation**, not a validated ground truth. It measures embedding-space similarity, which may not correspond to human-perceived semantic coherence.

**Crucially: MSCI measures alignment between model representations, not alignment with meaning.** When we compute cosine similarity between CLIP embeddings, we measure how similarly CLIP encodes content — not whether the content is genuinely coherent. High MSCI reflects learned model biases, not a ground-truth semantic relationship.

### Why This Matters

- **Model alignment is not semantic alignment**: Embedding proximity means the models encode content similarly, not that the underlying content is semantically aligned
- **No causal relationship established**: High MSCI does not prove content is coherent; it measures embedding proximity
- **Model-dependent**: Results are specific to CLIP (ViT-B/32) and CLAP (HTSAT-BERT) and may not generalize to other embedding models

### What We've Done About It

- RQ3 validates that MSCI correlates with human judgments (Spearman's rho = 0.379, p = 0.039)
- The correlation is moderate — MSCI captures some but not all aspects of perceived coherence
- Per-modality sub-metrics (st_i, st_a) are reported alongside the composite MSCI
- MSCI is still reported as a proxy metric, not ground truth — the moderate correlation confirms this distinction

---

## 2. Embedding Space Separation

### The Problem

MSCI relies on two embedding models trained independently:

| Model | Training Data | Semantic Space |
|-------|---------------|----------------|
| CLIP (ViT-B/32) | Image-text pairs (400M) | Visual-language alignment |
| CLAP (HTSAT-BERT) | Audio-text pairs | Audio-language alignment |

These models were **not trained together**, so their embedding spaces are fundamentally separate.

### Implications

- Cosine similarity between CLIP image embeddings and CLAP audio embeddings is **meaningless** — they occupy different vector spaces
- Image-audio similarity (si_a) is **omitted from MSCI** for this reason
- MSCI uses text as an anchor: text-image via CLIP, text-audio via CLAP, computed independently
- The 0.45/0.45 weights renormalize to 1.0 (effectively equal weight)

### What We've Done About It

- si_a is omitted by default (set to None in coherence engine)
- ProjectionHead uses identity when in_dim == out_dim (preserves pre-trained within-space alignment)
- A `CrossSpaceBridge` architecture is built (`src/embeddings/cross_space_bridge.py`) that maps CLIP image and CLAP audio into a shared 256-d bridge space via learned projection heads with symmetric InfoNCE loss. Architecture is complete but **untrained** — requires paired image-audio data.
- Once trained, the bridge integrates via `engine.load_bridge()` to enable si_a and the full MSCI formula: `0.45*st_i + 0.45*st_a + 0.10*si_a`
- The architecture is documented clearly: `embed_text()` (CLIP) vs `embed_text_for_audio()` (CLAP)

---

## 3. Fixed MSCI Weights

### The Problem

MSCI uses fixed weights: `w_ti=0.45, w_ta=0.45` (si_a omitted, renormalized).

These weights are **hypothesized, not learned**:
- 0.45 for text-image: Assumed equal importance to visual alignment
- 0.45 for text-audio: Assumed equal importance to audio alignment

### Implications

- Optimal weights may differ by task, content type, or domain
- Fixed weights may over/under-weight certain modalities
- No empirical basis for these specific values

### What We're Doing About It

- Per-modality similarities reported separately (st_i, st_a)
- Weight optimization from human data proposed in roadmap (requires RQ3 completion)
- RQ1 sub-metric analysis confirms modality-specific sensitivity is preserved

---

## 4. Human Evaluation Scale

### The Problem

Human evaluation used 3 raters on 30 samples, which limits statistical power for per-condition analyses.

### What Was Done

1. **3 independent raters**: Dr Nikhil, Fariha, Pratik — all rated all 30 samples
2. **Blind evaluation**: Condition labels hidden from evaluators
3. **Randomized order**: Per-evaluator deterministic shuffle
4. **Structured rubric**: Explicit criteria with 5-point Likert scales
5. **Session persistence**: Evaluators could save/resume to reduce fatigue

### Inter-Rater Reliability (Achieved)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICC(3,1) | 0.697 | Moderate (single rater) |
| ICC(3,k) | 0.873 | Good (averaged across 3 raters) |
| Krippendorff's alpha | 0.684 | Acceptable (> 0.667) |

### Remaining Limitations

- 3 raters is the minimum for ICC — more raters would improve reliability estimates
- Per-condition sample size (n=10) too small for between-condition significance tests
- Generalizability to broader population unknown
- Rater pool from a single academic group may share biases

---

## 5. Retrieval Pool Ceiling

### The Problem

The core dataset constrains retrieval quality:

| Modality | Pool Size (expanded) | Original Size |
|----------|---------------------|---------------|
| Images | 57 files (5 nature, 23 urban, 9 water, 20 other) | 21 |
| Audio | 104 files (22 nature, 28 urban, 33 water, 21 other) | 60 |

The dataset was expanded from 21 images / 60 audio to 57 / 104 via Wikimedia and Freesound. While significantly improved, domain balance remains uneven (e.g., only 5 nature images).

### Implications

- **RQ2 null result is explained by this ceiling**: Planning cannot improve retrieval when no better match exists in the pool
- 72.6% of image retrievals fall below 0.25 cosine similarity (floor-level alignment)
- The top-5 similarity gap for images is only 0.06 — the best match is barely better than alternatives
- Results may not generalize to larger, more diverse datasets

### What We've Done About It

- Dataset expanded via Wikimedia (images) and Freesound (audio) scripts
- Generative pipelines available (Stable Diffusion for images, AudioLDM for audio) to bypass retrieval entirely
- Retrieval bottleneck is fully quantified in the results summary (similarity distributions, per-domain breakdown)
- Further expansion possible by re-running the download scripts with different targets

---

## 6. Image Generation

### Current State

Image generation defaults to **retrieval** from pre-indexed datasets. A **hybrid generator** with Stable Diffusion (SDXL/SD1.5) is implemented but not used in the reported RQ1/RQ2 experiments.

### Implications for Reported Results

- RQ1/RQ2 images are constrained to the 21-image retrieval pool
- Coherence failures may be dataset limitations, not pipeline failures
- Image-text similarity ceiling is limited by retrieval quality

### Available but Not Yet Validated

- `HybridImageGenerator` supports SDXL -> SD1.5 -> retrieval fallback
- Generative mode bypasses the retrieval ceiling
- Results with generative images have not yet been formally evaluated

---

## 7. Audio Generation Quality

### Current State

Audio generation supports two backends:

| Backend | Quality | Use Case |
|---------|---------|----------|
| **AudioLDM** (cvssp/audioldm) | Semantically meaningful, prompt-specific | RQ3 human evaluation |
| **CLAP Retrieval** | Limited to pool diversity | RQ1/RQ2 experiments |
| **Ambient Fallback** | Synthetic drones, poor CLAP similarity | Emergency fallback only |

### Implications

- RQ3 uses AudioLDM-generated audio (unique per sample, semantically meaningful)
- RQ1/RQ2 used CLAP retrieval, subject to the pool ceiling
- The ambient fallback produces audio that CLAP cannot meaningfully score — must be avoided for evaluation
- Embedding cache must be cleared after regenerating audio at the same file paths

---

## 8. Sample Size and Statistical Power

### The Problem

| Effect Size | Required N (alpha=0.05, power=0.80) |
|-------------|-------------------------------------|
| Large (d=0.8) | ~15 |
| Medium (d=0.5) | ~34 |
| Small (d=0.3) | ~90 |

### Current State

- RQ1 (N=30): Massively overpowered (observed d > 2.2, required N = 2)
- RQ2 (N=30): Adequate for medium effects; small effects (d=0.3-0.4) detected as trends but do not survive correction
- RQ3 (N=30): Adequate for large correlations; may miss weak correlations

### What We're Doing About It

- Power analysis reported alongside results
- Effect sizes always reported alongside p-values
- Bootstrap BCa confidence intervals provide non-parametric uncertainty estimates
- Null results interpreted cautiously with explicit power discussion

---

## 9. Prompt Domain Bias

### The Problem

All 30 prompts span 4 domains: nature, urban, water, mixed. Results may not generalize to:
- Abstract concepts
- Emotional content
- Action scenes requiring temporal understanding
- Indoor environments
- Rare or specialized domains

### Mitigations

- Stratified sampling across available domains
- Per-domain analysis available (nature/urban/water/mixed breakdowns)
- Domain-specific retrieval quality quantified in bottleneck analysis

---

## 10. Determinism vs. Realism Trade-off

### The Problem

Deterministic mode (fixed seed -> same output) is used for reproducibility, but:
- Real-world usage involves stochastic generation
- Deterministic outputs may not represent typical behavior
- Variance estimates may be artificially narrow

### Mitigations

- Three seeds per prompt (42, 123, 7) sample stochastic variation
- Results averaged across seeds before hypothesis testing (paired by prompt)
- More seeds would improve precision of prompt-level estimates

---

## Summary Table

| Limitation | Severity | Mitigation Status |
|------------|----------|-------------------|
| MSCI proxy metric | Medium | RQ3 validates moderate correlation (rho = 0.379) |
| Embedding space separation | High | si_a omitted; bridge architecture built (untrained) |
| Fixed weights | Medium | Per-modality reporting; optimization planned |
| Human eval scale | Medium | 3 raters, ICC = 0.873; more raters would strengthen |
| Retrieval pool ceiling | Medium | Expanded to 57 images / 104 audio; further expansion possible |
| Image generation (retrieval only in experiments) | Medium | SD hybrid implemented, not yet evaluated |
| Audio quality | Medium | AudioLDM for human eval; retrieval for experiments |
| Sample size | Low-Medium | Power analysis; bootstrap CIs; Shapiro-Wilk + Wilcoxon backups |
| Prompt domain bias | Low | Stratified sampling |
| Determinism trade-off | Low | Multiple seeds |

---

## Honest Interpretation Guidelines

When interpreting results from this framework:

1. **Report MSCI as a proposed metric**, not ground truth
2. **Include effect sizes** alongside p-values
3. **Acknowledge the retrieval ceiling** when discussing RQ2
4. **Note embedding space separation** for cross-modal comparisons
5. **Report human evaluation reliability** (intra-rater kappa)
6. **Discuss generalizability limits** to other pipelines, models, and domains
7. **Report bootstrap CIs** alongside parametric confidence intervals

---

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Hallgren, K. A. (2012). Computing Inter-Rater Reliability for Observational Data
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- Elizalde, B., et al. (2023). CLAP: Learning Audio Concepts from Natural Language Supervision
- Liu, H., et al. (2023). AudioLDM: Text-to-Audio Generation with Latent Diffusion Models
