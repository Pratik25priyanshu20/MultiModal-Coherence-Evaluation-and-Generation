# Evaluating Cross-Modal Semantic Coherence in Multimodal Generation

---

## Abstract

Multimodal content generation systems that combine text, images, and audio face a fundamental evaluation challenge: how to measure whether the generated modalities are semantically coherent with each other. We propose the **Multimodal Semantic Coherence Index (MSCI)**, a metric that leverages pre-trained CLIP and CLAP embedding spaces to quantify text–image and text–audio alignment across both retrieval-based and generative multimodal pipelines. We evaluate MSCI through three research questions using two complementary paradigms: a retrieval pipeline (curated index of 57 images and 104 audio clips) and a hybrid generative pipeline (Stable Diffusion XL for image generation with CLAP-based audio retrieval). **RQ1** tests whether MSCI is sensitive to controlled semantic perturbations. Under retrieval, MSCI reliably detects both image and audio mismatches with large effect sizes (Cohen's *d* = 2.11–3.64, all *p*-adj < 10⁻¹²). Under generation, sensitivity is preserved and amplified (*d* = 2.02–4.52, all *p*-adj < 10⁻¹¹), confirming pipeline-agnostic validity. **RQ2** investigates whether structured planning strategies improve cross-modal alignment. Under retrieval, a well-powered null result (80% power for *d* ≥ 0.53) finds no benefit (all |*d*| ≤ 0.19), identifying the retrieval index as the bottleneck. Under generation, planning actively *reduces* alignment (*d* = −0.82 to −1.51, all *p*-adj < 0.03) due to CLIP token truncation of verbose planned prompts. **RQ3** validates MSCI against human coherence judgments from three independent raters (ICC = 0.70, Krippendorff's α = 0.68), finding a significant positive correlation (Spearman's ρ = 0.379, *p* = 0.039). We conclude that MSCI is a statistically sensitive and human-aligned metric for evaluating cross-modal coherence, and that the choice of pipeline architecture fundamentally shapes how planning strategies interact with alignment.

**Keywords:** multimodal coherence, CLIP, CLAP, cross-modal evaluation, multimodal generation, semantic alignment

---

## 1. Introduction

The proliferation of multimodal AI systems—those that generate or compose content across text, image, and audio modalities—has created a pressing need for evaluation metrics that assess not just the quality of individual modalities but their *semantic coherence* as a unified experience (Baltrusaitis et al., 2019). A nature scene paired with urban traffic noise, or a beach description accompanied by a photograph of a city skyline, would each represent a coherence failure that degrades the user experience regardless of how high-quality the individual components are.

Existing evaluation approaches either focus on single-modality quality (FID for images, PESQ for audio, perplexity for text) or require expensive human annotation for every new composition. Pre-trained vision-language models such as CLIP (Radford et al., 2021) and audio-language models such as CLAP (Wu et al., 2023) offer a potential solution: their shared embedding spaces can serve as automatic coherence proxies, measuring the semantic distance between a text description and its paired image or audio.

However, several questions remain unanswered:

1. **Sensitivity:** Are embedding-based coherence metrics actually sensitive to semantic perturbations, or do they merely capture surface-level features?
2. **Planning:** Can structured prompt planning (decomposition, multi-agent deliberation) improve the coherence of generated multimodal bundles?
3. **Human alignment:** Do automatic coherence scores correlate with human perception of multimodal coherence?

This paper addresses these questions through a systematic evaluation of MSCI, a composite metric that combines CLIP-based text–image similarity with CLAP-based text–audio similarity. We evaluate MSCI across two complementary multimodal pipelines—a retrieval-based system that selects media from a curated index and a hybrid generative system that produces images via Stable Diffusion XL—using controlled perturbation experiments, planning ablations, and multi-rater human evaluation. By testing each research question under both paradigms, we assess whether MSCI's validity is pipeline-specific or generalizable.

To our knowledge, this is among the first systematic perturbation-based validations of embedding-derived multimodal coherence metrics across text, image, and audio, and the first to compare retrieval and generative pipelines within the same evaluation framework.

### Contributions

- We propose MSCI, a composite cross-modal coherence metric using pre-trained CLIP and CLAP embeddings, and demonstrate its statistical sensitivity to controlled perturbations under both retrieval and generative pipelines.
- We evaluate MSCI across two paradigms—retrieval-based (curated index) and hybrid generative (SDXL + CLAP retrieval)—establishing that the metric is pipeline-agnostic while revealing that pipeline architecture fundamentally shapes how planning strategies interact with alignment.
- We provide evidence that structured planning does not improve retrieval-based alignment (identifying the retrieval index as the bottleneck) and actively *reduces* generative alignment due to CLIP token truncation—a novel finding with practical implications for prompt engineering.
- We validate MSCI against human coherence judgments with acceptable inter-rater reliability (ICC = 0.70, α = 0.68), establishing it as a viable automatic proxy for human evaluation.

---

## 2. Related Work

### 2.1 Vision-Language Models

CLIP (Contrastive Language-Image Pre-training; Radford et al., 2021) learns a shared 512-dimensional embedding space for images and text through contrastive learning on 400 million image-text pairs. CLIPScore (Hessel et al., 2021) has been widely adopted as a reference-free evaluation metric for image captioning and text-to-image generation. We build on this approach by extending it to the audio modality.

### 2.2 Audio-Language Models

CLAP (Contrastive Language-Audio Pre-training; Wu et al., 2023) applies the same contrastive learning paradigm to audio and text, learning a shared embedding space from audio-text pairs. CLAP enables zero-shot audio classification and retrieval, and its embedding space provides a natural analogue to CLIP for measuring text–audio coherence.

### 2.3 Multimodal Evaluation

Evaluating multimodal coherence remains challenging. Metrics such as FID (Heusel et al., 2017), IS (Salimans et al., 2016), and PESQ (Rix et al., 2001) assess individual modality quality but cannot capture cross-modal alignment. Human evaluation remains the gold standard but is expensive and difficult to scale. Recent work on composite metrics for vision-language tasks (Lee et al., 2023) and audio-visual alignment (Yariv et al., 2024) motivates our approach of combining modality-specific embedding similarities into a unified coherence score.

### 2.4 Planning in Multimodal Generation

Chain-of-thought prompting (Wei et al., 2022), multi-agent deliberation (Du et al., 2024), and decomposed task planning (Khot et al., 2023) have shown improvements in single-modality generation quality. Whether these techniques improve *cross-modal coherence* in multimodal settings is an open question that we address in RQ2.

---

## 3. Methodology

### 3.1 Multimodal Semantic Coherence Index (MSCI)

We define MSCI as a weighted combination of pairwise cosine similarities between modality embeddings:

$$\text{MSCI} = w_{ti} \cdot s(\mathbf{e}_t^{\text{CLIP}}, \mathbf{e}_i^{\text{CLIP}}) + w_{ta} \cdot s(\mathbf{e}_t^{\text{CLAP}}, \mathbf{e}_a^{\text{CLAP}})$$

where:
- $s(\cdot, \cdot)$ denotes cosine similarity
- $\mathbf{e}_t^{\text{CLIP}}$ and $\mathbf{e}_i^{\text{CLIP}}$ are the CLIP text and image embeddings (ViT-B/32, 512-d)
- $\mathbf{e}_t^{\text{CLAP}}$ and $\mathbf{e}_a^{\text{CLAP}}$ are the CLAP text and audio embeddings (HTSAT-unfused, 512-d)
- $w_{ti} = w_{ta} = 0.45$ are the channel weights

**Architectural constraint.** CLIP and CLAP occupy distinct embedding spaces—CLIP text embeddings are aligned with images, while CLAP text embeddings are aligned with audio. Direct comparison between CLIP image embeddings and CLAP audio embeddings is not meaningful without a trained cross-space projection. We therefore omit the image–audio similarity term ($s_{ia}$) from MSCI and use separate text encoders for each channel: `embed_text()` (CLIP) for text–image and `embed_text_for_audio()` (CLAP) for text–audio. MSCI thus operationalizes multimodal coherence as convergence toward a shared textual semantic anchor rather than direct pairwise compatibility between all modalities.

**Projection.** When embedding dimensions match the pre-trained model output (512-d for both CLIP and CLAP), projection heads operate as identity functions, preserving the pre-trained alignment. This is a deliberate design choice: random linear projections would destroy the contrastive structure learned during pre-training.

### 3.2 Retrieval Pipeline

Our multimodal generation pipeline operates as follows:

1. **Text generation.** Given a scene prompt, an LLM (Ollama, local inference) generates descriptive text. In skip-text mode, the original prompt is used directly.
2. **Domain gating.** The prompt is classified into an environmental domain (nature, urban, water) using keyword matching. Retrieved media must be compatible with the prompt domain.
3. **Image retrieval.** The prompt is encoded via CLIP text encoder. Cosine similarity against the image embedding index selects the best domain-compatible match above a threshold of 0.20.
4. **Audio retrieval.** The prompt is encoded via CLAP text encoder. Cosine similarity against the audio embedding index selects the best domain-compatible match.
5. **MSCI computation.** The text–image and text–audio similarities are combined into the MSCI score.

### 3.3 Hybrid Generative Pipeline

As a second evaluation paradigm, we employ a hybrid generative pipeline that pairs generated images with retrieved audio. Images are generated using Stable Diffusion XL (SDXL; Podell et al., 2024), while audio is retrieved from the CLAP embedding index. This hybrid design enables direct comparison between retrieval and generation for the image modality while holding the audio channel constant. The pipeline uses a sequential architecture—SDXL image generation, then CLAP audio retrieval, then MSCI evaluation—to remain within hardware memory constraints. CLIP and CLAP embeddings remain unchanged from the retrieval pipeline, ensuring comparable MSCI computation across both paradigms.

### 3.4 Planning Strategies (RQ2)

We compare four generation modes:

- **Direct:** Single LLM call generates text from the prompt; retrieval uses the prompt directly.
- **Planner:** A decomposed planning step breaks the prompt into sub-goals before generation.
- **Council:** Multi-agent deliberation where multiple LLM calls propose and vote on generation strategies.
- **Extended Prompt:** The prompt is enriched with domain context and sensory descriptors before generation.

---

## 4. Experimental Design

### 4.1 RQ1: Sensitivity to Controlled Perturbations

**Design.** Within-subject, three conditions: baseline (matched image + audio), wrong-image (mismatched image, matched audio), and wrong-audio (matched image, mismatched audio).

**Retrieval pipeline:** 30 prompts × 3 seeds × 3 conditions = 270 runs per mode. Primary analysis uses skip-text mode (text held constant at the original prompt) to isolate the retrieval variable. Full-pipeline mode (LLM text generation included) serves as a robustness check.

**Generative pipeline:** 30 prompts × 1 seed × 3 conditions = 90 runs. Images generated by SDXL; audio retrieved via CLAP. For image perturbation, SDXL generates from a different-domain prompt. For audio perturbation, a different-domain audio clip is retrieved.

**Statistical tests:** Paired t-tests (one-sided: baseline > perturbation), Holm–Bonferroni correction for 2 comparisons, Shapiro–Wilk normality verification, Wilcoxon signed-rank as non-parametric backup, Cohen's *d* with 95% CIs, bootstrap CIs (10,000 resamples).

### 4.2 RQ2: Effect of Planning on Alignment

**Design.** Within-subject, four modes: direct, planner, council, extended prompt.

**Retrieval pipeline:** 30 prompts × 3 seeds × 4 modes = 360 runs. Full pipeline with LLM text generation.

**Generative pipeline:** 10 stratified prompts (3 nature, 3 urban, 2 water, 2 mixed) × 4 modes × 1 seed = 40 runs. Images generated by SDXL; audio retrieved via CLAP. MSCI uses the original prompt as text anchor (skip-text mode) to isolate image/audio variation.

**Statistical tests:** Paired t-tests (two-sided) for each planning mode vs. direct, Holm–Bonferroni correction for 3 comparisons, Shapiro–Wilk, Wilcoxon signed-rank, Cohen's *d* with 95% CIs, post-hoc sensitivity analysis (minimum detectable effect at 80% and 90% power).

### 4.3 RQ3: Human Alignment Validation

**Design.** 30 stratified samples (10 baseline, 10 wrong-image, 10 wrong-audio) selected from RQ1/RQ2 results spanning the full MSCI range [0.105, 0.471]. Three independent raters evaluated each sample blindly via a web-based interface (Streamlit), rating overall coherence on a 1–5 Likert scale.

**Statistical tests:** Inter-rater reliability (ICC(3,1), ICC(3,k), Krippendorff's α), Spearman's ρ and Kendall's τ between MSCI and consensus human ratings (median across raters), Kruskal–Wallis test for between-condition differences in human ratings.

---

## 5. Results

### 5.1 RQ1: Perturbation Sensitivity

#### 5.1.1 Retrieval Pipeline

**Table 1.** MSCI under controlled perturbations — retrieval pipeline (Holm–Bonferroni corrected).

| Condition | Mode | Mean MSCI (SD) | 95% CI | Δ vs Baseline | Cohen's *d* [95% CI] | *p*-adj |
|-----------|------|----------------|--------|---------------|----------------------|---------|
| Baseline | skip-text | 0.394 (0.056) | [0.374, 0.413] | — | — | — |
| Wrong Image | skip-text | 0.347 (0.054) | [0.328, 0.366] | −0.047 | 2.27 [1.59, 2.95] | 1.99 × 10⁻¹³ |
| Wrong Audio | skip-text | 0.170 (0.039) | [0.158, 0.185] | −0.224 | 3.64 [2.65, 4.63] | 1.73 × 10⁻¹⁸ |
| Baseline | full | 0.348 (0.062) | [0.323, 0.367] | — | — | — |
| Wrong Image | full | 0.313 (0.063) | [0.289, 0.334] | −0.035 | 2.11 [1.47, 2.75] | 1.14 × 10⁻¹² |
| Wrong Audio | full | 0.172 (0.042) | [0.158, 0.188] | −0.176 | 2.78 [1.99, 3.57] | 2.20 × 10⁻¹⁵ |

*All p-values Holm–Bonferroni corrected for 2 comparisons per mode.*

**Normality verification.** Shapiro–Wilk tests on paired differences confirmed normality for all comparisons (all *p* > 0.13). Wilcoxon signed-rank tests yielded consistent results (all *p* < 10⁻⁹, rank-biserial *r* ≥ 0.996).

MSCI is statistically sensitive to cross-modal perturbations under retrieval. Both wrong-image and wrong-audio substitutions produce significant MSCI decreases with large effect sizes. The audio channel shows a substantially stronger perturbation signal (Δ = −0.224, *d* = 3.64) than the image channel (Δ = −0.047, *d* = 2.27). This asymmetry is discussed in Section 6.1. All 30 prompts showed decreased MSCI under both perturbation types (100% consistency). The full-pipeline baseline (0.348) is lower than the skip-text baseline (0.394), reflecting additional variance introduced by LLM text generation. Both modes yield consistent conclusions (Figure A.5).

[Figure 1: Raincloud plot — MSCI distribution by perturbation condition]
[Figure 2: Paired slope plot — per-prompt MSCI trajectories]

#### 5.1.2 Generative Pipeline

**Table 2.** MSCI under controlled perturbations — generative pipeline (SDXL images + CLAP-retrieved audio).

| Condition | st_i (CLIP) | st_a (CLAP) | MSCI | Cohen's *d* | *p*-adj |
|-----------|-------------|-------------|------|-------------|---------|
| Baseline | 0.327 | 0.545 | 0.436 | — | — |
| Wrong Image | 0.142 | 0.545 | 0.344 | 4.52 | < 10⁻²⁰ |
| Wrong Audio | 0.327 | 0.231 | 0.279 | 2.02 | < 10⁻¹¹ |

All 30 prompts show higher baseline MSCI than both perturbation conditions, and both comparisons survive Holm–Bonferroni correction.

MSCI's sensitivity is preserved and amplified under generation. SDXL-generated images achieve strong CLIP alignment (st_i = 0.327), and mismatched-domain generation produces a reliable MSCI decrease (*d* = 4.52)—larger than the retrieval-based image effect (*d* = 2.27). This suggests that generative models produce *more domain-distinctive* images than a curated retrieval index, amplifying the perturbation signal. The audio channel, using the same CLAP retrieval, replicates the expected large effect (*d* = 2.02 vs retrieval *d* = 3.64).

[Figure 6: Retrieval vs Generative comparison — side-by-side MSCI by condition and effect sizes]

#### 5.1.3 Cross-Paradigm Comparison

Both paradigms confirm that MSCI is sensitive to cross-modal perturbations with large effect sizes (*d* > 2.0 in all comparisons). The generative pipeline produces *larger* image perturbation effects (*d* = 4.52 vs 2.27) because SDXL generates highly domain-specific images that diverge strongly when prompted with the wrong domain. The retrieval pipeline produces larger audio perturbation effects (*d* = 3.64 vs 2.02) because CLAP retrieval from a curated index is more discriminative when operating over a controlled corpus. Together, these results confirm that MSCI captures genuine cross-modal semantic alignment rather than pipeline-specific artifacts.

### 5.2 RQ2: Effect of Planning on Alignment

#### 5.2.1 Retrieval Pipeline

**Table 3.** MSCI by planning strategy — retrieval pipeline (Holm–Bonferroni corrected).

| Planning Mode | Mean MSCI (SD) | 95% CI | Δ vs Direct | Cohen's *d* [95% CI] | *p*-adj |
|---------------|----------------|--------|-------------|----------------------|---------|
| Direct | 0.348 (0.062) | [0.323, 0.367] | — | — | — |
| Extended Prompt | 0.349 (0.042) | [0.333, 0.363] | +0.001 | 0.01 [−0.35, 0.37] | 0.972 |
| Council | 0.338 (0.046) | [0.322, 0.354] | −0.010 | −0.19 [−0.55, 0.17] | 0.885 |
| Planner | 0.337 (0.044) | [0.321, 0.352] | −0.011 | −0.18 [−0.54, 0.18] | 0.639 |

*All p-values Holm–Bonferroni corrected for 4 comparisons.*

**Normality verification.** Shapiro–Wilk tests passed for all comparisons (*p* > 0.29) except extended_prompt vs. direct (*p* = 0.042). The Wilcoxon signed-rank test confirmed the non-significant result for this comparison (*p* = 0.61).

**Sensitivity analysis.** At N = 30 and α = 0.05, a paired t-test achieves 80% power to detect effects of *d* ≥ 0.53 (medium) and 90% power for *d* ≥ 0.62 (see Figure A.3 for the full power curve). The observed effect sizes (|*d*| ≤ 0.19) fall well below this threshold, with observed statistical power ranging from 5.0% to 17.8%.

No planning strategy significantly outperformed the direct baseline. This null result is well-powered and robust across both parametric and non-parametric tests.

[Figure 3: Gardner-Altman estimation plots — planning modes vs. direct]

#### 5.2.2 Generative Pipeline

**Table 4.** MSCI by planning strategy — generative pipeline (SDXL images + CLAP retrieval, Holm–Bonferroni corrected).

| Planning Mode | Mean MSCI (SD) | Δ vs Direct | Cohen's *d* | *p*-adj |
|---------------|----------------|-------------|-------------|---------|
| Direct | 0.448 (0.047) | — | — | — |
| Planner | 0.405 (0.075) | −0.043 | −0.82 | 0.029 |
| Council | 0.378 (0.072) | −0.070 | −1.40 | 0.003 |
| Extended Prompt | 0.392 (0.046) | −0.056 | −1.51 | 0.003 |

Planning strategies not only fail to improve alignment under the generative pipeline but actively *reduce* it, with large negative effect sizes (*d* = −0.82 to −1.51, all significant after Holm–Bonferroni correction).

#### 5.2.3 Cross-Paradigm Comparison

The contrast between retrieval and generative results is striking: planning is **neutral** under retrieval (|*d*| ≤ 0.19) but **harmful** under generation (*d* = −0.82 to −1.51). This divergence has a clear mechanistic explanation. Under retrieval, planning affects the text generation step, but the subsequent retrieval step is constrained by a fixed embedding index—no matter how well-crafted the prompt, the retrieved media are drawn from the same finite pool. Under generation, the prompt directly controls SDXL image synthesis. Planning modes produce longer, more descriptive prompts that exceed CLIP's 77-token context window, causing truncation. SDXL, conditioned on these truncated embeddings, generates images that diverge from the original prompt's semantic core. The direct mode's concise original prompt remains best aligned with both CLIP and CLAP embedding spaces. This finding suggests that prompt engineering for generative multimodal systems should prioritize semantic density over elaboration.

### 5.3 RQ3: MSCI Correlates with Human Coherence Judgments

**Table 5.** Inter-rater reliability.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICC(3,1) single measures | 0.697 | Moderate (Koo & Li, 2016) |
| ICC(3,k) average measures | 0.873 | Good |
| Krippendorff's α (ordinal) | 0.684 | Acceptable (≥ 0.667; Krippendorff, 2011) |

Pairwise Spearman correlations between raters ranged from ρ = 0.55 to ρ = 0.71 (all *p* < 0.002).

**Table 6.** MSCI–human correlation.

| Metric | Value | *p*-value | 95% CI |
|--------|-------|-----------|--------|
| Spearman's ρ | 0.379 | 0.039 | [0.021, 0.650] |
| Kendall's τ | 0.293 | 0.038 | [0.041, 0.546] |

**Table 7.** Human ratings by condition.

| Condition | Human Mean (SD) | Human Median | MSCI Mean |
|-----------|----------------|--------------|-----------|
| Baseline | 2.70 (1.49) | 2.5 | 0.386 |
| Wrong Image | 2.10 (0.54) | 2.0 | 0.332 |
| Wrong Audio | 2.10 (1.04) | 2.0 | 0.228 |

A Kruskal–Wallis test found no significant difference in human ratings across conditions (*H* = 0.785, *p* = 0.675), likely due to the small per-condition sample size (n = 10).

**Findings.** MSCI demonstrates statistically significant, moderate alignment with human coherence judgments. Both Spearman's ρ and Kendall's τ are significant at α = 0.05, supporting MSCI's validity as an automatic proxy for human coherence evaluation. The moderate magnitude suggests that MSCI captures meaningful aspects of perceived coherence while leaving variance attributable to perceptual factors beyond embedding geometry unexplained.

[Figure 4: Forest plot — all effect sizes across all experiments]
[Figure 5: Scatter plot — MSCI vs. human coherence ratings]

### 5.4 Summary of Findings

**Table 8.** Summary of research questions across both paradigms.

| RQ | Pipeline | Hypothesis | Verdict | Key Statistics |
|----|----------|-----------|---------|----------------|
| RQ1 | Retrieval | MSCI is sensitive to perturbations | **Supported** | *d* = 2.11–3.64, all *p*-adj < 10⁻¹² |
| RQ1 | Generative | MSCI is sensitive to perturbations | **Supported** | *d* = 2.02–4.52, all *p*-adj < 10⁻¹¹ |
| RQ2 | Retrieval | Planning improves alignment | **Not supported** | |*d*| ≤ 0.19, 80% power for *d* ≥ 0.53 |
| RQ2 | Generative | Planning improves alignment | **Not supported** | *d* = −0.82 to −1.51 (planning hurts) |
| RQ3 | Retrieval | MSCI aligns with human judgment | **Supported** | ρ = 0.379, *p* = 0.039 |

---

## 6. Discussion

### 6.1 Audio Channel Dominance in Perturbation Detection

The most striking finding from RQ1 is the asymmetry between audio and image perturbation effects. Wrong-audio substitution produces a mean MSCI decrease of 0.224 (*d* = 3.64), roughly five times larger than wrong-image substitution (Δ = 0.047, *d* = 2.27). The channel decomposition analysis (Figure A.1) reveals the mechanism: when audio is mismatched, the text–audio similarity drops from 0.545 to 0.097, while text–image similarity remains unchanged. In contrast, wrong-image substitution reduces text–image similarity from 0.243 to 0.150, a proportionally smaller change.

We attribute this to two factors. First, CLAP's text–audio embedding space appears more discriminative for environmental domain distinctions: mismatched audio (e.g., urban traffic paired with a nature prompt) produces near-orthogonal embeddings. Second, the image retrieval index, while curated, contains images that may share low-level visual features across domains (e.g., green tones in both nature and urban park scenes), leading to smaller similarity drops when images are mismatched.

This has practical implications: audio coherence may be a more reliable signal of overall multimodal quality than image coherence in environmental scene compositions.

### 6.2 Planning Interacts Differently with Each Pipeline

The most instructive finding of RQ2 is not any single result but the *contrast* between the two paradigms. Under retrieval, planning has no effect (|*d*| ≤ 0.19); under generation, planning is actively harmful (*d* = −0.82 to −1.51). This divergence reveals how pipeline architecture mediates the relationship between prompt engineering and cross-modal alignment.

**Under retrieval,** planning strategies affect the text generation step, but the subsequent retrieval step is constrained by the fixed embedding index. No matter how well-crafted the prompt, the retrieved image and audio are selected from the same finite pool. The channel decomposition (Figure A.1) provides further evidence: text–image similarity (mean = 0.243) is consistently lower than text–audio similarity (mean = 0.545), confirming that the image index is the bottleneck.

**Under generation,** the prompt directly controls SDXL image synthesis. Planning modes produce longer, more descriptive prompts that exceed CLIP's 77-token context window, causing truncation. SDXL, conditioned on these truncated embeddings, generates images that diverge from the original prompt's semantic core. The more elaborate the planning (council > planner > direct), the worse the alignment—a dose-response relationship consistent with the truncation mechanism.

This finding has practical implications: prompt engineering strategies must be tailored to the pipeline architecture. Retrieval systems benefit most from expanding the corpus and refining embeddings, while generative systems benefit from concise, semantically dense prompts that respect token limits.

### 6.3 Moderate but Significant Human Alignment

The RQ3 correlation of ρ = 0.379 is moderate but meaningful. For context, CLIPScore correlations with human judgments in image captioning tasks typically range from ρ = 0.35 to ρ = 0.55 (Hessel et al., 2021). Our result falls within this range, especially given the added complexity of evaluating three modalities simultaneously.

The inter-rater reliability metrics (ICC(3,1) = 0.70, Krippendorff's α = 0.68) indicate that multimodal coherence is a genuinely difficult construct for humans to assess, with moderate agreement across raters. This suggests that some variance in the MSCI–human correlation is attributable to human disagreement rather than metric failure.

The by-condition analysis shows that human ratings trend in the expected direction (baseline: 2.70 > wrong-image/audio: 2.10), but the between-condition differences did not reach statistical significance due to the small per-condition sample size (n = 10). A larger human evaluation study would likely resolve this.

### 6.4 Skip-Text vs. Full Pipeline

The full-pipeline baseline MSCI (0.348) is lower than the skip-text baseline (0.394), a decrease of 0.046. This reflects the variance introduced by LLM text generation: the model occasionally rephrases prompts in ways that reduce CLIP/CLAP similarity relative to the original prompt text. Importantly, both experimental modes yield identical conclusions for RQ1, confirming that the perturbation effect is not an artifact of the experimental protocol.

---

## 7. Limitations

**Dataset size.** The evaluation dataset (57 images, 104 audio clips) is modest compared to large-scale benchmarks. Domain coverage is uneven, with water underrepresented in images (12 of 57). A larger, more balanced dataset would strengthen generalizability.

**Number of raters.** Three raters meet the minimum threshold for inter-rater reliability computation but provide limited statistical power for between-condition comparisons. Five or more raters would yield tighter confidence intervals and more robust consensus ratings.

**Domain scope.** All prompts describe environmental scenes spanning three domains (nature, urban, water). Generalization to other content types (e.g., indoor scenes, abstract concepts, human activities) remains untested.

**Embedding ceiling.** MSCI is bounded by the quality of CLIP and CLAP embeddings, which were trained on broad internet data and may not capture domain-specific nuances. Fine-tuning on environmental scene data could improve discriminative power.

**No cross-space bridge.** CLIP and CLAP occupy separate embedding spaces. Image–audio coherence cannot be assessed directly without a trained cross-space projection. MSCI therefore captures only text-mediated coherence, missing potential image–audio misalignments that text similarity does not detect.

**Construct validity.** While MSCI correlates with human judgments, the moderate magnitude (ρ = 0.379) indicates that automatic embedding-based similarity captures only part of the perceptual coherence construct. Factors such as aesthetic quality, emotional tone, temporal congruence, and cultural context likely contribute to human coherence perception but are not represented in CLIP/CLAP embedding spaces.

**LLM variability.** The full-pipeline mode uses a local LLM (Ollama) whose text generation introduces variance. Different LLM backends could produce different absolute MSCI values, though the relative ranking of conditions should be preserved.

---

## 8. Conclusion

We presented MSCI, a composite metric for evaluating cross-modal semantic coherence in multimodal generation. Through three systematic experiments and generative validations, we established that:

1. **MSCI is statistically sensitive to cross-modal perturbations** under controlled conditions, with large effect sizes (*d* = 2.11–3.64) that are robust across skip-text, full-pipeline, and hybrid generative modes (*d* = 2.02–4.52).

2. **Structured planning does not improve cross-modal alignment** under either retrieval-based or generative pipelines. Under retrieval, a well-powered null result (80% power for *d* ≥ 0.53) identifies the retrieval index as the bottleneck. Under generation, planning actively reduces alignment (*d* = −0.82 to −1.51) due to CLIP's 77-token context limit truncating verbose planned prompts.

3. **MSCI correlates with human coherence judgments** (ρ = 0.379, *p* = 0.039), establishing it as a viable automatic proxy for human evaluation of multimodal coherence.

MSCI does not claim to measure semantic truth or objective meaning, but rather alignment within the representational geometry of pre-trained multimodal embedding models. Its value lies in providing a reproducible, scalable proxy that tracks human perception at moderate fidelity—sufficient for automated evaluation and system comparison, while acknowledging that embedding-based similarity captures only part of the perceptual coherence construct.

These findings have practical implications for multimodal system design: investment in retrieval corpus quality and embedding fine-tuning is likely to yield greater coherence improvements than investment in generation-side planning strategies.

### Reproducibility

All experiments use fixed random seeds [42, 123, 7], local-only inference (no cloud API calls), and deterministic retrieval. The full codebase—including raw results (JSON), analysis scripts, figure generation, and the 30-prompt evaluation set—is publicly available. All reported statistics can be reproduced from the raw data using the provided analysis pipeline (`scripts/analyze_results.py`, `scripts/analyze_rq3.py`).

### Future Work

Future research could address the limitations identified above by: (a) expanding the dataset to additional domains and larger scale, (b) training the cross-space bridge between CLIP and CLAP—whose architecture is implemented and integrated—to enable direct image–audio coherence assessment (si_a) and activate the full MSCI formula, (c) investigating learned MSCI weights through correlation with larger-scale human evaluations, and (d) extending the framework to video and other temporal modalities.

---

## References

Baltrusaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423–443.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235, 11733–11763.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., & Choi, Y. (2021). CLIPScore: A reference-free evaluation metric for image captioning. *Proceedings of EMNLP 2021*, 7514–7528.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *Advances in Neural Information Processing Systems*, 30, 6626–6637.

Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark, P., & Sabharwal, A. (2023). Decomposed prompting: A modular approach for solving complex tasks. *Proceedings of ICLR 2023*.

Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155–163.

Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. *Departmental Papers (ASC)*, 43.

Lee, T., Yasunaga, M., Meng, C., Mai, Y., Park, J. S., Gupta, A., Zhang, Y., Narayanan, D., Teufel, H. B., Bellagente, M., Kang, M., Park, T., Leskovec, J., Zhu, J.-Y., Li, F.-F., Wu, J., Ermon, S., & Liang, P. (2023). Holistic evaluation of text-to-image models. *Advances in Neural Information Processing Systems*, 36.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., & Rombach, R. (2024). SDXL: Improving latent diffusion models for high-resolution image synthesis. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, 8748–8763.

Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001). Perceptual evaluation of speech quality (PESQ)—a new method for speech quality assessment of telephone networks and codecs. *Proceedings of IEEE ICASSP 2001*, 749–752.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. *Advances in Neural Information Processing Systems*, 29, 2226–2234.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824–24837.

Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. *Proceedings of ICASSP 2023*, 1–5.

Yariv, G., Gat, I., Benaim, S., Wolf, L., Schwartz, I., & Adi, Y. (2024). Diverse and aligned audio-to-video generation via text-to-video model adaptation. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(7), 6639–6647.

---

## Appendix A: Prompt Set

| ID | Domain | Prompt Text |
|----|--------|-------------|
| nat_01 | Nature | A peaceful forest at dawn with birdsong and morning mist |
| nat_02 | Nature | A mountain meadow with wildflowers swaying in the wind |
| nat_03 | Nature | A dense jungle with exotic birds calling from the canopy |
| nat_04 | Nature | A foggy morning in the countryside with distant church bells |
| nat_05 | Nature | A sunlit garden with buzzing bees and rustling leaves |
| nat_06 | Nature | A snowy mountain peak under a clear blue winter sky |
| nat_07 | Nature | A field of golden wheat under a warm summer sunset |
| nat_08 | Nature | A quiet woodland path with autumn leaves crunching underfoot |
| urb_01 | Urban | A bustling city street at night with neon lights and traffic |
| urb_02 | Urban | A rainy day in a European city with cobblestone streets |
| urb_03 | Urban | A crowded marketplace with vendors shouting and music playing |
| urb_04 | Urban | A quiet alley in an old town with distant footsteps echoing |
| urb_05 | Urban | A rooftop view of a modern skyline at golden hour |
| urb_06 | Urban | A subway station with trains arriving and commuters rushing |
| urb_07 | Urban | A cafe terrace on a busy boulevard with clinking glasses |
| urb_08 | Urban | An empty parking lot under flickering streetlights at midnight |
| wat_01 | Water | Ocean waves crashing on a sandy beach at sunset |
| wat_02 | Water | A calm lake reflecting snow-capped mountains at dawn |
| wat_03 | Water | A tropical island with turquoise water and palm trees |
| wat_04 | Water | A river flowing through a rocky canyon with rapids |
| wat_05 | Water | Rain falling on a pond with ripples spreading across the surface |
| wat_06 | Water | A fishing boat anchored in a misty harbor at early morning |
| wat_07 | Water | A waterfall cascading into a lush green pool below |
| mix_01 | Mixed | A lighthouse on a cliff during a thunderstorm at night |
| mix_02 | Mixed | Children playing in a park fountain on a hot summer day |
| mix_03 | Mixed | A desert landscape with sand dunes under a blazing sun |
| mix_04 | Mixed | A train crossing a bridge over a deep valley at dusk |
| mix_05 | Mixed | A bonfire on a beach with waves and guitar music at night |
| mix_06 | Mixed | A hot air balloon floating over a patchwork of farm fields |
| mix_07 | Mixed | A stone bridge over a stream in an ancient village at twilight |

## Appendix B: Figure Index

### Main Figures

| Figure | Description | File |
|--------|-------------|------|
| Fig. 1 | Raincloud plot: MSCI distribution by perturbation condition (RQ1) | `fig1_rq1_raincloud_f.pdf` |
| Fig. 2 | Paired slope plot: per-prompt MSCI trajectories (RQ1) | `fig2_rq1_paired_slopes_f.pdf` |
| Fig. 3 | Gardner-Altman estimation plots: planning modes vs. direct (RQ2) | `fig3_rq2_estimation_f.pdf` |
| Fig. 4 | Forest plot: all effect sizes with 95% CIs (RQ1, RQ1-gen, RQ2, RQ2-gen) | `fig4_forest_plot_f.pdf` |
| Fig. 5 | Scatter plot: MSCI vs. human coherence ratings (RQ3) | `fig11_rq3_scatter_f.pdf` |
| Fig. 6 | Retrieval vs generative: MSCI comparison and effect sizes (RQ1-gen) | `fig13_retrieval_vs_generation_f.pdf` |

### Supplementary Figures (Appendix)

| Figure | Description | File |
|--------|-------------|------|
| Fig. A.1 | Channel decomposition: text–image vs. text–audio contributions | `fig5_rq1_channel_decomposition_f.pdf` |
| Fig. A.2 | Heatmap: MSCI by domain × condition | `fig6_rq1_domain_heatmap_f.pdf` |
| Fig. A.3 | Power curve: detectable effect sizes at N = 30 | `fig7_rq2_power_curve_f.pdf` |
| Fig. A.4 | Bootstrap distributions of mean MSCI differences | `fig8_rq1_bootstrap_f.pdf` |
| Fig. A.5 | Robustness: skip-text vs. full-pipeline comparison | `fig9_rq1_robustness_f.pdf` |
| Fig. A.6 | Seed stability: within-prompt variance across random seeds | `fig10_seed_stability_f.pdf` |
| Fig. A.7 | Box plot: human ratings by condition (RQ3) | `fig12_rq3_conditions_f.pdf` |

## Appendix C: Supplementary Figures

[Figure A.1: Channel decomposition — text–image vs. text–audio similarity contributions by condition]

[Figure A.2: Heatmap — mean MSCI by prompt domain × perturbation condition]

[Figure A.3: Power curve — statistical power at N = 30 for varying effect sizes]

[Figure A.4: Bootstrap distributions of mean MSCI differences (10,000 resamples)]

[Figure A.5: Robustness — skip-text vs. full-pipeline MSCI comparison]

[Figure A.6: Seed stability — within-prompt variance across random seeds]

[Figure A.7: Box plot — human coherence ratings by perturbation condition]

## Appendix D: Reproducibility

**Software:** Python 3.11, PyTorch 2.x, CLIP (ViT-B/32, OpenAI), CLAP (HTSAT-unfused, LAION), Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0), Ollama (local LLM inference), SciPy 1.16, NumPy 2.3, Matplotlib 3.10, Seaborn 0.13, Diffusers (HuggingFace).

**Random seeds:** [42, 123, 7] for all experiments.

**Hardware:** Apple Silicon (macOS), local inference only (no cloud API calls).

**Total compute:** RQ1 skip-text: 90.5 min (270 runs). RQ1 full: 121.8 min (270 runs). RQ2: 222.8 min (360 runs). RQ1-gen hybrid: SDXL generation ~123 min (90 runs) + evaluation ~20 min. RQ2-gen hybrid: planning ~7 min + SDXL generation ~105 min (40 runs) + evaluation ~14 min. Total: ~13.1 hours.

**Data availability:** All prompts, raw results (JSON), analysis outputs, and figure generation scripts are included in the repository.
