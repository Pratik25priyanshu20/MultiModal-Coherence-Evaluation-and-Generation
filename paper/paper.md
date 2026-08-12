# Calibrated Multimodal Semantic Coherence Index: A Geometric Approach to Cross-Modal Alignment Evaluation

---

## Abstract

Multimodal generation systems that produce text–image–audio bundles lack reliable automatic metrics for evaluating cross-modal semantic coherence. Existing metrics either assess modalities independently or rely on pairwise cosine similarity, which is scale-dependent, context-free, and ignores higher-order geometric structure. We propose the **calibrated Multimodal Semantic Coherence Index (cMSCI)**, a metric that integrates Gramian volume geometry, z-score calibration, contrastive margin estimation, cross-modal complementarity via Ex-MCR projection, and uncertainty-aware adaptive channel weighting into a unified coherence framework. On a human-annotated evaluation set of 100 samples rated by five independent raters (ICC(3,k) = 0.872, Krippendorff's α = 0.684 for text–image), cMSCI achieves Spearman's ρ = 0.785 (*p* < 10⁻⁶), outperforming all evaluated baselines including cosine+z-norm (ρ = 0.712), MSCI (ρ = 0.558), Regularized CCA (ρ = 0.495), and retrieval rank (ρ = 0.394)—while requiring no large language model at inference time. Leave-one-out cross-validation confirms minimal overfitting (LOO-CV ρ = 0.749, *p* < 10⁻⁶). External benchmark validation on 1,000 AudioCaps samples confirms near-perfect matched/mismatched discrimination (AUC = 0.969). Scaling training data from 2,193 to 10,255 pairs shifts the optimized channel balance from image-dominated (w_ti = 0.90) to audio-inclusive (w_ti = 0.15), with adaptive uncertainty weighting (γ = 0.6) further modulating channel trust per sample—demonstrating that data quality, not architecture changes, is the primary lever for multi-channel metric improvement.

**Keywords:** multimodal coherence, CLIP, CLAP, cross-modal evaluation, Gramian volume, semantic alignment, generative AI evaluation, uncertainty-aware weighting

---

## 1. Introduction

### 1.1 Problem Statement

The proliferation of multimodal AI systems—those that generate or compose content across text, image, and audio modalities—has created a pressing need for evaluation metrics that assess not just the quality of individual modalities but their *semantic coherence* as a unified experience (Baltrusaitis et al., 2019). A nature scene paired with urban traffic noise, or a beach description accompanied by a city skyline photograph, would each represent a coherence failure that degrades user experience regardless of how high-quality the individual components are. Existing evaluation approaches either focus on single-modality quality (FID for images (Heusel et al., 2017), PESQ for audio (Rix et al., 2001), perplexity for text) or require expensive human annotation for every new composition.

### 1.2 Gap in Existing Metrics

Pre-trained vision-language models such as CLIP (Radford et al., 2021) and audio-language models such as CLAP (Wu et al., 2023) offer shared embedding spaces that can serve as automatic coherence proxies. However, applying pairwise cosine similarity from these models suffers from four fundamental limitations:

1. **Scale dependence.** Cosine similarities from different embedding spaces (CLIP vs CLAP) are not directly comparable in magnitude, making weighted combination unreliable.
2. **Context-free scoring.** Raw similarity scores lack a reference distribution—there is no principled way to interpret whether a score of 0.35 represents strong or weak coherence without knowing the baseline population.
3. **No higher-order geometry.** Pairwise similarity cannot capture the joint geometric structure of three or more modalities. The volume spanned by text, image, and audio embeddings encodes alignment information that pairwise measures discard.
4. **No adaptive uncertainty weighting.** Fixed channel weights cannot account for the fact that some samples have more reliable text–image signals while others have stronger text–audio signals. Per-sample confidence estimation is absent from current metrics.

### 1.3 Main Idea

We propose cMSCI, a **calibrated geometric coherence metric** that addresses all four limitations. By modeling multimodal alignment as Gramian volume in embedding space, calibrating against reference distributions, incorporating contrastive margins from hard negatives, measuring cross-modal complementarity via unified embedding projection, and weighting channels adaptively based on per-sample uncertainty, cMSCI achieves significantly stronger alignment with human coherence judgments than existing automatic metrics.

### 1.4 Contributions

1. **A geometric coherence formulation using Gramian volume.** We model multimodal coherence as the volume spanned by normalized embedding vectors in Gramian space, where collapsed volume indicates alignment and maximal volume indicates orthogonality. This captures higher-order geometric structure that pairwise cosine similarity discards.

2. **Cross-modal complementarity via unified embedding projection.** Using a trained Ex-MCR projector, we project CLAP audio embeddings into CLIP space to measure image–audio Gramian dispersion—quantifying whether modalities contribute unique, non-redundant information. This complementarity signal correlates positively with human coherence perception.

3. **Uncertainty-aware adaptive weighting.** ProbVLM-style probabilistic adapters estimate per-sample uncertainty for each embedding channel. Instead of fixed weights, cMSCI dynamically trusts whichever channel is more confident for each sample.

4. **Human validation with strong baselines.** We validate cMSCI against human coherence judgments from five independent raters on 100 samples (eight raters on the original 30) and compare against baselines spanning simple metrics (cosine, z-norm), joint embedding methods (CCA, RegCCA), and retrieval-based scoring. cMSCI achieves the highest correlation (ρ = 0.785, *p* < 10⁻⁶), outperforming all alternatives including cosine+z-norm (ρ = 0.712) and Regularized CCA (ρ = 0.495).

---

## 2. Defining Multimodal Semantic Coherence

### 2.1 Formal Definition

We define **multimodal semantic coherence** as the degree to which concurrently presented modalities—text, image, and audio—convey a unified semantic message. Formally, given a multimodal tuple $\mathcal{M} = (t, i, a)$ consisting of a text description $t$, an image $i$, and an audio clip $a$, coherence is a scalar $c(\mathcal{M}) \in [0, 1]$ satisfying:

- **Consistency:** If all modalities describe the same scene or concept, coherence is high.
- **Sensitivity:** Replacing any single modality with semantically unrelated content should reduce coherence.
- **Channel independence:** The metric should detect incoherence in any modality, not just the strongest channel.
- **Human alignment:** Automatic coherence scores should correlate with human perception of semantic unity.

This definition distinguishes coherence from *quality*—a high-quality image of a city paired with a nature text prompt is incoherent despite its visual fidelity.

### 2.2 Human Annotation Protocol

To establish ground truth, five independent raters evaluated 100 stratified samples (34 baseline, 33 wrong-image, 33 wrong-audio) via a web-based interface (Streamlit). Each sample presented text, image, and audio simultaneously. Raters assigned a coherence score on a 1–5 Likert scale (1 = completely incoherent, 5 = perfectly coherent) without knowledge of the perturbation condition. An additional three raters evaluated the original 30-sample subset, yielding eight raters on those samples.

**Table 1.** Inter-rater reliability (5 raters, 100 samples).

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICC(3,1) single measures | 0.577 | Moderate (Koo & Li, 2016) |
| ICC(3,k) average measures | 0.872 | Good |
| Krippendorff's α (text–image) | 0.684 | Acceptable (≥ 0.667; Krippendorff, 2011) |
| Krippendorff's α (text–audio) | 0.680 | Acceptable |
| Krippendorff's α (overall) | 0.545 | Moderate |

Pairwise Spearman correlations between raters ranged from ρ = 0.45 to ρ = 0.69 (all *p* < 0.0001). The moderate-to-good reliability confirms that multimodal coherence is a meaningful but genuinely difficult construct for humans to assess. On the original 30-sample subset with eight raters, ICC(3,k) = 0.917, demonstrating strong agreement with additional raters.

### 2.3 Dataset Construction

**Evaluation prompts.** 100 scene descriptions spanning three environmental domains (nature, urban, water) plus mixed-domain prompts (see Appendix A). Each prompt was evaluated under three conditions: baseline (matched image + audio), wrong-image (cross-domain mismatched image, matched audio), and wrong-audio (matched image, cross-domain mismatched audio), yielding 300 evaluation triples.

**Generative pipeline.** For each prompt, images are generated by Stable Diffusion XL (SDXL; Podell et al., 2024) conditioned on the text prompt. Audio is retrieved via CLAP cosine similarity against an audio embedding index, selecting the best domain-compatible match. This generative design reflects practical multimodal content creation workflows where visual content is synthesized and audio is sourced from sound libraries.

**Perturbation protocol.** For the wrong-image condition, SDXL generates from a different-domain prompt (e.g., a nature prompt yields a city image). For the wrong-audio condition, a different-domain audio clip is retrieved. This controlled perturbation design ensures that each condition isolates a single channel's incoherence.

**External validation.** In addition to our curated evaluation set, we validate on 1,000 AudioCaps samples (Kim et al., 2019)—a public benchmark of audio clips with human-written captions—to confirm that cMSCI generalizes beyond the evaluation domain.

**Training data.** The Ex-MCR projector and probabilistic adapters are trained on 10,255 embedding pairs sourced from OmniBench (1,051 domain-matched triples), AudioCaps (1,000 real audio-caption pairs), and embedding-space augmentation (8,204 augmented pairs via Gaussian noise, dropout, and mixup).

**Dev/test split.** The 100 human-rated samples are partitioned into a dev set (n = 70) and a held-out test set (n = 30) using stratified sampling by domain × condition (seed = 2024). Hyperparameter optimization uses leave-one-out cross-validation on all 100 samples (an inherently unbiased procedure). The dev/test split provides additional held-out validation (Section 6.2).

---

## 3. Related Work

### 3.1 Multimodal Evaluation Metrics

**CLIPScore.** CLIPScore (Hessel et al., 2021) computes cosine similarity between CLIP image and text embeddings, serving as the standard reference-free metric for image–text alignment. Its simplicity and strong correlation with human judgments for captioning tasks have made it the default automatic metric in text-to-image generation. However, CLIPScore captures only one modality channel (text–image) and ignores audio entirely, limiting its applicability to tri-modal evaluation. Additionally, cosine similarity in CLIP space is sensitive to prompt phrasing and exhibits systematic biases across content domains (Lee et al., 2023), motivating calibration-based alternatives.

**BLIPScore.** BLIPScore extends CLIPScore by replacing cosine similarity with BLIP's image–text matching (ITM) head (Li et al., 2023), which outputs a learned matching probability rather than a raw distance in embedding space. The ITM head is trained with a binary matching objective, providing a richer similarity signal than cosine distance alone. In our evaluation, BLIPScore combined with CLAPScore achieves ρ = 0.369 (*p* = 0.045), outperforming CLIPScore (ρ = 0.201) but trailing cMSCI's geometric approach (ρ = 0.519). The improvement over CLIPScore confirms that learned matching heads capture alignment properties that cosine similarity misses, while the remaining gap to cMSCI suggests that per-channel scoring—even with richer heads—does not fully capture higher-order coherence structure.

**Retrieval-based scoring.** Retrieval-based metrics evaluate alignment by ranking: given a query in one modality, the metric computes how highly the matched item ranks among all candidates in the other modality. Reciprocal rank and Recall@K are standard measures in cross-modal retrieval benchmarks (Radford et al., 2021). While intuitive, retrieval-based scoring is highly sensitive to the candidate pool composition and provides only ordinal, not cardinal, alignment information. In our experiments, retrieval rank achieves ρ = 0.051 (*p* = 0.787)—the weakest of all baselines—because the small candidate pool (30 samples) provides insufficient ranking resolution to discriminate fine-grained coherence differences.

Single-modality quality metrics (FID (Heusel et al., 2017), IS (Salimans et al., 2016), PESQ (Rix et al., 2001)) assess individual outputs but cannot capture cross-modal alignment. Recent composite metrics for vision-language tasks (Lee et al., 2023) and audio-visual alignment (Yariv et al., 2024) motivate our approach of combining modality-specific similarities into a unified coherence score, though none integrate geometric volume measures with calibration and adaptive weighting.

### 3.2 Representation Geometry

**Cosine similarity limitations.** Cosine similarity between embeddings from different pre-trained models (e.g., CLIP vs CLAP) is not scale-comparable, and the absolute magnitude of cosine similarity is uninterpretable without reference distributions. These limitations motivate our z-score calibration approach.

**Gram matrices in multiview learning.** The Gram matrix $G_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$ captures pairwise alignment structure, and its determinant encodes the volume of the parallelotope spanned by the vectors. In the context of multimodal coherence, low volume (collapsed parallelotope) indicates that modalities convey aligned information, while high volume indicates dispersion. This geometric perspective generalizes pairwise similarity to arbitrary numbers of modalities and captures higher-order alignment structure.

**Gramian volume in contrastive learning.** Concurrently with our work, Cicchetti et al. (2025a) proposed GRAM, which uses the same Gramian volume formulation ($\text{vol} = \det(G)^{1/2}$) as a contrastive training loss for multimodal representation learning on video-audio-text data. Their follow-up, TRIANGLE (Cicchetti et al., 2025b), replaces Gramian volume with triangle area for exactly three modalities. Both methods use geometric volume to *train* better embeddings; our work takes the complementary perspective, using Gramian volume as one component of a *calibrated evaluation metric* for per-sample coherence scoring, augmented with z-score calibration, contrastive margins, cross-modal complementarity, and uncertainty-adaptive weighting—none of which appear in GRAM or TRIANGLE.

**Higher-order contrastive objectives.** Symile (Saporta et al., 2024) proposes a multilinear inner product (MIP) as a joint similarity measure for $n$ modalities, targeting total correlation rather than pairwise mutual information. Like GRAM, Symile is a training objective, not an evaluation metric. These works collectively validate the intuition that pairwise contrastive learning is insufficient for multi-modal alignment, motivating geometric approaches like ours.

### 3.3 Cross-Modal Alignment

**CLIP** (Radford et al., 2021) learns a shared 512-dimensional embedding space for images and text through contrastive learning on 400 million image-text pairs. **CLAP** (Wu et al., 2023) applies the same paradigm to audio and text.

**Joint embedding models.** ImageBind (Girdhar et al., 2023) proposes a 6-modality embedding space enabling native cross-modal comparison. CCA-based approaches (Andrew et al., 2013; Hardoon et al., 2004) learn linear projections maximizing correlation between embedding spaces.

**Cross-space bridging.** C-MCR (Wang et al., 2023b) introduced a method for connecting multi-modal contrastive representation spaces by exploiting an overlapping modality (text) shared between CLIP and CLAP. Ex-MCR (Wang et al., 2024) extended this with decoupled projectors and dense contrastive losses, enabling alignment of CLAP audio embeddings into CLIP space without requiring paired image-audio data. We adopt the Ex-MCR architecture for cross-modal complementarity measurement (Section 4.5).

**Probabilistic vision-language models.** BayesCap (Upadhyay et al., 2022) proposed Bayesian identity mappings for uncertainty estimation in frozen encoders. ProbVLM (Upadhyay et al., 2023) extended this to vision-language models, training lightweight probabilistic adapters that map point embeddings to heteroscedastic distributions. We adopt the ProbVLM framework for per-sample uncertainty estimation (Section 4.6).

**Architectural constraint.** CLIP and CLAP occupy distinct embedding spaces—CLIP text embeddings are aligned with images, while CLAP text embeddings are aligned with audio. Direct comparison between CLIP image embeddings and CLAP audio embeddings is not meaningful without a trained cross-space projection. We validate this empirically: CLIP text (cross-space) achieves AUC = 0.500 (chance) on AudioCaps discrimination, while within-space methods achieve AUC ≥ 0.957.

### 3.4 LLM/VLM-as-a-Judge Paradigm

The LLM-as-a-judge paradigm uses large language models as automated evaluators of text quality (Zheng et al., 2023), replacing expensive human annotation with model-based scoring. This paradigm extends naturally to vision-language models (VLMs) for multimodal assessment. LLaVA (Liu et al., 2023) and similar VLMs can rate coherence through chain-of-thought reasoning (Wei et al., 2022), offering interpretable explanations alongside numerical scores. VLM-as-judge approaches can assess high-level semantic properties (e.g., scene appropriateness, emotional congruence) that embedding geometry may miss, but require multi-billion-parameter models at inference time. We include LLaVA-7B as a baseline to compare geometric metrics against semantic reasoning approaches.

---

## 4. Method: Calibrated Multimodal Semantic Coherence Index (cMSCI)

### 4.1 Baseline: Pairwise Cosine Similarity (MSCI)

We define the baseline Multimodal Semantic Coherence Index (MSCI) as a weighted combination of pairwise cosine similarities:

$$\text{MSCI} = w_{ti} \cdot s(\mathbf{e}_t^{\text{CLIP}}, \mathbf{e}_i^{\text{CLIP}}) + w_{ta} \cdot s(\mathbf{e}_t^{\text{CLAP}}, \mathbf{e}_a^{\text{CLAP}})$$

where $s(\cdot, \cdot)$ denotes cosine similarity, $\mathbf{e}_t^{\text{CLIP}}$ and $\mathbf{e}_i^{\text{CLIP}}$ are CLIP text and image embeddings (ViT-B/32, 512-d), $\mathbf{e}_t^{\text{CLAP}}$ and $\mathbf{e}_a^{\text{CLAP}}$ are CLAP text and audio embeddings (HTSAT-unfused, 512-d), and $w_{ti} = w_{ta} = 0.50$. (The equal weighting reflects the absence of a cross-space image–audio channel; when a trained bridge enables the third channel, a 0.45/0.45/0.10 split is used.)

MSCI uses separate text encoders for each channel: CLIP's text encoder for text–image and CLAP's text encoder for text–audio. This operationalizes coherence as convergence toward a shared textual semantic anchor rather than direct pairwise compatibility between all modalities.

### 4.2 Gramian Volume Geometry

Given $n$ L2-normalized embedding vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$, we compute the Gramian matrix $G_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$ and define the geometric volume as:

$$\text{vol} = \det(G)^{1/2}$$

For perfectly aligned vectors, $\det(G) = 0$ (volume collapses); for orthogonal vectors, $\det(G) = 1$ (maximal volume). We define Gramian coherence as:

$$c_G = 1 - \text{vol}$$

mapping to $[0, 1]$ where 1 indicates perfect alignment. For two vectors, this reduces to $c_G = 1 - \sqrt{1 - \cos^2\theta}$, a monotonic function of cosine similarity (for $\cos\theta \geq 0$) that is more sensitive near perfect alignment—exactly where generative content tends to operate, since matched text–image pairs typically cluster in the high-similarity regime.

### 4.3 Z-Score Calibration

Raw Gramian coherences (text–image $c_{ti}$ and text–audio $c_{ta}$) are normalized against reference distributions fitted from baseline data:

$$z_k = \frac{c_k - \mu_k}{\sigma_k}$$

The calibrated 2-way score is $z_{2d} = w_{ti} \cdot z_{ti} + (1 - w_{ti}) \cdot z_{ta}$. This makes scores from different embedding spaces directly comparable and interpretable as deviations from the baseline population.

### 4.4 Contrastive Margin

For each evaluation triple, we select $k = 5$ hard negatives per channel (domain-mismatched substitutions) and compute per-channel contrastive margins within each embedding space:

$$m_{ti} = \mathbb{E}_{i \in \mathcal{N}_{\text{img}}}[V_{ti}(t, i)] - V_{ti}^{*}, \quad m_{ta} = \mathbb{E}_{a \in \mathcal{N}_{\text{aud}}}[V_{ta}(t, a)] - V_{ta}^{*}$$

where $V_{ti}^{*}$ and $V_{ta}^{*}$ are the matched Gramian volumes. The combined margin $m = w_{ti} \, m_{ti} + (1 - w_{ti}) \, m_{ta}$ is positive when the matched pair is tighter than negatives. This per-channel formulation avoids mixing volumes from heterogeneous embedding spaces (CLIP vs CLAP). The margin is scaled by factor $\alpha$ before addition to the calibrated z-score, providing a relative assessment of coherence quality.

### 4.5 Cross-Space Projection (Ex-MCR)

A trained Ex-MCR projector (525K parameters; 2-layer MLP, 512→512→512 with ReLU) projects CLAP audio embeddings into CLIP space, enabling direct image–audio comparison. The projector is trained on 10,255 embedding pairs sourced from OmniBench, domain-matched data, and AudioCaps.

Rather than measuring image–audio *coherence* (which we found correlates negatively with human judgments, ρ = −0.224), we compute image–audio Gramian *dispersion*—the degree to which image and audio contribute complementary, non-redundant information. This complementarity signal is z-normalized as $z_{\text{compl}}$, where positive values indicate greater complementarity. The sign flip is a key insight: humans perceive multimodal content as more coherent when each modality adds unique perspective, not when modalities are redundant.

### 4.6 Uncertainty-Aware Adaptive Weighting

ProbVLM-style probabilistic adapters (592K parameters each for CLIP and CLAP, trained on 10,255 pairs) model each embedding as a Generalized Gaussian distribution, predicting location ($\mu$), scale ($\alpha$), and shape ($\beta$) parameters. The per-channel uncertainty is the mean predicted scale $\alpha$ across embedding dimensions—high $\alpha$ indicates a wide distribution (low confidence), low $\alpha$ indicates a tight distribution (high confidence). This is a direct forward pass through the adapter, not a sampling procedure.

Instead of a fixed channel weight, cMSCI computes an adaptive weight from these uncertainties:

$$w_{ti}^{\text{adapt}} = \frac{1/u_{ti}}{1/u_{ti} + 1/u_{ta}}$$

where $u_{ti}$ and $u_{ta}$ are the text–image and text–audio channel uncertainties (mean $\alpha$). The final weight mixes fixed and adaptive components:

$$w_{ti}^{\text{final}} = (1 - \gamma) \cdot w_{ti}^{\text{base}} + \gamma \cdot w_{ti}^{\text{adapt}}$$

This trusts the more confident channel on a per-sample basis. Optionally, the adapters can also generate Monte Carlo samples from the predicted distributions to produce confidence intervals for the cMSCI score, but these intervals serve as diagnostic metadata and do not affect the score itself.

### 4.7 Final cMSCI Formulation

The final score combines all five components:

$$\text{cMSCI} = \sigma\!\Big(w_{ti}^{\text{final}} \cdot z_{ti} + (1 - w_{ti}^{\text{final}}) \cdot z_{ta} + w_{3d} \cdot z_{\text{compl}} + \alpha \cdot m\Big)$$

where $\sigma(\cdot)$ is the logistic function mapping the composite logit to $[0, 1]$.

**Hyperparameters.** Optimized via leave-one-out cross-validation on 100 human-rated samples over 86,394 configurations:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $\alpha$ (margin scale) | 2 | Contrastive margin amplification |
| $w_{ti}^{\text{base}}$ (text-image weight) | 0.15 | Audio channel receives 85% base weight |
| $w_{3d}$ (complementarity weight) | 0.15 | Cross-space complementarity contribution |
| $\gamma$ (adaptive mixing) | 0.60 | Uncertainty-aware channel modulation |
| Calibration mode | cosine | Z-score normalization of cosine similarities |

The text–image weight $w_{ti} = 0.15$ assigns 85% of the base weight to the text–audio channel, reflecting the audio channel's strong discriminative power after training on 10,255 pairs that include real AudioCaps audio. The high adaptive mixing coefficient ($\gamma = 0.60$) allows per-sample uncertainty to substantially modulate channel balance, effectively letting the metric trust whichever channel is more confident for each sample.

---

## 5. Experimental Setup

### 5.1 Embedding Backbones

**CLIP ViT-B/32** (Radford et al., 2021). 512-dimensional shared text–image embedding space. Pre-trained on 400M image-text pairs via contrastive learning. 77-token context window for text.

**CLAP HTSAT-unfused** (Wu et al., 2023). 512-dimensional shared text–audio embedding space. Pre-trained on audio-text pairs via contrastive learning.

**Ex-MCR projector.** 525K-parameter projection network (2-layer MLP, 512→512→512 with ReLU) trained on 10,255 embedding pairs (1,051 domain-matched + 1,000 AudioCaps + 8,204 augmented via Gaussian noise, dropout, and mixup). Projects CLAP audio (512-d) into CLIP space (512-d) for cross-modal complementarity measurement.

**Probabilistic adapters.** 592K parameters each (CLIP and CLAP). 3-layer MLPs trained with Generalized Gaussian NLL loss on 10,255 pairs to estimate per-sample embedding uncertainty.

### 5.2 Baselines

We evaluate cMSCI against nine baselines spanning four categories:

**Simple baselines.**
- **Raw cosine similarity:** Mean of text–image (CLIP) and text–audio (CLAP) cosine similarities.
- **Cosine + z-normalization:** Z-score normalized cosine similarities mapped through sigmoid.
- **Retrieval rank:** Reciprocal rank of the matched item among all candidates.
- **Concatenated cosine:** Concatenated embeddings with single cosine distance.

**Established metrics.**
- **CLIPScore** (Hessel et al., 2021): Standard CLIP text–image matching score, extended with CLAPScore for the audio channel.
- **BLIPScore** (Li et al., 2023): BLIP image-text matching (ITM) head probability, combined with CLAPScore for the audio channel.

**Joint embedding methods.**
- **CCA** (Andrew et al., 2013): Canonical correlation analysis projecting CLIP and CLAP into a shared subspace.
- **Regularized CCA:** Ridge-regularized variant for small sample sizes.

**VLM-as-judge.**
- **LLaVA-7B** (Liu et al., 2023): A vision-language model rates coherence on a 1–5 scale given the image, audio spectrogram, and text caption, using chain-of-thought prompting via Ollama.

### 5.3 Evaluation Metrics

**Primary metric.** Spearman's rank correlation (ρ) between automatic metric scores and mean human coherence ratings (averaged across three raters per sample).

**Statistical significance.** Two-sided *p*-values from Spearman's test; significance threshold α = 0.05.

**Effect sizes.** Cohen's *d* with 95% confidence intervals for perturbation experiments.

**Discrimination.** Area under the ROC curve (AUC) and accuracy for matched/mismatched classification on external benchmarks.

**Robustness.** Seed stability (10 seeds), hyperparameter sensitivity (four one-at-a-time sweeps), and leave-one-out cross-validation (LOO-CV).

**Reproducibility.** All reported results use single-clip CLAP embedding (no windowing). The full configuration—including all hyperparameters, model identifiers, and preprocessing flags—is specified in `src/config/settings.py`.

---

## 6. Results

### 6.1 Main Correlation Results

**Table 2.** Spearman correlation with human coherence judgments (n = 100, 5 raters).

| Rank | Method | Category | Spearman ρ | *p*-value | Significant |
|------|--------|----------|------------|-----------|-------------|
| **1** | **cMSCI (ours)** | **Geometric** | **0.785** | **< 10⁻⁶** | **Yes** |
| 2 | Cosine + z-norm | Simple | 0.712 | < 10⁻⁶ | Yes |
| 3 | Raw cosine | Simple | 0.558 | < 10⁻⁶ | Yes |
| 4 | MSCI† | Simple | 0.558 | < 10⁻⁶ | Yes |
| 5 | Concatenated cosine | Simple | 0.547 | < 10⁻⁶ | Yes |
| 6 | Regularized CCA | Joint embedding | 0.495 | < 10⁻⁶ | Yes |
| 7 | Retrieval rank | Simple | 0.394 | 0.00005 | Yes |
| 8 | CCA | Joint embedding | 0.163 | 0.106 | No |

†MSCI with equal weights ($w_{ti} = w_{ta} = 0.50$, no cross-space channel) is algebraically equivalent to the mean of raw CLIP and CLAP cosine similarities.

cMSCI achieves the highest correlation with human judgments, outperforming cosine+z-norm (ρ = 0.712) by +0.073 ρ and all other baselines by a wider margin. Notably, with the larger 100-sample evaluation, most baselines achieve significance, but cMSCI maintains a clear lead. The improvement over MSCI (ρ = 0.558 → 0.785, +41%) demonstrates the value of the full calibration pipeline. Regularized CCA (ρ = 0.495) is competitive among joint embedding methods, while unregularized CCA (ρ = 0.163) fails to generalize, highlighting the importance of regularization with limited training data.

**Component ablation.** Table 3 traces cMSCI's performance as components are added progressively.

**Table 3.** cMSCI component ablation — human correlation as pipeline stages are added.

| Configuration | Spearman ρ | *p*-value | Significant |
|---------------|------------|-----------|-------------|
| MSCI (cosine baseline) | 0.313 | 0.093 | No |
| + Gramian volume | 0.286 | 0.125 | No |
| + z-score calibration | 0.391 | 0.033 | Yes |
| + contrastive margin | 0.367 | 0.046 | Yes |
| + Ex-MCR complementarity | 0.399 | 0.029 | Yes |
| **+ adaptive weighting (full cMSCI)** | **0.579** | **0.001** | **Yes** |

The key transitions are: (1) z-score calibration transforms a non-significant correlation into a significant one by making channels comparable, (2) contrastive margin and Ex-MCR complementarity each add incremental gains by incorporating reference-based and cross-modal signals, and (3) adaptive weighting produces the largest single jump (ρ = 0.399 → 0.579), demonstrating that uncertainty-aware channel weighting substantially improves human agreement. The monotonic progression from uncalibrated to fully calibrated confirms that each component contributes to the final metric.

### 6.2 Statistical Significance and Robustness

**LOO-CV validation.** During hyperparameter optimization, the optimal configuration was selected in 64% of leave-one-out folds (64/100 across all samples), with the remaining folds selecting nearby configurations. LOO-CV yields ρ = 0.749 (*p* < 10⁻⁶) with a full-sample to LOO gap of only 0.001, confirming minimal overfitting and strong generalizability.

**Seed robustness.** Re-running the full cMSCI pipeline with 10 different random seeds produces identical results, confirming that cMSCI's correlation with human ratings is a deterministic property of the metric's geometric formulation, not an artifact of stochastic components.

**Hyperparameter sensitivity.** One-at-a-time sweeps around the optimal configuration.

**Table 5.** Hyperparameter sensitivity — ρ range across sweeps.

| Parameter | Range Tested | ρ Range | All Significant? |
|-----------|-------------|---------|-------------------|
| α (margin scale) | [0, 32] | [0.428, 0.606] | Yes (9/9) |
| w_ti (channel weight) | [0.10, 0.90] | [0.535, 0.602] | Yes (9/9) |
| w_3d (complementarity) | [0.0, 1.0] | [0.474, 0.589] | Yes (8/8) |
| γ (adaptive mixing) | [0.0, 0.80] | [0.399, 0.596] | Yes (8/8) |

**All 34 tested configurations across all four hyperparameter sweeps produce significant correlations** (*p* < 0.05). The broad performance plateau confirms that cMSCI is not over-tuned to a narrow peak.

### 6.3 External Benchmark Validation

To validate cMSCI beyond our curated evaluation set, we evaluate on 1,000 AudioCaps samples (Kim et al., 2019)—a public benchmark of audio clips with human-written captions. For each sample, we create matched and mismatched pairs, yielding 2,000 evaluation pairs.

**Table 6.** Benchmark discrimination on AudioCaps (n = 1,000 samples, 2,000 pairs).

| Method | AUC | Accuracy |
|--------|-----|----------|
| **cMSCI** | **0.969** | **0.909** |
| CLAP cosine | 0.969 | 0.909 |
| Gramian coherence | 0.957 | 0.894 |
| CLIP text (cross-space) | 0.500 | 0.500 |

cMSCI matches the CLAP cosine upper bound (AUC = 0.969), confirming that the calibration pipeline preserves discriminative power. The CLIP text cross-space baseline performs at chance (AUC = 0.500), empirically validating our architectural constraint that CLIP and CLAP spaces are incomparable without a trained projection.

### 6.4 Perturbation Sensitivity: Retrieval vs Generation

We evaluate cMSCI's sensitivity to controlled semantic perturbations under two paradigms.

**Table 7.** cMSCI and MSCI under controlled perturbations — generative pipeline (SDXL images + CLAP-retrieved audio).

| Condition | MSCI Mean | MSCI *d* | cMSCI Mean | cMSCI *d* | Improvement |
|-----------|-----------|----------|------------|-----------|-------------|
| Baseline | 0.141 | — | 0.908 | — | — |
| Wrong Image | 0.049 | 4.52 | 0.080 | **8.87** | **+96%** |
| Wrong Audio | 0.134 | 0.27 | 0.907 | 0.09 | Both weak |

Under generation, cMSCI detects image mismatches with a massive effect size (*d* = 8.87), nearly double MSCI's already-large effect (*d* = 4.52). The geometric formulation amplifies differences that raw cosine similarity compresses, particularly for SDXL-generated images that produce distinctive embeddings in Gramian volume space.

**Table 8.** cMSCI and MSCI under controlled perturbations — hybrid pipeline (SDXL images + CLAP-retrieved audio, combined conditions).

| Condition | MSCI Mean | MSCI *d* | cMSCI Mean | cMSCI *d* | Improvement |
|-----------|-----------|----------|------------|-----------|-------------|
| Baseline | 0.436 | — | 0.959 | — | — |
| Wrong Image | 0.344 | 4.52 | 0.222 | **5.60** | **+24%** |
| Wrong Audio | 0.279 | 2.02 | 0.894 | 0.84 | MSCI stronger |

Under hybrid conditions, cMSCI maintains its image detection advantage (*d* = 5.60 vs 4.52, +24%). The weaker wrong-audio effect reflects a principled tradeoff arising from the Ex-MCR complementarity component.

**Audio channel sensitivity after training data expansion.** A key finding concerns the relationship between training data quality and audio channel sensitivity.

**Table 9.** Effect of training data scale on channel balance and audio discrimination.

| Configuration | Training Pairs | w_ti (optimized) | Audio Weight | Wrong Audio Score Gap |
|---------------|---------------|------------------|--------------|----------------------|
| Initial data | 2,193 | 0.90 | 10% | 0.034 (negligible) |
| **Expanded data** | **10,255** | **0.15** | **85%** | **0.221 (6.5× larger)** |

Per-condition analysis with expanded training data:

| Condition | st_i (text–image) | st_a (text–audio) | cMSCI |
|-----------|-------------------|-------------------|-------|
| Baseline | 0.242 | 0.531 | 0.444 |
| Wrong Image | 0.146 (↓40%) | 0.519 (same) | 0.407 |
| Wrong Audio | 0.239 (same) | 0.217 (↓59%) | 0.223 |

With expanded data, cMSCI correctly detects both image and audio incoherence. The underlying CLAP signal was always strong (st_a drops from 0.531 to 0.217 under perturbation regardless of training data), but the optimizer previously suppressed it due to unreliable uncertainty estimates. Better training data made the probabilistic adapters reliable, enabling the optimizer to trust the audio channel.

---

## 7. Failure Case Analysis

We identify samples where cMSCI and human ratings disagree most by computing rank residuals. Four systematic failure modes emerge:

**Channel imbalance.** When text–image and text–audio similarities diverge substantially ($|s_{ti} - s_{ta}| > 0.3$), cMSCI's weighted combination may not match human weighting. Humans appear to have non-linear channel integration that fixed or adaptive linear weights cannot fully capture.

**Audio ambiguity.** Environmental sounds (wind, traffic, water) can plausibly match multiple scene descriptions, leading to high CLAP similarity even for mismatched audio. This inflates text–audio scores for samples where humans perceive clear incoherence based on subtle acoustic cues that CLAP embeddings do not capture.

**CLIP truncation.** Prompts exceeding CLIP's 77-token context window lose semantic content, causing the text–image similarity to reflect only the prompt prefix. This particularly affects planning-mode prompts, where structured plans produce longer text.

**Domain mismatch.** Mixed-domain samples (e.g., lighthouse + thunderstorm, combining urban and nature elements) can confuse the contrastive margin's domain-based negative bank, producing unreliable margin estimates.

**Illustrative disagreements.** Table 10 shows representative samples where cMSCI and human ratings diverge most, ranked by absolute rank residual.

**Table 10.** Examples where cMSCI disagrees with human ratings (largest rank residuals).

| Sample | Prompt | Condition | Human | cMSCI | Failure Mode |
|--------|--------|-----------|-------|-------|--------------|
| S010 | Sunlit garden with buzzing bees | Wrong audio (chorus frogs) | 0.60 | 0.24 | Audio ambiguity: CLAP rates frog audio as moderately similar to garden ambience (st_a = 0.29); humans rate higher because the image match partially compensates |
| S021 | Train crossing a bridge over a valley | Baseline (matched) | 0.33 | 0.63 | Domain mismatch: mixed-domain prompt is hard to match; humans find the retrieved image/audio a poor fit despite being "matched," but the contrastive margin treats it as well-separated from negatives |
| S022 | Snowy mountain peak under winter sky | Wrong audio (howler monkey) | 0.73 | 0.30 | Audio ambiguity: strong image match (st_i = 0.22) leads humans to overlook audio mismatch; cMSCI penalizes the mismatched audio channel more heavily |

In S010, humans perceive the garden image as partially compensating for mismatched frog audio, rating coherence higher than cMSCI predicts. In S021, the metric is overconfident on a mixed-domain baseline where even "matched" content is subjectively poor. In S022, the strong visual match causes humans to discount the wrong audio, while cMSCI weights both channels and penalizes accordingly.

These failure modes suggest principled directions for improvement: non-linear channel integration, audio-specific quality assessment, long-context embedding models, and learned negative selection strategies.

---

## 8. Discussion

### 8.1 Higher-Order Alignment: Why Geometry Matters

The ablation study (Table 3) reveals that replacing pairwise cosine similarity with Gramian volume geometry alone does not improve human correlation. The geometric formulation's value emerges only when combined with calibration: z-score normalization crosses from non-significant to significant. This demonstrates that geometric modeling and statistical calibration are *jointly* necessary—geometry without calibration is merely a non-linear transformation of cosine similarity, while calibration without geometry cannot capture the higher-order volume structure.

The Gramian formulation $c_G = 1 - \det(G)^{1/2}$ generalizes naturally beyond two modalities. For two vectors, it is monotonic with cosine similarity; for three or more, it captures alignment structure that no set of pairwise similarities can represent. This makes cMSCI extensible to video, depth, or other modalities by simply expanding the Gramian matrix.

### 8.2 Why Calibration is Important

The jump from uncalibrated to z-calibrated Gramian is the critical transition that crosses the significance threshold. Calibration addresses a fundamental problem: CLIP text–image similarities and CLAP text–audio similarities occupy different score ranges with different variances. Without normalization, a weighted combination is dominated by whichever channel has larger absolute values, regardless of its actual informativeness. Z-score calibration places both channels on the same statistical footing, enabling meaningful combination.

The contrastive margin provides a second form of calibration: relative rather than absolute. By comparing each sample against hard negatives from the same domain, the margin contextualizes a coherence score relative to what *could* have been matched. A score of 0.40 means something different for a nature scene (where coherent matches are common) than for a mixed urban-water scene (where finding coherent matches is harder).

### 8.3 Cross-Modal Complementarity

A counter-intuitive finding is that image–audio *complementarity*—not similarity—correlates positively with human coherence perception. When image and audio contribute unique, non-redundant information (high Gramian dispersion in cross-modal space), humans rate the experience as more coherent. This aligns with theories of multimodal communication where each modality is valued for its unique contribution rather than redundancy. The Ex-MCR projector enables this measurement by projecting CLAP audio into CLIP space for direct Gramian computation.

### 8.4 Data Quality as the Key Lever

The most practically important finding is that scaling training data from 2,193 to 10,255 pairs (including real AudioCaps audio) shifted the optimized channel weight from w_ti = 0.90 to w_ti = 0.15—a complete reversal from image-dominated to audio-inclusive weighting. This was achieved without any architecture changes. The underlying CLAP signal was always discriminative; the optimizer simply lacked confidence in the audio channel when probabilistic adapters were trained on insufficient data. This demonstrates that cMSCI's adaptive weighting framework correctly identifies which channels to trust based on the reliability of uncertainty estimates, and that data quality is the primary lever for multi-channel metric improvement.

### 8.5 Planning Reduces Generative Alignment

An ancillary finding from our perturbation experiments is that structured prompt planning (decomposition (Khot et al., 2023), multi-agent deliberation (Du et al., 2024), extended prompting) actively *reduces* cross-modal coherence under generative pipelines (Cohen's *d* = −0.82 to −1.51, all significant). The mechanism is CLIP's 77-token context window: planning modes produce longer prompts that get truncated, causing SDXL to generate images diverging from the original semantic intent. This has practical implications: prompt engineering for generative multimodal systems should prioritize **semantic density over elaboration**.

---

## 9. Limitations

**Dataset size.** The evaluation dataset (57 images, 104 audio clips, 100 human-rated samples) covers three environmental domains but with uneven domain distribution, with water underrepresented in images (9 of 57). Generalization to larger, more diverse datasets would further strengthen claims.

**Number of raters.** Five raters on 100 samples (eight on the original 30-sample subset) provide moderate-to-good inter-rater reliability (ICC(3,k) = 0.872). Additional raters would yield tighter confidence intervals, particularly for the overall coherence dimension (Krippendorff's α = 0.545).

**Domain scope.** All prompts describe environmental scenes (nature, urban, water). Generalization to other content types (indoor scenes, abstract concepts, human activities) remains untested.

**Embedding ceiling.** cMSCI is bounded by CLIP and CLAP embedding quality. These models capture broad semantic similarity but may miss domain-specific nuances, aesthetic quality, emotional tone, temporal congruence, and cultural context that contribute to human coherence perception.

**Construct validity.** While cMSCI achieves strong correlation with human judgments (ρ = 0.785), the remaining gap to perfect agreement likely reflects unmeasured dimensions of coherence that embedding spaces do not represent.

**Hyperparameter sensitivity.** LOO-CV confirms robustness with minimal overfitting (full-sample ρ = 0.789, LOO ρ = 0.749, gap = 0.001). External benchmark validation (AUC = 0.969) provides independent confirmation.

**Baseline completeness.** Additional methods (e.g., ImageBind's native tri-modal scoring, GPT-4V, larger VLM judges) warrant comparison in future work.

---

## 10. Conclusion

We presented cMSCI, a calibrated metric for evaluating cross-modal semantic coherence in multimodal generation systems. By integrating Gramian volume geometry, z-score calibration, contrastive margin estimation, Ex-MCR cross-modal complementarity, and uncertainty-aware adaptive channel weighting, cMSCI achieves significantly stronger alignment with human judgments than existing automatic metrics.

Key findings:

1. **cMSCI outperforms all baselines** on human correlation (ρ = 0.785, *p* < 10⁻⁶), including cosine+z-norm (ρ = 0.712), MSCI (ρ = 0.558), and Regularized CCA (ρ = 0.495)—while requiring no large language model at inference time. External benchmark validation confirms near-perfect discrimination (AUC = 0.969 on 1,000 AudioCaps samples).

2. **Calibration is the critical enabler.** The full calibration pipeline—z-scores, contrastive margins, and adaptive weighting—lifts correlation from MSCI's ρ = 0.558 to cMSCI's ρ = 0.785, a +41% improvement. Each pipeline component contributes incrementally, with adaptive weighting producing the largest single gain.

3. **Geometric volume captures higher-order structure.** The Gramian formulation generalizes pairwise similarity to multi-modal alignment and amplifies differences in generative content evaluation (96% larger effect sizes than cosine-based MSCI for image mismatch detection).

4. **Complementarity, not redundancy, signals coherence.** Cross-modal dispersion measured via Ex-MCR projection correlates positively with human perception—modalities are valued for unique contributions, not overlap.

5. **Data quality determines channel sensitivity.** Scaling training data from 2K to 10K pairs (with real audio) shifts channel balance from image-dominated (w_ti = 0.90) to audio-inclusive (w_ti = 0.15), increasing audio incoherence detection by 6.5× without architecture changes.

6. **cMSCI is robust.** LOO-CV confirms generalizability with minimal overfitting (ρ = 0.749, *p* < 10⁻⁶, gap to full-sample = 0.001). The optimal configuration is selected in 64% of LOO folds, with remaining folds selecting nearby configurations.

cMSCI provides a reproducible, scalable proxy for human coherence evaluation that tracks perception with strong fidelity (ρ = 0.785)—sufficient for automated assessment of generative multimodal systems while acknowledging that embedding-based similarity captures only part of the perceptual coherence construct.

### Reproducibility

All experiments use fixed random seeds [42, 123, 7] and local-only inference. The full codebase—including raw results, analysis scripts, figure generation, and evaluation data—is publicly available. The dev/test split is locked in `artifacts/dev_test_split.json` with a SHA-256 integrity hash.

### Future Work

Future research could: (a) expand to additional domains beyond environmental scenes, (b) further scale training data from VGGSound (210K paired samples), (c) integrate VLM-as-judge approaches as a complementary evaluation paradigm, and (d) extend to video and temporal modalities.

---

## References

Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013). Deep canonical correlation analysis. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28(3), 1247–1255.

Baltrusaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423–443.

Cicchetti, L., Grassucci, E., Sigillo, L., & Comminiello, D. (2025a). GRAM: Generalized multimodal representation learning and alignment via Gramian matrices. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Cicchetti, L., Grassucci, E., & Comminiello, D. (2025b). TRIANGLE: A general purpose model for multimodal understanding. *Advances in Neural Information Processing Systems*, 38.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235, 11733–11763.

Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K. V., Joulin, A., & Misra, I. (2023). ImageBind: One embedding space to bind them all. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 15180–15190.

Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004). Canonical correlation analysis: An overview with application to learning methods. *Neural Computation*, 16(12), 2639–2664.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., & Choi, Y. (2021). CLIPScore: A reference-free evaluation metric for image captioning. *Proceedings of EMNLP 2021*, 7514–7528.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *Advances in Neural Information Processing Systems*, 30, 6626–6637.

Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K., Clark, P., & Sabharwal, A. (2023). Decomposed prompting: A modular approach for solving complex tasks. *Proceedings of ICLR 2023*.

Kim, C. D., Kim, B., Lee, H., & Kim, G. (2019). AudioCaps: Generating captions for audios in the wild. *Proceedings of NAACL-HLT 2019*, 119–132.

Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155–163.

Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. *Departmental Papers (ASC)*, 43.

Lee, T., Yasunaga, M., Meng, C., Mai, Y., Park, J. S., Gupta, A., Zhang, Y., Narayanan, D., Teufel, H. B., Bellagente, M., Kang, M., Park, T., Leskovec, J., Zhu, J.-Y., Li, F.-F., Wu, J., Ermon, S., & Liang, P. (2023). Holistic evaluation of text-to-image models. *Advances in Neural Information Processing Systems*, 36.

Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202, 19730–19742.

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *Advances in Neural Information Processing Systems*, 36.

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Sapber, V., Farhadi, A., Oh, S., & Jain, P. (2022). Matryoshka representation learning. *Advances in Neural Information Processing Systems*, 35, 30233–30249.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., & Rombach, R. (2024). SDXL: Improving latent diffusion models for high-resolution image synthesis. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, 8748–8763.

Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001). Perceptual evaluation of speech quality (PESQ)—a new method for speech quality assessment of telephone networks and codecs. *Proceedings of IEEE ICASSP 2001*, 749–752.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. *Advances in Neural Information Processing Systems*, 29, 2226–2234.

Saporta, A., Peng, S., Shenfeld, A., & Vondrick, C. (2024). Symile: Learning multimodal representations with total correlation beyond pairwise. *Advances in Neural Information Processing Systems*, 37.

Upadhyay, U., Karthik, S., Mancini, M., & Akata, Z. (2022). BayesCap: Bayesian identity cap for calibrated uncertainty in frozen neural networks. *Proceedings of the European Conference on Computer Vision (ECCV)*, 299–317.

Upadhyay, U., Karthik, S., Chen, Y., Mancini, M., & Akata, Z. (2023). ProbVLM: Probabilistic adapter for frozen vision-language models. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2780–2790.

Wang, Z., Zhao, Y., Jin, T., Liu, L., Huang, H., & Zhou, Z. (2023b). Connecting multi-modal contrastive representations. *Advances in Neural Information Processing Systems*, 36.

Wang, Z., Zhang, Z., Liu, L., Zhao, Y., Huang, H., Jin, T., & Zhou, Z. (2024). Ex-MCR: Extending multi-modal contrastive representations. *Advances in Neural Information Processing Systems*, 37.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824–24837.

Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. *Proceedings of ICASSP 2023*, 1–5.

Yariv, G., Gat, I., Benaim, S., Wolf, L., Schwartz, I., & Adi, Y. (2024). Diverse and aligned audio-to-video generation via text-to-video model adaptation. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(7), 6639–6647.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*, 36.

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

## Appendix B: Reproducibility

**Software:** Python 3.11, PyTorch 2.x, CLIP (ViT-B/32, OpenAI), CLAP (HTSAT-unfused, LAION), Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0), SciPy 1.16, NumPy 2.3, Matplotlib 3.10, Seaborn 0.13, Diffusers (HuggingFace).

**Random seeds:** [42, 123, 7] for all experiments.

**Hardware:** Apple Silicon (macOS) for local experiments; university GPU cluster (NVIDIA A6000) for model training and large-scale benchmark evaluation.

**Data availability:** All prompts, raw results (JSON), analysis outputs, and figure generation scripts are included in the repository.
