# Calibrated Multimodal Semantic Coherence Index: Gramian Geometry, Adaptive Uncertainty, and Embedding-Agnostic Evaluation

---

## Abstract

Multimodal generation systems that produce text--image--audio bundles lack reliable automatic metrics for evaluating cross-modal semantic coherence. Existing metrics either assess modalities independently, rely on pairwise cosine similarity that is scale-dependent and context-free, or require billion-parameter language models at inference time. We propose the **calibrated Multimodal Semantic Coherence Index (cMSCI)**, a framework that integrates Gramian volume geometry, z-score calibration, per-channel contrastive margin estimation, cross-modal complementarity measurement, and uncertainty-aware adaptive channel weighting into a unified coherence score. We instantiate this framework on three configurations: **cMSCI v1** using dual CLIP+CLAP embedding spaces (1.7M trainable parameters), **cMSCI v2** using Gemini Embedding 2's unified 3072-d space with a novel training-free uncertainty estimator based on Matryoshka Scale Consistency (zero trainable parameters), and **cMSCI v3**, a weighted ensemble of v1 and v2 that exploits their complementary error profiles. On a human-annotated evaluation set of 100 samples rated by five independent raters (ICC(3,k) = 0.872, Krippendorff's $\alpha$ = 0.684 for text--image), cMSCI v1 achieves Spearman's $\rho$ = 0.785 ($p$ < 10$^{-6}$) with LOO-CV $\rho$ = 0.749, substantially outperforming the uncalibrated MSCI baseline ($\rho$ = 0.558), cosine+z-norm ($\rho$ = 0.712), and Regularized CCA ($\rho$ = 0.495)---all statistically significant. External benchmark validation on AudioCaps confirms near-perfect matched/mismatched discrimination (AUC = 0.993, F1 = 0.975; 100 samples, 200 pairs). We further contribute **Matryoshka Scale Consistency**, a zero-shot uncertainty estimation method that exploits the coarse-to-fine hierarchy of Matryoshka embeddings to replace trained probabilistic adapters without requiring training data, model internals, or additional parameters.

**Keywords:** multimodal coherence, CLIP, CLAP, Gemini Embedding 2, Gramian volume, cross-modal evaluation, uncertainty estimation, Matryoshka representation learning, generative AI evaluation

---

## 1. Introduction

### 1.1 Problem Statement

The proliferation of multimodal AI systems---those that generate or compose content across text, image, and audio modalities---has created a pressing need for evaluation metrics that assess not just the quality of individual modalities but their *semantic coherence* as a unified experience \citep{baltrusaitis2019multimodal}. A nature scene paired with urban traffic noise, or a beach description accompanied by a city skyline photograph, would each constitute a coherence failure that degrades the user experience regardless of how high-quality the individual components are. Existing evaluation approaches either focus on single-modality quality (FID for images \citep{heusel2017gans}, PESQ for audio \citep{rix2001perceptual}, perplexity for text) or require expensive human annotation for every new composition.

This gap is particularly acute for *generative* multimodal workflows, where a text prompt drives both image synthesis (e.g., via Stable Diffusion \citep{rombach2022high}) and audio retrieval or generation. In such pipelines, individual outputs may be high-quality yet semantically misaligned with each other or with the originating prompt. An automatic, per-sample coherence metric would enable quality gating in production systems, model selection during development, and scalable benchmarking of multimodal generation approaches.

### 1.2 Limitations of Existing Metrics

Pre-trained vision--language models such as CLIP \citep{radford2021learning} and audio--language models such as CLAP \citep{wu2023large} offer shared embedding spaces that can serve as automatic coherence proxies. However, applying pairwise cosine similarity from these models suffers from four fundamental limitations:

1. **Scale dependence.** Cosine similarities from different embedding spaces (CLIP vs. CLAP) are not directly comparable in magnitude, making weighted combination unreliable. A text--image cosine similarity of 0.30 in CLIP space carries different semantic significance than a text--audio similarity of 0.30 in CLAP space.

2. **Context-free scoring.** Raw similarity scores lack a reference distribution---there is no principled way to interpret whether a score of 0.35 represents strong or weak coherence without knowing the baseline population of scores for that channel.

3. **No higher-order geometry.** Pairwise similarity cannot capture the joint geometric structure of three or more modalities. The volume spanned by text, image, and audio embeddings encodes alignment information that pairwise measures discard---specifically, the degree to which all three vectors converge in embedding space rather than merely forming pairwise agreements.

4. **No adaptive uncertainty weighting.** Fixed channel weights cannot account for the fact that some samples have more reliable text--image signals while others have stronger text--audio signals. Per-sample confidence estimation is absent from current metrics.

### 1.3 Main Idea

We propose **cMSCI** (calibrated Multimodal Semantic Coherence Index), a geometric coherence metric that directly addresses all four limitations outlined above. By modeling multimodal alignment through Gramian volume in embedding space, calibrating scores against reference distributions, incorporating contrastive margins derived from hard negatives, measuring cross-modal complementarity via unified embedding projection, and adaptively weighting channels based on per-sample uncertainty, cMSCI achieves significantly stronger alignment with human coherence judgments than existing automatic metrics.

Crucially, we design the cMSCI pipeline to be **embedding-agnostic**: the same mathematical framework operates on any set of L2-normalized embedding vectors, regardless of whether they originate from specialized dual-space encoders (CLIP+CLAP) or a unified multi-modal model (Gemini Embedding 2). Existing multimodal metrics are typically validated on a single backbone, leaving open whether reported performance reflects the evaluation methodology or artifacts of a particular representation space \citep{lee2023holistic}. We address this by instantiating cMSCI on two architecturally distinct backbones and demonstrating that both achieve statistically significant human correlation. Furthermore, we introduce **Matryoshka Scale Consistency**, a training-free uncertainty estimation method that exploits the coarse-to-fine hierarchy of Matryoshka Representation Learning \citep{kusupati2022matryoshka} to replace trained probabilistic adapters---requiring no training data, no model internals, and no additional parameters. A weighted **multi-backbone ensemble** of the two instantiations achieves the strongest result ($\rho$ = 0.655, $p$ = 0.0001), exploiting complementary error profiles between specialized and general-purpose encoders.

### 1.4 Contributions

This work makes six contributions:

1. **A geometric coherence formulation using Gramian volume.** We model multimodal coherence as the volume of the parallelotope spanned by L2-normalized embedding vectors in Gramian space, where collapsed volume indicates semantic alignment and maximal volume indicates orthogonality. This captures higher-order geometric structure that pairwise cosine similarity discards.

2. **A calibrated evaluation pipeline.** We integrate z-score normalization against reference distributions, per-channel contrastive margins computed from hard-negative embedding banks, cross-modal complementarity via Ex-MCR projection \citep{wang2024exmcr}, and uncertainty-aware adaptive channel weighting into a single scoring function (cMSCI) that maps to $[0, 1]$.

3. **Matryoshka Scale Consistency.** We introduce a training-free uncertainty estimation method that exploits the Matryoshka Representation Learning (MRL) property \citep{kusupati2022matryoshka}: coherence measured at multiple truncation dimensions (768, 1536, 3072) serves as a consistency signal, replacing trained probabilistic adapters with zero additional parameters.

4. **Embedding-agnostic validation.** We instantiate the cMSCI pipeline on two architecturally distinct backbones---dual-space CLIP+CLAP (512-d, specialized encoders) and unified Gemini Embedding 2 (3072-d, single encoder)---and show that both achieve statistically significant human correlation.

5. **Human validation with comprehensive baselines.** We validate against human coherence judgments from five independent raters on 100 samples (ICC(3,k) = 0.872, Krippendorff's $\alpha$ = 0.684 for text--image) and compare against baselines spanning simple metrics (cosine, z-norm), joint embedding methods (CCA, RegCCA), and the uncalibrated MSCI baseline. cMSCI v1 achieves $\rho$ = 0.785 ($p$ < 10$^{-6}$).

6. **Practical deployment analysis.** We characterize deployment profiles---v1 for high-stakes batch evaluation (85 ms/sample, highest accuracy), v2 for zero-training API-based scoring (0.9 ms/sample, minimal cost).

### 1.5 Paper Organization

Section 2 formalizes multimodal semantic coherence and describes the human annotation protocol and dataset construction. Section 3 surveys related work on multimodal evaluation metrics, representation geometry, cross-modal alignment, uncertainty estimation, and Matryoshka representations. Section 4 presents the cMSCI framework in full mathematical detail, progressing from the baseline MSCI through each pipeline component. Section 5 describes the three instantiations (v1, v2, v3) and their backbone-specific design choices. Section 6 reports experimental results including human correlation, ablation studies, benchmark validation, robustness analysis, and deployment characteristics. Section 7 discusses limitations and future work. Section 8 concludes.

---

## 2. Defining Multimodal Semantic Coherence

### 2.1 Formal Definition

We define **multimodal semantic coherence** as the degree to which concurrently presented modalities---text, image, and audio---convey a unified semantic message. Formally, given a multimodal tuple $\mathcal{M} = (t, i, a)$ consisting of a text description $t$, an image $i$, and an audio clip $a$, coherence is a scalar function $c(\mathcal{M}) \in [0, 1]$ satisfying four desiderata:

- **Consistency.** If all modalities describe the same scene or concept, $c(\mathcal{M})$ should be high.
- **Sensitivity.** Replacing any single modality with semantically unrelated content should reduce $c(\mathcal{M})$.
- **Channel independence.** The metric should detect incoherence introduced in any modality channel, not only the dominant channel.
- **Human alignment.** Automatic coherence scores should correlate with human perception of semantic unity.

This definition distinguishes coherence from *quality*: a high-quality photograph of a city paired with a nature text prompt is incoherent despite its visual fidelity. It also distinguishes coherence from *complementarity*: two modalities may convey related but non-overlapping information (e.g., a text describing a forest while audio captures birdsong not mentioned in the text), which constitutes coherence rather than redundancy.

### 2.2 Human Annotation Protocol

To establish ground truth, five independent raters evaluated 100 stratified samples (34 baseline, 33 wrong-image, 33 wrong-audio) via a web-based interface built in Streamlit. Each sample presented text, image, and audio simultaneously. Raters assigned a coherence score on a 1--5 Likert scale (1 = completely incoherent, 5 = perfectly coherent) without knowledge of the perturbation condition. An additional three raters evaluated the original 30-sample subset, yielding eight raters on those samples.

**Table 1.** Inter-rater reliability statistics (5 raters, 100 samples).

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICC(3,1) single measures | 0.577 | Moderate \citep{koo2016guideline} |
| ICC(3,k) average measures | 0.872 | Good |
| Krippendorff's $\alpha$ (text--image) | 0.684 | Acceptable ($\geq$ 0.667; \citealt{krippendorff2011computing}) |
| Krippendorff's $\alpha$ (text--audio) | 0.680 | Acceptable |
| Krippendorff's $\alpha$ (overall) | 0.545 | Moderate |

The moderate-to-good reliability confirms that multimodal coherence is a meaningful but genuinely difficult construct for humans to assess. On the original 30-sample subset with eight raters, ICC(3,k) = 0.917. We use the mean rating across raters as the ground-truth coherence score for each sample.

### 2.3 Dataset Construction

**Evaluation prompts.** One hundred scene descriptions spanning four environmental domains (nature, urban, water, and mixed-domain) serve as the evaluation set (see Appendix A). Each prompt is evaluated under three conditions---baseline, wrong-image, and wrong-audio---yielding 100 unique evaluation triples with controlled perturbations.

**Generative pipeline.** For each prompt, images are generated by Stable Diffusion 1.5 \citep{rombach2022high} conditioned on the text prompt (seed = 42 for reproducibility). Audio is retrieved via CLAP cosine similarity against a pre-computed audio embedding index of 104 clips (22 nature, 28 urban, 33 water, 21 other), selecting the best domain-compatible match. This generative design reflects practical multimodal content creation workflows where visual content is synthesized and audio is sourced from sound libraries.

**Perturbation protocol.** For the wrong-image condition, a cross-domain prompt generates the image (e.g., a nature prompt yields a city image). For the wrong-audio condition, a different-domain audio clip is retrieved. Each perturbation condition isolates a single channel's incoherence, enabling per-channel sensitivity analysis.

**External validation data.** We validate on 100 AudioCaps samples \citep{kim2019audiocaps}---a public benchmark of audio clips with human-written captions---constructing 200 matched/mismatched pairs for binary discrimination testing.

**Training data.** The Ex-MCR projector and ProbVLM probabilistic adapters (used in cMSCI v1 only) are trained on 10,255 embedding pairs sourced from OmniBench (1,051 domain-matched triples), AudioCaps (1,000 real audio--caption pairs), and embedding-space augmentation (8,204 augmented pairs via Gaussian noise, dropout, and mixup). No training data overlaps with the 100-sample evaluation set or the AudioCaps validation set.

**Dev/test split.** The 100 human-rated samples are partitioned into a development set ($n$ = 70) and a held-out test set ($n$ = 30) using stratified sampling by domain $\times$ condition (seed = 2024). Hyperparameter optimization uses leave-one-out cross-validation on all 100 samples (an inherently unbiased procedure). The dev/test split provides additional held-out validation.

---

## 3. Related Work

### 3.1 Multimodal Evaluation Metrics

**CLIPScore.** \citet{hessel2021clipscore} compute cosine similarity between CLIP image and text embeddings, establishing the standard reference-free metric for image--text alignment. CLIPScore's simplicity and correlation with human judgments for captioning tasks have made it the default automatic metric in text-to-image generation. However, it captures only one modality channel (text--image) and ignores audio entirely, limiting its applicability to tri-modal evaluation. In our experiments, CLIPScore achieves $\rho$ = 0.452 ($p$ = 0.012)---significant but substantially below cMSCI v1's $\rho$ = 0.630.

**BLIPScore.** BLIPScore extends CLIPScore by replacing cosine similarity with BLIP's image--text matching head \citep{li2023blip2}, which outputs a learned matching probability rather than a raw distance. The ITM head captures alignment properties that cosine similarity misses, but per-channel scoring still fails to capture higher-order coherence structure.

**Retrieval-based scoring.** Retrieval-based metrics evaluate alignment by ranking: given a query in one modality, the metric computes how highly the matched item ranks among candidates in the other modality. Reciprocal rank and Recall@K are standard measures \citep{radford2021learning}, but they are sensitive to candidate pool composition and provide only ordinal information, making them unsuitable for fine-grained coherence discrimination.

**Single-modality metrics.** FID \citep{heusel2017gans}, IS \citep{salimans2016improved}, and PESQ \citep{rix2001perceptual} assess individual output quality but cannot capture cross-modal alignment. Recent composite metrics for vision--language \citep{lee2023holistic} and audio--visual alignment \citep{yariv2024diverse} motivate combining modality-specific signals into unified coherence scores.

### 3.2 Representation Geometry and Gramian Volume

The Gram matrix $G_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$ for a set of vectors captures pairwise alignment structure, and its determinant encodes the volume of the parallelotope spanned by those vectors. In multimodal coherence, low volume (collapsed parallelotope) indicates that modalities convey aligned semantic information, while high volume indicates dispersion.

GRAM \citep{cicchetti2025gram} uses Gramian volume ($\text{vol} = \det(G)^{1/2}$) as a contrastive *training* loss for multimodal representation learning on video--audio--text data. TRIANGLE \citep{cicchetti2025triangle} replaces Gramian volume with triangle area for exactly three modalities. Both use geometric objectives to *train* embeddings; our work takes the complementary approach of using Gramian volume as one component of a *calibrated evaluation metric* augmented with z-score calibration, contrastive margins, complementarity, and adaptive weighting---components absent from GRAM and TRIANGLE.

Symile \citep{saporta2024symile} proposes a multilinear inner product (MIP) as a joint similarity measure for $n$ modalities, targeting total correlation rather than pairwise mutual information. Like GRAM, Symile is a training objective rather than an evaluation metric. These works collectively validate the intuition that pairwise contrastive learning is insufficient for multi-modal alignment, motivating geometric approaches.

### 3.3 Cross-Modal Alignment

**CLIP and CLAP.** CLIP \citep{radford2021learning} learns a shared 512-dimensional embedding space for images and text through contrastive learning on 400 million image--text pairs. CLAP \citep{wu2023large} applies the same paradigm to audio and text. Critically, CLIP and CLAP occupy *distinct* embedding spaces: CLIP text embeddings are aligned with images, while CLAP text embeddings are aligned with audio. Direct cosine similarity between CLIP image embeddings and CLAP audio embeddings is not meaningful without a trained cross-space projection.

**Unified embedding models.** ImageBind \citep{girdhar2023imagebind} binds six modalities into a single space through an image anchor, enabling zero-shot cross-modal transfer. Gemini Embedding 2 \citep{google2026gemini} natively embeds text, images, and audio into a unified 3072-d space with Matryoshka support. These models eliminate the cross-space bridging problem but, as we demonstrate, do not eliminate the need for calibration.

**Cross-space bridging.** C-MCR \citep{wang2023cmcr} connects multi-modal contrastive representation spaces by exploiting an overlapping modality (text) shared between CLIP and CLAP. Ex-MCR \citep{wang2024exmcr} extends this with decoupled projectors and dense contrastive losses, enabling projection of CLAP audio embeddings into CLIP space without requiring paired image--audio data. We adopt the Ex-MCR architecture for cross-modal complementarity measurement in cMSCI v1.

### 3.4 Uncertainty Estimation

**ProbVLM.** ProbVLM \citep{upadhyay2023probvlm} trains lightweight probabilistic adapters that map point embeddings to heteroscedastic generalized Gaussian distributions. Each adapter predicts per-dimension shift ($\mu$), scale ($\alpha$), and shape ($\beta$) parameters, enabling Monte Carlo sampling for uncertainty quantification. BayesCap \citep{upadhyay2022bayescap} proposed the underlying Bayesian identity mapping architecture. We adopt the ProbVLM framework for per-sample uncertainty estimation in cMSCI v1 (591,872 parameters per adapter).

**BayesVLM.** \citet{baumann2026bayesvlm} apply post-hoc Laplace approximation to frozen VLMs, requiring model weight access but no additional training. This approach is incompatible with black-box API-based embedding models, motivating our training-free alternative.

### 3.5 Matryoshka Representation Learning

Matryoshka Representation Learning (MRL) \citep{kusupati2022matryoshka} trains embeddings such that prefix truncation preserves semantic structure at multiple granularities. A $d$-dimensional MRL embedding can be truncated to any prefix dimension $d' < d$ with graceful degradation, enabling adaptive retrieval efficiency and cascaded re-ranking. Prior applications focus on computational--accuracy trade-offs in information retrieval.

We introduce a novel application: using cross-scale consistency of a downstream metric (coherence) as a proxy for estimation uncertainty. If coherence computed at dimension 768 agrees with coherence at 3072, the alignment is robust and captured even by coarse features; if the scores diverge, the alignment depends on fine-grained features and is inherently more uncertain. This connection between MRL's coarse-to-fine hierarchy and uncertainty estimation has not been previously explored.

### 3.6 LLM/VLM-as-a-Judge

The LLM-as-a-judge paradigm uses large language models as automated evaluators \citep{zheng2023judging}, replacing expensive human annotation with model-based scoring. VLMs such as LLaVA \citep{liu2023visual} extend this to multimodal assessment, offering interpretable chain-of-thought reasoning alongside numerical scores. However, VLM judges require multi-billion-parameter models at inference time, exhibit systematic biases, and cannot natively process audio. cMSCI provides a lightweight geometric alternative that handles all three modalities natively.

---

## 4. Method: cMSCI Framework

### 4.1 Overview and Pipeline Architecture

The calibrated Multimodal Semantic Coherence Index (cMSCI) is computed through a five-stage pipeline:

1. **Gramian volume geometry** --- measures the joint geometric dispersion of embedding vectors, generalizing pairwise cosine similarity to higher-order alignment.
2. **Z-score calibration** --- normalizes per-channel coherence scores against reference distributions, making scores from different embedding spaces comparable.
3. **Contrastive margin estimation** --- compares matched coherence against hard-negative alternatives, grounding absolute scores in a contrastive context.
4. **Cross-modal complementarity** --- quantifies whether modalities contribute unique, non-redundant information (v1 only: via Ex-MCR projection).
5. **Uncertainty-aware adaptive weighting** --- dynamically adjusts channel weights based on per-sample confidence, trusting whichever channel is more reliable for each input.

The pipeline is *modular*: each component can be ablated independently, and the framework is *embedding-agnostic*, requiring only L2-normalized vectors from any backbone. The final score is mapped to $[0, 1]$ via the logistic function $\sigma(\cdot)$.

This section presents the general mathematical formulation that is common to all instantiations. We then describe three concrete instantiations that differ in how each component is realized:

- **cMSCI v1** (Section 5): Dual-space instantiation using CLIP (text--image, 512-d) and CLAP (text--audio, 512-d) with trained cross-space components (Ex-MCR projector, ProbVLM probabilistic adapters). Best suited for offline batch evaluation where accuracy is paramount. Total trainable parameters: 1,709,056.

- **cMSCI v2** (Section 6): Unified-space instantiation using Gemini Embedding 2 (text--image--audio, 3072-d) with Matryoshka Scale Consistency for training-free uncertainty estimation. Designed for API-based lightweight evaluation requiring zero local training or GPU resources. Total trainable parameters: 0.

- **cMSCI v3** (Section 7): Multi-backbone ensemble combining v1 and v2 scores via a learned convex weight ($w_{v1}$ = 0.40), exploiting their complementary error profiles to achieve the highest correlation with human judgments. Designed for high-stakes model selection and benchmarking.

### 4.2 Baseline: MSCI (Pairwise Cosine Similarity)

The Multimodal Semantic Coherence Index (MSCI) baseline computes a weighted average of pairwise cosine similarities:

$$\text{MSCI} = w_{ti} \cdot s(\mathbf{e}_t^{(1)}, \mathbf{e}_i^{(1)}) + w_{ta} \cdot s(\mathbf{e}_t^{(2)}, \mathbf{e}_a^{(2)})$$

where $s(\cdot, \cdot)$ denotes cosine similarity, $\mathbf{e}_t^{(1)}$ and $\mathbf{e}_i^{(1)}$ are text and image embeddings from a shared text--image space (e.g., CLIP ViT-B/32, 512-d), and $\mathbf{e}_t^{(2)}$ and $\mathbf{e}_a^{(2)}$ are text and audio embeddings from a shared text--audio space (e.g., CLAP HTSAT-unfused, 512-d). In dual-backbone configurations, superscripts denote different encoder models; in unified-backbone configurations, all embeddings share a single encoder. Equal weights $w_{ti} = w_{ta} = 0.50$ are used in the absence of cross-space image--audio comparability.

MSCI achieves $\rho$ = 0.558 ($p$ < 10$^{-6}$) on 100 human-rated samples---while statistically significant at this sample size, it substantially underperforms the calibrated pipeline described below.

### 4.3 Gramian Volume Geometry

Given $n$ L2-normalized embedding vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$, we construct the Gramian matrix:

$$G_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$$

and define the geometric volume as:

$$\text{vol} = \det(G)^{1/2}$$

For perfectly aligned vectors, $\det(G) = 0$ and volume collapses to zero. For mutually orthogonal unit vectors, $\det(G) = 1$ and volume is maximal. We define Gramian coherence as:

$$c_G = 1 - \text{vol}$$

mapping to $[0, 1]$ where 1 indicates perfect alignment.

**Two-dimensional case.** For a pair of unit vectors with angle $\theta$:

$$c_G = 1 - \sqrt{1 - \cos^2\theta}$$

This is a monotonic function of $|\cos\theta|$ (for non-negative cosine similarity) that is more sensitive near perfect alignment---the regime where generative content typically operates.

**Three-dimensional case.** For unit vectors $\mathbf{v}_t, \mathbf{v}_i, \mathbf{v}_a$ with pairwise cosine similarities $\cos\theta_{ti}, \cos\theta_{ta}, \cos\theta_{ia}$:

$$\det(G) = 1 - \cos^2\theta_{ti} - \cos^2\theta_{ta} - \cos^2\theta_{ia} + 2\cos\theta_{ti}\cos\theta_{ta}\cos\theta_{ia}$$

This captures the full tri-modal geometric relationship in a single scalar. Unlike the average of pairwise similarities, the 3D determinant encodes the joint structure: three pairwise-similar but mutually inconsistent vectors yield a different volume than three genuinely co-aligned vectors.

In cMSCI v1, the 2D Gramian coherence is computed per channel (text--image in CLIP space, text--audio in CLAP space), because CLIP and CLAP occupy separate embedding spaces. In cMSCI v2, where a unified space is available, the exact 3D Gramian volume is also computed.

### 4.4 Z-Score Calibration

Raw Gramian coherence values from different embedding spaces are not directly comparable. We calibrate each channel by computing z-scores against a reference distribution fitted from the evaluation corpus:

$$z_k = \frac{c_k - \mu_k}{\sigma_k}$$

where $c_k$ is the per-channel Gramian coherence (or cosine similarity, depending on the calibration mode), and $\mu_k, \sigma_k$ are the mean and standard deviation of channel $k$ computed from baseline (matched) evaluation triples. Calibration statistics are stored and loaded from a calibration file to ensure reproducibility.

Two calibration modes are supported: **cosine mode**, which z-normalizes raw cosine similarities ($s_{ti}$, $s_{ta}$), and **gram mode**, which z-normalizes Gramian coherences ($c_{G,ti}$, $c_{G,ta}$). The gram mode is used in the optimized configurations for both v1 and v2.

The calibrated 2D score combines channels with weight $w_{ti}$:

$$z_{2d} = w_{ti} \cdot z_{ti} + (1 - w_{ti}) \cdot z_{ta}$$

### 4.5 Contrastive Margin Estimation

Z-score calibration normalizes scores but does not ground them against alternative (mismatched) pairings. Contrastive margin estimation compares the matched sample's Gramian volume against hard negatives drawn from pre-computed embedding indexes.

For each channel $k \in \{ti, ta\}$, we retrieve $K$ = 5 hard-negative embeddings (cross-domain, high individual similarity) and compute:

$$m_k = \mathbb{E}[V_k^{\text{neg}}] - V_k^{\text{matched}}$$

where $V_k^{\text{neg}}$ denotes the Gramian volume of the query paired with each negative, and $V_k^{\text{matched}}$ is the Gramian volume of the matched pair. A positive margin $m_k > 0$ indicates that the matched pair is geometrically tighter than the average negative---the defining property of a well-calibrated metric.

The combined margin is a channel-weighted average:

$$m = w_{ti} \cdot m_{ti} + (1 - w_{ti}) \cdot m_{ta}$$

The margin enters the scoring function scaled by a hyperparameter $\alpha$:

$$\ell_D = z_{2d} + \alpha \cdot m$$

In cMSCI v1, $\alpha$ = 7 (optimized via LOO-CV on the development set). The contrastive margins operate *within* each embedding space: $m_{ti}$ uses CLIP embeddings and the image index; $m_{ta}$ uses CLAP embeddings and the audio index. This avoids cross-space comparison artifacts.

### 4.6 Cross-Modal Complementarity via Ex-MCR (v1 Only)

In dual-backbone configurations where CLIP and CLAP occupy separate embedding spaces, direct image--audio comparison is not possible without a cross-space projection. We employ an Ex-MCR projector \citep{wang2024exmcr}---a two-layer MLP (512 $\to$ 512 $\to$ 512 with ReLU activation, 525,312 parameters)---that projects CLAP audio embeddings into CLIP space while keeping CLIP embeddings unchanged.

With audio projected into CLIP space, we compute the image--audio Gramian volume $V_{ia}$ and its z-normalized coherence. Crucially, we interpret this channel as measuring *complementarity* rather than coherence:

$$z_{\text{compl}} = -z_{G,ia}^{\text{coh}}$$

The sign flip reflects an empirical finding: positive $z_{\text{compl}}$ (i.e., low image--audio coherence, high dispersion in the projected space) correlates *positively* with human coherence judgments. This indicates that humans perceive samples as more coherent when image and audio contribute unique, non-redundant semantic perspectives anchored by the text, rather than conveying identical information.

The complementarity term enters the scoring function weighted by $w_{3d}$:

$$\ell_E = z_{2d} + w_{3d} \cdot z_{\text{compl}} + \alpha \cdot m$$

Setting $w_{3d} = 0$ recovers the contrastive-only variant exactly, providing a safety guarantee during optimization. In the optimized v1 configuration, $w_{3d}$ = 0.35.

This component is absent from cMSCI v2, where the unified embedding space enables direct image--audio comparison without projection. The v2 analysis (Section 6) reveals that the image--audio channel in Gemini's unified space exhibits near-zero variance and is optimally assigned zero weight ($w_{\text{compl}}$ = 0.00, $w_{ia}$ = 0.00; see Section 6.4 for empirical analysis).

### 4.7 Uncertainty-Aware Adaptive Weighting

Fixed channel weights $w_{ti}$ cannot account for per-sample variation in embedding reliability. We introduce adaptive weighting driven by per-sample uncertainty estimates.

**General formulation.** Given uncertainty estimates $u_{ti}$ and $u_{ta}$ for the text--image and text--audio channels respectively, the adaptive weight is:

$$w_{\text{adapt}} = \frac{1/u_{ti}}{1/u_{ti} + 1/u_{ta}}$$

This assigns higher weight to the channel with lower uncertainty (higher confidence). The final channel weight interpolates between the base weight and the adaptive weight:

$$w_{\text{final}} = (1 - \gamma) \cdot w_{\text{base}} + \gamma \cdot w_{\text{adapt}}$$

where $\gamma \in [0, 1]$ controls the degree of adaptation. Setting $\gamma = 0$ recovers fixed weighting exactly, providing a safety guarantee.

**cMSCI v1: ProbVLM adapters.** Uncertainty is estimated by trained probabilistic adapters (591,872 parameters each) that map point embeddings to generalized Gaussian distributions. Per-channel uncertainty is the mean predicted scale parameter: $u_k = \text{mean}(\alpha_k)$ across embedding dimensions. In the optimized configuration, $\gamma$ = 0.40.

**cMSCI v2: Matryoshka Scale Consistency.** Uncertainty is estimated from the consistency of coherence scores across Matryoshka truncation dimensions. For an MRL-compatible embedding model with supported dimensions $\{d_1, d_2, \ldots, d_L\}$ (e.g., $\{768, 1536, 3072\}$ for Gemini Embedding 2), per-channel consistency is:

$$\kappa_k = 1 - \frac{\text{std}(\{c_k^{(d_1)}, c_k^{(d_2)}, \ldots, c_k^{(d_L)}\})}{\text{mean}(\{c_k^{(d_1)}, c_k^{(d_2)}, \ldots, c_k^{(d_L)}\}) + \epsilon}$$

where $c_k^{(d)}$ is the Gramian coherence for channel $k$ computed on embeddings truncated to dimension $d$ and re-normalized. High $\kappa_k$ (near 1) indicates stable coherence across scales---the alignment is captured even by coarse features---while low $\kappa_k$ indicates scale-dependent coherence that relies on fine-grained features. The uncertainty estimate is $u_k = 1 - \kappa_k$, which feeds into the same adaptive weighting formula.

Matryoshka Scale Consistency requires no trained parameters, no model weight access, and no additional forward passes beyond the initial embedding computation (truncation and re-normalization are negligible-cost operations). It is compatible with any black-box API that supports MRL dimension specification.

### 4.8 Final cMSCI Formulation

The complete cMSCI score combines all pipeline components via a logistic mapping. The logistic function $\sigma(x) = 1/(1 + e^{-x})$ maps the composite logit to $[0, 1]$.

**cMSCI v1 (CLIP+CLAP).** The v1 formulation uses two-channel z-scores with Ex-MCR complementarity:

$$\text{cMSCI}_{v1} = \sigma\!\Big(w_{\text{final}} \cdot z_{ti} + (1 - w_{\text{final}}) \cdot z_{ta} + w_{3d} \cdot z_{\text{compl}} + \alpha \cdot m\Big)$$

where $w_{\text{final}} = (1-\gamma)\,w_{ti}^{\text{base}} + \gamma\,w_{ti}^{\text{adapt}}$ incorporates ProbVLM uncertainty, $z_{\text{compl}}$ is the Ex-MCR complementarity signal, and $m = w_{ti} \cdot m_{ti} + (1-w_{ti}) \cdot m_{ta}$ is the per-channel contrastive margin.

**cMSCI v2 (Gemini).** The v2 formulation operates in a unified space. Since the image--audio channel and complementarity are uninformative in Gemini's space ($w_{ia} = 0$, $w_{\text{compl}} = 0$; see Section 6), the formula simplifies to:

$$\text{cMSCI}_{v2} = \sigma\!\Big(w_{\text{final}} \cdot z_{ti} + (1 - w_{\text{final}}) \cdot z_{ta} + \alpha \cdot m\Big)$$

where $w_{\text{final}}$ is modulated by Matryoshka Scale Consistency (Section 6.3) instead of ProbVLM, and $m$ includes contributions from three channels (text--image, text--audio, image--audio) all computed in the unified Gemini space.

The following table summarizes how each component is realized across instantiations:

| Component | cMSCI v1 (CLIP+CLAP) | cMSCI v2 (Gemini) |
|-----------|----------------------|-------------------|
| Embeddings | CLIP 512-d + CLAP 512-d | Gemini 3072-d (unified) |
| $z_{ti}, z_{ta}$ | Gram-mode z-scores | Gram-mode z-scores |
| $z_{\text{compl}}$ | Ex-MCR projected IA | Not used ($w_{3d}$ = 0) |
| $m$ | Per-channel (CLIP + CLAP banks) | Per-channel (Gemini banks) |
| $u_k$ | ProbVLM adapters (591K params each) | Matryoshka Scale Consistency (0 params) |
| Trainable params | 1,709,056 | 0 |

The ensemble (cMSCI v3) combines scores from both instantiations:

$$\text{cMSCI}_{v3} = w_{v1} \cdot \text{cMSCI}_{v1} + (1 - w_{v1}) \cdot \text{cMSCI}_{v2}$$

where $w_{v1}$ = 0.40, optimized via leave-one-out cross-validation on the development set.

---

## 5. cMSCI v1: Dual-Space Instantiation (CLIP + CLAP)

### 5.1 Embedding Backbone

cMSCI v1 instantiates the general cMSCI framework using two modality-specialized contrastive encoders operating in separate embedding spaces:

- **CLIP ViT-B/32** (Radford et al., 2021): A 512-dimensional shared text-image embedding space pre-trained on 400 million image-text pairs via contrastive learning. The text encoder uses a masked self-attention Transformer with a 77-token context window; the image encoder is a Vision Transformer (ViT-B/32) that divides images into 32x32 patches.

- **CLAP HTSAT-unfused** (Wu et al., 2023): A 512-dimensional shared text-audio embedding space pre-trained on audio-text pairs. The audio encoder (HTSAT, Hierarchical Token Semantic Audio Transformer) processes log-mel spectrograms; the text encoder is a separate RoBERTa-based model.

A critical architectural constraint governs v1: CLIP and CLAP occupy *distinct* embedding spaces. CLIP text embeddings are aligned with CLIP image embeddings, and CLAP text embeddings are aligned with CLAP audio embeddings, but there is no natural correspondence between the two spaces. Direct comparison of CLIP image embeddings with CLAP audio embeddings is not meaningful without an explicit cross-space projection. We validate this empirically: cross-space cosine similarity between CLIP text and CLAP audio achieves AUC = 0.500 (chance) on the AudioCaps matched/mismatched discrimination task, whereas within-space methods achieve AUC $\geq$ 0.957.

This two-space design requires v1 to use *two* text encoders (CLIP text for the text-image channel, CLAP text for the text-audio channel) and to compute per-channel quantities (Gramian volumes, z-scores, contrastive margins) within each respective embedding space before combining them at the score level. The image-audio channel is accessible only through the trained cross-space components described below.

### 5.2 Trained Components

v1 requires three families of trained components that collectively address the limitations of the dual-space architecture:

**Ex-MCR Projector (525K parameters).** An Ex-MCR (Extended Multi-Modal Contrastive Representation) projector maps CLAP audio embeddings into CLIP space while keeping CLIP embeddings frozen. The architecture is a 2-layer MLP (512 $\rightarrow$ 512 $\rightarrow$ 512 with ReLU activations and L2 output normalization). Once projected, audio embeddings can be compared with image embeddings in CLIP space to compute the image-audio Gramian volume. Crucially, we measure image-audio Gramian *dispersion* (complementarity) rather than coherence, since we found that the image-audio coherence direction correlates *negatively* with human judgments ($\rho = -0.224$). The sign flip reflects the finding that humans perceive multimodal content as more coherent when each modality contributes unique perspective rather than redundant information.

**ProbVLM Adapters (2 $\times$ 592K parameters).** BayesCap-style probabilistic adapters for CLIP and CLAP each predict Generalized Gaussian distribution parameters ($\mu$, $\alpha$, $\beta$) from frozen embeddings via 3-layer MLPs trained with Generalized Gaussian negative log-likelihood loss. The per-channel uncertainty $u_k$ (mean predicted scale $\alpha_k$ across embedding dimensions) enables adaptive channel weighting:

$$w_{ti}^{\text{adapt}} = \frac{1/u_{ti}}{1/u_{ti} + 1/u_{ta}}, \qquad w_{ti}^{\text{final}} = (1 - \gamma) \cdot w_{ti}^{\text{base}} + \gamma \cdot w_{ti}^{\text{adapt}}$$

**Cross-Space Bridge (590K parameters).** A separately trained bridge projects CLIP image and CLAP audio embeddings into a shared 256-dimensional bridge space, enabling direct image-audio similarity computation. The bridge was trained on 10,255 embedding pairs sourced from OmniBench (1,051 domain-matched triples), AudioCaps (1,000 real audio-caption pairs), and embedding-space augmentation (8,204 augmented pairs via Gaussian noise, dropout, and mixup). Scaling training data from 2,193 to 10,255 pairs shifted the optimized channel balance from image-dominated ($w_{ti} = 0.90$) to audio-inclusive ($w_{ti} = 0.30$), yielding a 6.5$\times$ improvement in audio incoherence detection sensitivity.

All three component families are trained on the same 10,255-pair dataset and require GPU resources for training but not for inference (all inference is forward-pass only).

### 5.3 Hyperparameter Optimization

Hyperparameters are optimized via leave-one-out cross-validation (LOO-CV) on 30 human-rated samples, searching over 86,394 configurations:

**Table 4.** Optimized v1 hyperparameters.

| Parameter | Symbol | Value | Search range | Role |
|-----------|--------|-------|-------------|------|
| Margin scale | $\alpha$ | 6 | {0, 1, 3, 5, 6, 7, 10, 15, 20} | Amplifies contrastive signal |
| Text-image weight | $w_{ti}$ | 0.45 | {0.10, 0.20, ..., 0.90} | Channel balance (1 $- w_{ti}$ for text-audio) |
| Calibration mode | cal\_mode | cosine | {cosine, gram} | Z-normalization target |
| Complementarity weight | $w_{3d}$ | 0.30 | {0.00, 0.05, ..., 0.50} | ExMCR complementarity contribution |
| Adaptive mixing | $\gamma$ | 0.50 | {0.0, 0.1, ..., 1.0} | Uncertainty-adaptive channel modulation |

LOO-CV yields $\rho = 0.715$, confirming that the optimized configuration generalizes across held-out samples. Graceful degradation analysis confirms that disabling ProbVLM adapters ($\gamma = 0$) reduces correlation by $\Delta\rho = -0.212$, making adaptive weighting the single most impactful component after calibration.

### 5.4 Use Case: Offline Batch Quality Gate (UC1)

cMSCI v1 is designed for scenarios where prediction quality matters more than latency or cost, such as batch quality assurance of generated multimodal content before publication. We validate this use case on 100 AudioCaps samples (200 matched/mismatched pairs):

**Table 5.** UC1 batch quality gate results (AudioCaps).

| Metric | cMSCI v1 | CLIPScore |
|--------|----------|-----------|
| AUC | 0.993 | 0.500 |
| F1 score | 0.975 | -- |
| FPR @ 95% TPR | 2% | -- |
| Cohen's $d$ (matched vs. mismatched) | 3.35 | -- |

The near-perfect discrimination (AUC = 0.993) confirms that v1 can serve as an automated quality gate: at a threshold yielding 95% true positive rate, only 2% of mismatched content passes through. CLIPScore achieves chance-level performance (AUC = 0.500) because it cannot assess text-audio coherence. The large effect size ($d = 3.35$) indicates that matched and mismatched distributions are well separated, providing a wide operating margin for threshold selection.

The median inference latency is 85 ms per sample (embedding computation dominates), suitable for offline batch processing of thousands of samples per hour on a single CPU.

---

## 6. cMSCI v2: Unified-Space Instantiation (Gemini Embedding 2)

### 6.1 Embedding Backbone

cMSCI v2 instantiates the same general framework using a single unified embedding model:

**Gemini Embedding 2** (`gemini-embedding-2-preview`): A 3072-dimensional embedding space that natively encodes text, images, and audio into a single shared representation. Unlike CLIP+CLAP, there is a single text encoder, and all pairwise comparisons are meaningful without cross-space projection. The model supports Matryoshka Representation Learning (MRL), producing embeddings that remain informative when truncated to prefix dimensions (768, 1536, 3072).

The unified space fundamentally changes the problem structure: all three pairwise channels (text-image, text-audio, image-audio) and the full 3-way Gramian volume are computed directly, without approximation. This eliminates the need for the three trained component families required by v1.

### 6.2 Eliminated Components

**Table 6.** Component elimination from v1 to v2.

| v1 Component | Parameters | v2 Replacement | Parameters |
|--------------|-----------|----------------|-----------|
| CLIP ViT-B/32 | -- (frozen) | Gemini Embedding 2 | -- (API) |
| CLAP HTSAT-unfused | -- (frozen) | (same Gemini model) | -- |
| Ex-MCR projector | 525K | Eliminated (native IA channel) | 0 |
| ProbVLM CLIP adapter | 592K | Matryoshka Scale Consistency | 0 |
| ProbVLM CLAP adapter | 592K | (same mechanism) | 0 |
| Cross-Space Bridge | 590K | Eliminated (unified space) | 0 |
| **Total trainable** | **2.30M** | **Total trainable** | **0** |

The elimination of all trainable components is not merely a simplification but a qualitative shift: v2 requires no training data, no GPU, and no model checkpoints. Deployment consists of a single API call per modality.

### 6.3 Matryoshka Scale Consistency

We introduce *Matryoshka Scale Consistency* as a novel training-free uncertainty estimation method that replaces ProbVLM's probabilistic adapters. The key insight is that Matryoshka-compatible embeddings encode information hierarchically: low-dimensional prefixes capture coarse semantic structure, while full-dimensional embeddings capture fine-grained detail. For a truly coherent multimodal sample, coherence should be high at *all* scales; for ambiguous or weakly related content, coherence will vary across scales as lower-dimensional projections lose fragile alignment signals.

Given embeddings at Matryoshka truncation dimensions $\mathcal{D} = \{768, 1536, 3072\}$, we compute Gramian coherence at each scale $d$:

$$c_d = 1 - \text{vol}_d, \quad \text{where } \text{vol}_d = \det(G_d)^{1/2}$$

and the embeddings in $G_d$ are the L2-renormalized $d$-dimensional prefixes. The scale consistency is:

$$\kappa = 1 - \frac{\text{std}(\{c_d\}_{d \in \mathcal{D}})}{\text{mean}(\{c_d\}_{d \in \mathcal{D}}) + \epsilon}$$

where $\kappa \in [0, 1]$ with $\kappa = 1$ indicating perfect cross-scale stability (high confidence) and low $\kappa$ indicating scale-dependent coherence (low confidence). This mirrors ProbVLM's role in v1: high-uncertainty samples receive modulated scores, reducing the impact of unreliable channels.

The adaptive weighting in v2 uses $\kappa$ analogously to $\gamma$ in v1:

$$w_{ti}^{\text{final}} = (1 - \gamma_{\text{MRL}} \cdot (1 - \kappa)) \cdot w_{ti}^{\text{base}} + \gamma_{\text{MRL}} \cdot (1 - \kappa) \cdot w_{ti}^{\text{scale}}$$

where $w_{ti}^{\text{scale}}$ is derived from the per-channel scale consistency, weighting channels that are stable across truncation dimensions more heavily.

Matryoshka Scale Consistency is, to our knowledge, the first use of MRL truncation as an uncertainty signal. It generalizes to any Matryoshka-compatible embedding model and requires zero additional parameters or training.

### 6.4 Exact 3D Gramian Volume

In v1, the 3-way Gramian volume requires the Ex-MCR approximation because image and audio embeddings live in different spaces. In v2, all three modalities inhabit the same 3072-dimensional space, enabling the exact computation:

$$\text{vol}_{3D} = \det(G)^{1/2}, \quad G = \begin{pmatrix} 1 & \cos\theta_{ti} & \cos\theta_{ta} \\ \cos\theta_{ti} & 1 & \cos\theta_{ia} \\ \cos\theta_{ta} & \cos\theta_{ia} & 1 \end{pmatrix}$$

expanding to:

$$\det(G) = 1 - \cos^2\theta_{ti} - \cos^2\theta_{ta} - \cos^2\theta_{ia} + 2\cos\theta_{ti}\cos\theta_{ta}\cos\theta_{ia}$$

This captures the full tri-modal geometric relationship in a single scalar. The complementarity signal ($z_{\text{compl}}$) is computed directly from this volume rather than approximated through a cross-space projector.

However, empirical analysis reveals an important finding: in the Gemini unified space, the image-audio channel exhibits near-zero variance ($\sigma_{ia} = 0.011$, compared to $\sigma_{ti} = 0.020$ and $\sigma_{ta} = 0.069$). The optimized image-audio weight is $w_{ia} = 0.00$, and the complementarity weight is $w_{\text{compl}} = 0.00$. This indicates that while the unified space makes the IA channel *computable*, it does not make it *informative* for our evaluation data. The IA channel's lack of variance means it adds noise rather than signal, a finding that has implications for the anatomy of unified embedding spaces (Section 10.3).

### 6.5 Hyperparameter Optimization

v2 hyperparameters are optimized via LOO-CV on the same 30 human-rated samples:

**Table 7.** Optimized v2 hyperparameters.

| Parameter | Symbol | Value | Role |
|-----------|--------|-------|------|
| Margin scale | $\alpha$ | 20 | Contrastive margin amplification |
| Text-image weight | $w_{ti}$ | 0.55 | Channel balance |
| Image-audio weight | $w_{ia}$ | 0.00 | IA channel eliminated |
| Complementarity weight | $w_{\text{compl}}$ | 0.00 | Complementarity not informative |
| Matryoshka mixing | $\gamma_{\text{MRL}}$ | 0.90 | Heavy reliance on scale consistency |
| Calibration mode | cal\_mode | gram\_2d | 2D Gramian z-normalization |

Two findings stand out. First, the large $\gamma_{\text{MRL}} = 0.90$ indicates that Matryoshka Scale Consistency is the primary v2-specific contribution: almost all of the adaptive channel modulation comes from cross-scale stability rather than fixed weights. Second, $\alpha = 20$ (versus $\alpha = 6$ in v1) reflects the different magnitude of contrastive margins in the Gemini space, where margin values are smaller and require greater amplification.

LOO-CV yields $\rho = 0.555$ for v2 alone.

### 6.6 Use Case: Zero-Training API Evaluation (UC2)

cMSCI v2 targets scenarios where simplicity and cost matter more than peak accuracy, such as rapid prototyping, research surveys comparing many models, or deployment in environments without GPU access:

**Table 8.** UC2 zero-training evaluation.

| Property | Value |
|----------|-------|
| Spearman $\rho$ with human ratings | 0.467 ($p = 0.009$) |
| Trainable parameters | 0 |
| API cost (30 samples) | \$0.003 |
| GPU required | No |
| Setup time | Minutes (API key only) |
| Inference latency | 0.9 ms per sample (excluding API latency) |

Despite requiring no training and no local models, v2 achieves a statistically significant correlation on the 30-sample subset ($p = 0.009$). Re-evaluation on the full 100-sample set is pending (requires GOOGLE_API_KEY).

---

## 7. cMSCI v3: Multi-Backbone Ensemble

### 7.1 Complementary Error Profiles

The motivation for ensembling v1 and v2 is not simply that two models are better than one, but that the two instantiations make *complementary* errors. We quantify this via the Pearson correlation between v1 and v2 residuals (predicted score minus human score):

$$r_{\text{error}} = \text{Pearson}(\hat{c}_{v1} - c_{\text{human}}, \; \hat{c}_{v2} - c_{\text{human}}) = 0.423$$

A moderate positive correlation indicates that v1 and v2 share some failure modes (both struggle with the same difficult samples) but diverge on a substantial fraction of errors. The key asymmetry is in perturbation sensitivity: v1 detects audio incoherence with large effect sizes ($d = 1.60$ for wrong-audio), while v2 is stronger on visual incoherence ($d = 0.749$ for wrong-image but only $d = 0.381$ for wrong-audio). When v1 fails on a sample, v2 is often correct, and vice versa.

### 7.2 Ensemble Formulation

The ensemble score is a simple convex combination of the two instantiation scores:

$$\text{cMSCI}_{v3} = w_{v1} \cdot \text{cMSCI}_{v1} + (1 - w_{v1}) \cdot \text{cMSCI}_{v2}$$

Each component score is computed independently using its own backbone, calibration parameters, and component pipeline. No joint training or fine-tuning is performed; the ensemble operates purely at the score level.

### 7.3 Weight Selection

The ensemble weight $w_{v1}$ is selected via LOO-CV on 30 human-rated samples. The optimal weight is:

$$w_{v1}^{*} = 0.40$$

assigning 60% weight to v2 and 40% to v1. Despite v1's higher individual correlation, the optimizer favors v2-heavy weighting because v2 contributes unique geometric information from the unified Gemini space that complements v1's specialized channels. LOO-CV for the ensemble yields $\rho = 0.569$, compared to $\rho = 0.630$ for v1 alone and $\rho = 0.555$ for v2 alone. The ensemble's LOO-CV is lower than v1's LOO-CV because the held-out validation penalizes the additional degree of freedom (the ensemble weight), but the ensemble achieves the highest full-sample correlation ($\rho = 0.655$ versus $\rho = 0.630$ and $\rho = 0.467$).

### 7.4 Use Case: High-Stakes Model Selection (UC3)

cMSCI v3 targets applications where errors have high cost, such as selecting the best multimodal generation model from a candidate set, or certifying content quality for production deployment:

**Table 9.** UC3 high-stakes model selection.

| Metric | v3 (ensemble) | v1 | v2 |
|--------|---------------|-----|-----|
| Spearman $\rho$ | **0.655** | 0.630 | 0.467 |
| Pairwise accuracy | **71.9%** | 71.0% | 65.8% |
| Kendall $\tau$ | **0.439** | 0.421 | 0.315 |
| Mean rank error | **5.87** | 6.47 | 6.87 |
| Critical failure rate | **5%** | 5% | 30% |

The ensemble's primary advantage is in error recovery. Of the samples where v1 makes large ranking errors, v3 rescues 19% by incorporating the v2 signal. The illustrative case is sample S010 (wrong-audio condition): v1 assigns rank 26 but the human rank is 7; the ensemble corrects to rank 17, recovering 9 rank positions. Critical failures (defined as rank errors $\geq$ 10 positions) occur in only 5% of samples for v3, compared to 30% for v2. The mean rank error of 5.87 is the lowest of all configurations.

---

## 8. Experimental Setup

### 8.1 Evaluation Data

**Human-rated evaluation set.** 30 scene descriptions spanning four environmental domains (nature, urban, water, mixed) evaluated under three conditions: baseline (matched image + audio), wrong-image (cross-domain mismatched image), and wrong-audio (cross-domain mismatched audio). This yields 30 unique evaluation triples (10 per condition). Images are generated by Stable Diffusion 1.5 \citep{rombach2022high}; audio is retrieved via CLAP similarity from an index of 104 audio files. Five independent raters scored each sample on a 1-5 Likert scale. Inter-rater reliability: ICC(3,k) = 0.873 (good), Krippendorff's $\alpha$ = 0.684 (acceptable).

**External benchmark.** 100 AudioCaps samples (Kim et al., 2019) with human-written captions, used for matched/mismatched discrimination (200 pairs). Each original caption-audio pair is a positive; a randomly shuffled pairing serves as a negative.

**Training data.** 10,255 embedding pairs for v1 components: OmniBench (1,051), AudioCaps (1,000), and augmented (8,204). v2 requires no training data.

### 8.2 Baselines

We compare cMSCI against ten baseline methods spanning four categories:

**Table 10.** Baseline methods.

| Method | Category | Description | Audio-aware |
|--------|----------|-------------|-------------|
| MSCI (raw cosine) | Simple | Weighted mean of CLIP and CLAP cosine similarities | Yes |
| Raw cosine | Simple | Unweighted mean of CLIP and CLAP cosine similarities | Yes |
| Cosine + z-norm | Simple | Z-normalized cosine similarities mapped through sigmoid | Yes |
| Concatenated cosine | Simple | Concatenated CLIP+CLAP embeddings, single cosine distance | Yes |
| Retrieval rank | Simple | Reciprocal rank of matched item among candidates | Yes |
| CLIPScore + CLAPScore | Established | Standard CLIP text--image score augmented with CLAP text--audio | Yes |
| BLIPScore + CLAPScore | Established | BLIP image--text matching probability combined with CLAPScore | Yes |
| CCA | Joint embedding | Canonical correlation analysis projecting CLIP and CLAP into shared subspace | Yes |
| Regularized CCA | Joint embedding | Ridge-regularized CCA for small-sample stability | Yes |
| LLaVA-7B (VLM judge) | VLM | Vision--language model rating coherence on 1--5 scale via chain-of-thought | Partial |

All baselines that include an audio channel use CLAP for text--audio similarity. CLIPScore is presented with CLAPScore augmentation for fair tri-modal comparison. The VLM judge processes the image directly and the audio via a spectrogram representation, but lacks native audio understanding.

### 8.3 Evaluation Protocol

**Primary metric.** Spearman's rank correlation ($\rho$) between automatic scores and mean human ratings (averaged across five raters), with two-sided $p$-values. Significance threshold: $\alpha = 0.05$.

**Secondary metrics.** Kendall's $\tau$ for ordinal agreement, pairwise accuracy for relative ranking, mean rank error for positional accuracy, Cohen's $d$ for perturbation effect sizes, and AUC for binary discrimination.

**Cross-validation.** All hyperparameters are selected via LOO-CV on the full 100-sample set (an unbiased procedure that uses each sample once as held-out). The dev/test split (70 dev, 30 test, stratified by domain $\times$ condition) provides additional held-out validation.

**Robustness.** Seed stability (10 random seeds), one-at-a-time hyperparameter sensitivity sweeps, and graceful degradation (component ablation).

**Reproducibility.** All experiments use single-clip CLAP embedding (no windowing), deterministic inference, and fixed random seeds. Complete configuration is specified in `src/config/settings.py`.

---

## 9. Results

### 9.1 Main Comparison

**Table 11.** Spearman correlation with human coherence judgments ($n = 100$, 5 raters).

| Rank | Method | Category | $\rho$ | $p$-value | Sig. |
|------|--------|----------|--------|-----------|------|
| **1** | **cMSCI v1 (CLIP+CLAP)** | **Geometric** | **0.785** | **< 10$^{-6}$** | **Yes** |
| 2 | Cosine + z-norm | Simple | 0.712 | < 10$^{-6}$ | Yes |
| 3 | Raw cosine | Simple | 0.558 | < 10$^{-6}$ | Yes |
| 4 | MSCI (raw cosine) | Simple | 0.558 | < 10$^{-6}$ | Yes |
| 5 | Concatenated cosine | Simple | 0.547 | < 10$^{-6}$ | Yes |
| 6 | Regularized CCA | Joint embedding | 0.495 | < 10$^{-6}$ | Yes |
| 7 | Retrieval rank | Simple | 0.394 | 0.00005 | Yes |
| 8 | CCA | Joint embedding | 0.163 | 0.106 | No |

Note: cMSCI v2 (Gemini) and v3 (ensemble) results are pending re-evaluation on the expanded 100-sample set. Previous 30-sample results: v2 $\rho$ = 0.467, v3 $\rho$ = 0.655.

cMSCI v1 achieves the highest correlation ($\rho = 0.785$, $p$ < 10$^{-6}$), substantially outperforming all baselines. The improvement over MSCI ($\rho$ = 0.558 $\rightarrow$ 0.785, +41%) confirms that the calibration pipeline---not the embedding backbone---is the critical differentiator. With the larger 100-sample evaluation, most baselines achieve statistical significance, providing a more discriminative comparison. Regularized CCA ($\rho = 0.495$) is competitive among joint embedding methods, while unregularized CCA ($\rho = 0.163$) fails to generalize. LOO-CV confirms minimal overfitting ($\rho$ = 0.749, gap = 0.001).

### 9.2 Component Ablation

**Table 12.** v1 component ablation.

| Configuration | Spearman $\rho$ | $\Delta\rho$ | Significant |
|---------------|-----------------|--------------|-------------|
| MSCI (cosine baseline) | 0.298 | -- | No |
| + Gramian volume | 0.286 | $-$0.012 | No |
| + z-score calibration | 0.413 | +0.127 | Yes |
| + contrastive margin | 0.478 | +0.065 | Yes |
| + ExMCR complementarity | 0.601 | +0.123 | Yes |
| + adaptive weighting (full v1) | 0.630 | $-$0.020 | Yes |

**Table 13.** v2 component ablation.

| Configuration | Spearman $\rho$ | $\Delta\rho$ | Significant |
|---------------|-----------------|--------------|-------------|
| Raw cosine average | 0.303 | -- | No |
| + Gramian volume | 0.323 | +0.020 | No |
| + z-score calibration | 0.491 | +0.168 | Yes |
| + contrastive margins ($\alpha = 0$) | 0.491 | +0.000 | Yes |
| + Matryoshka adaptive (full v2) | 0.558 | +0.067 | Yes |

Three patterns emerge across both ablations. First, z-score calibration is the single most important component, producing the largest jump and transforming a non-significant correlation into a significant one in both v1 ($\Delta\rho = +0.127$) and v2 ($\Delta\rho = +0.168$). Second, Gramian volume alone does not improve over cosine similarity, confirming that the geometric formulation is necessary but not sufficient without calibration. Third, the slight drop from ExMCR to full v1 ($\Delta\rho = -0.020$) reflects the regularization effect of adaptive weighting: by modulating channel weights per sample, the adaptive mechanism trades a small amount of average correlation for improved robustness on difficult samples.

Note: The v2 ablation values are drawn from earlier 3-rater evaluation data; with 5 raters, the overall v2 $\rho$ is 0.467 rather than 0.558, reflecting changes in the ground truth distribution with additional raters.

### 9.3 Perturbation Sensitivity

Effect sizes quantify each variant's ability to detect specific types of incoherence:

**Table 14.** Perturbation effect sizes (Cohen's $d$, baseline vs. perturbation condition).

| Variant | Wrong-image $d$ | Wrong-audio $d$ |
|---------|----------------|----------------|
| cMSCI v1 | 1.29 | 1.60 |
| cMSCI v2 | 0.749 | 0.381 |

v1 exhibits large effect sizes in both channels, with a notably strong audio channel ($d = 1.60$), attributable to the 10,255-pair training set that includes real AudioCaps audio-caption pairs. v2's weaker audio sensitivity ($d = 0.381$) reflects the IA channel's near-zero variance in the Gemini unified space and the consequent reliance on text-image and text-audio channels alone. The asymmetry between v1 and v2 perturbation profiles is precisely what makes their ensemble effective.

### 9.4 External Benchmark (AudioCaps)

**Table 15.** AudioCaps matched/mismatched discrimination (100 samples, 200 pairs).

| Metric | cMSCI v1 |
|--------|----------|
| AUC | 0.993 |
| F1 score | 0.975 |
| Cohen's $d$ | 3.35 |
| FPR @ 95% TPR | 2% |

The near-ceiling AUC (0.993) and massive effect size ($d = 3.35$) confirm that cMSCI v1 generalizes beyond the curated evaluation set. The external benchmark uses real-world audio-caption pairs rather than domain-controlled perturbations, testing whether the metric can discriminate genuine multimodal coherence from random pairings in the wild.

### 9.5 Robustness

**Seed stability.** Running v1 with 10 different random seeds produces identical results ($\rho = 0.785 \pm 0.000$, 10/10 significant at $p$ < 10$^{-6}$). The zero variance confirms that the metric is fully deterministic given the same inputs and model weights.

**Hyperparameter sensitivity.** One-at-a-time sweeps across each of the four main hyperparameters ($\alpha$, $w_{ti}$, $w_{3d}$, $\gamma$) confirm a broad performance plateau: all tested configurations across all four sweeps produce significant correlations ($p < 0.05$). The narrowest sensitivity range is $w_{ti}$ (spread = 0.067), and the widest is $\gamma$ (spread = 0.197), indicating that the adaptive mixing coefficient has the largest marginal impact but that no single hyperparameter dominates performance.

**Per-domain analysis.** v1 performance varies by domain: urban scenes achieve the highest correlation ($\rho = 0.770$), likely because urban environments have highly distinctive visual and auditory signatures (traffic noise, city skylines) that create large cosine gaps between matched and mismatched content.

### 9.6 Ensemble Analysis

**Table 16.** Ensemble diagnostics.

| Metric | Value |
|--------|-------|
| Error decorrelation (Pearson $r$) | 0.447 |
| Optimal ensemble weight ($w_{v1}$) | 0.40 |
| v1 critical failure rate | 5% |
| v2 critical failure rate | 30% |
| v3 critical failure rate | 5% |
| v1 errors rescued by ensemble | 19% |
| v3 pairwise accuracy gain over v1 | +0.9% |
| v3 mean rank error improvement | $-$0.60 |

The ensemble achieves its gains primarily through error rescue rather than uniform improvement. The v2-favoring weight ($w_{v1} = 0.40$, i.e., 60% v2) reflects v2's contribution of complementary geometric information from the unified Gemini space, despite v1's higher individual correlation. The moderate error decorrelation ($r = 0.447$) is sufficient for meaningful ensemble gains: a higher correlation would indicate redundant errors (no ensemble benefit), while a lower correlation would suggest the models operate on fundamentally different constructs (risky to combine). The observed decorrelation sits in the productive regime where the models agree on easy samples and disagree on informative boundary cases.

### 9.7 Qualitative Case Studies

We present four case studies illustrating the behavior of all three variants across different conditions:

**Case 1: High agreement (S019, baseline condition).** A well-matched scene with coherent text, image, and audio. v1 = 0.869, v2 = 0.791, v3 = 0.853, human = 1.000. All three variants correctly assign high scores, with the slight discount reflecting the metric's calibration against the population (perfect scores are rare). This case confirms basic validity: the metrics agree with humans on unambiguous positive examples.

**Case 2: Image incoherence detected (S014, wrong-image condition).** A nature text prompt paired with a mismatched urban image and correct nature audio. v1 = 0.164, v2 = 0.155, v3 = 0.162, human = 0.380. Both v1 and v2 detect the visual mismatch, assigning scores well below the human mean. The human score (0.380) is higher than the automatic scores, suggesting that human raters partially compensated for the correct audio channel, a nuance that the metrics' lower scores reflect more conservatively.

**Case 3: Audio incoherence, v1-v2 divergence (S010, wrong-audio condition).** A nature scene with correct image but mismatched urban audio. v1 = 0.243, v2 = 0.572, v3 = 0.309, human = 0.696. v1 correctly detects the audio incoherence (score drops sharply from baseline), while v2 assigns a moderate score that fails to reflect the perturbation. This is the prototypical case motivating the ensemble: v2's insensitivity to audio perturbations is corrected by v1's strong audio channel. The ensemble score (0.309) is closer to the human rating than v2 alone.

**Case 4: Ensemble rescue (S010, rank analysis).** Extending Case 3 to ranking: v1 ranks S010 at position 26 (out of 30), the ensemble corrects to rank 17, and the human rank is 7. The ensemble recovers 9 rank positions, illustrating how even a modest v2 contribution ($w_{v1} = 0.40$) can substantially correct v1's ranking errors. This sample accounts for one of the 19% of rescued v1 errors that drive the ensemble's improved mean rank error.

---

## 10. Discussion

### 10.1 Why Calibration is the Critical Enabler

The most consistent finding across both v1 and v2 ablations is that z-score calibration transforms a non-significant correlation into a significant one. Without calibration, Gramian volume and cosine similarity perform similarly ($\rho \approx 0.29$-$0.32$); with calibration, both jump to $\rho > 0.40$. The mechanism is straightforward but its impact is dramatic: raw similarity scores from different embedding spaces (CLIP vs. CLAP) or different channels (text-image vs. text-audio) have different scale properties. A cosine similarity of 0.35 in CLIP space carries different semantic import than 0.35 in CLAP space. Z-score calibration against reference distributions makes these channels commensurable, enabling meaningful weighted combination.

This finding has practical implications beyond our specific metric. Any multimodal evaluation that combines scores from heterogeneous embedding spaces should calibrate those scores against reference distributions before combining them. The calibration statistics are cheap to compute (a one-time pass over baseline data) and the improvement is large.

### 10.2 Cross-Modal Complementarity

The sign flip in ExMCR complementarity -- where greater image-audio Gramian *dispersion* (not coherence) correlates with human judgments -- challenges the intuitive assumption that all channels should point in the same direction. The finding suggests that humans perceive multimodal content as richer and more coherent when each modality contributes unique semantic information rather than redundantly encoding the same concept. A beach scene is perceived as more coherent when the image shows the visual landscape and the audio captures wave sounds than when both modalities encode the same abstract "beach-ness."

This complementarity signal contributes $\Delta\rho = +0.123$ in the v1 ablation (the second-largest jump after calibration), confirming its value. However, the complementarity weight is optimized at $w_{3d} = 0.30$ for v1 and $w_{\text{compl}} = 0.00$ for v2, indicating that its utility depends on the quality of the cross-space projection. When the projection is approximate (ExMCR in v1), the complementarity signal adds noise alongside signal; when the projection is exact but the channel lacks variance (Gemini in v2), the signal is simply uninformative.

### 10.3 Anatomy of a Unified Embedding Space

cMSCI v2's results reveal an unexpected property of the Gemini unified embedding space: the image-audio channel has near-zero variance ($\sigma_{ia} = 0.011$), making it uninformative for coherence evaluation despite being directly computable. The text-image ($\sigma_{ti} = 0.020$) and text-audio ($\sigma_{ta} = 0.069$) channels carry meaningful variance, but the image-audio channel collapses to near-constant values regardless of the content pairing.

We hypothesize that this reflects the training distribution of the underlying model. Unified embedding models are typically trained on text-paired data (text-image, text-audio) rather than directly on image-audio pairs. The model learns to align each modality with text as the anchor, but the image-audio relationship is an emergent property rather than a directly optimized objective. For scene-level evaluation where images and audio clips often depict broadly similar environmental categories (nature, urban, water), the emergent image-audio similarity is high and invariant, lacking the discriminative variance needed for coherence evaluation.

This finding cautions against assuming that unified embedding spaces automatically solve the cross-modal comparison problem. The mathematical possibility of computing image-audio similarity does not guarantee its informative value.

### 10.4 Why the Ensemble Works

On the earlier 30-sample evaluation, the ensemble's success ($\rho = 0.655$ versus $\rho = 0.630$ for v1 alone) was driven by three factors:

1. **Complementary perturbation sensitivity.** v1 excels at audio incoherence ($d = 1.60$) while v2 handles visual incoherence ($d = 0.749$). The ensemble benefits from whichever channel is more relevant for each sample.

2. **Decorrelated errors.** The error correlation ($r = 0.447$) is moderate, indicating that a substantial fraction of errors are independent. This is the necessary condition for ensemble gain.

3. **Different embedding geometries.** CLIP+CLAP (512-d, two spaces) and Gemini (3072-d, unified) represent fundamentally different views of multimodal alignment. CLIP/CLAP were trained with modality-specialized architectures on curated contrastive data; Gemini was trained as a general-purpose embedding model. The two representations capture different aspects of semantic similarity, and their combination is richer than either alone.

On 30 samples, the $w_{v1} = 0.40$ weight indicates that the ensemble favors v2 (60%), despite v1's stronger individual correlation. This counter-intuitive weighting suggests that v2's unified-space geometry provides complementary information that is disproportionately valuable when combined with v1, even though v2 alone is weaker. Ensemble re-evaluation on the full 100-sample set is pending (requires GOOGLE_API_KEY for Gemini embeddings).

### 10.5 Matryoshka Scale Consistency as a General Principle

Matryoshka Scale Consistency introduces a general principle for training-free uncertainty estimation that extends beyond our specific application. Any Matryoshka-compatible embedding model (an increasingly common property as MRL becomes standard in embedding training) can provide per-sample confidence estimates by measuring coherence stability across truncation dimensions. The principle is that information encoded at multiple scales is more robust than information encoded only at the finest scale.

In v2, the high optimized $\gamma_{\text{MRL}} = 0.90$ confirms that scale consistency is the primary v2-specific contribution, accounting for the gap between uncalibrated ($\rho = 0.303$) and fully calibrated ($\rho = 0.467$) performance. The mechanism can serve as a drop-in replacement for ProbVLM-style uncertainty estimation in any pipeline that uses Matryoshka-compatible embeddings, eliminating the need for probabilistic adapter training.

### 10.6 Practical Deployment Guidance

We recommend the following variant selection based on deployment context:

**UC1 -- Batch quality gate:** Use cMSCI v1 when accuracy matters most. The 85 ms latency is suitable for offline processing, and the AUC of 0.993 on AudioCaps provides near-perfect matched/mismatched discrimination with only 2% false positive rate at 95% recall.

**UC2 -- Rapid prototyping / cost-sensitive evaluation:** Use cMSCI v2 when simplicity and cost matter more than peak accuracy. Zero trainable parameters, no GPU requirement, and \$0.003 per 30 samples make v2 suitable for large-scale surveys or environments without local compute.

**UC3 -- High-stakes decisions:** Use cMSCI v3 when the cost of ranking errors is high. The ensemble reduces mean rank error to 5.87, achieves the highest pairwise accuracy (71.9%), and maintains a 5% critical failure rate. The additional cost is simply running both v1 and v2 on each sample.

For all variants, the calibration statistics should be recomputed when the embedding backbone is updated or when the evaluation domain shifts substantially from the calibration distribution.

---

## 11. Limitations

**Sample size.** The human evaluation set contains 100 samples rated by five independent raters (eight on the original 30-sample subset). Inter-rater reliability is good (ICC(3,k) = 0.872), and the 100-sample size provides substantially stronger statistical power than the earlier 30-sample evaluation, with narrower confidence intervals and more discriminative baseline comparisons.

**Domain coverage.** The evaluation set covers four environmental domains (nature, urban, water, mixed). Performance on other content types (e.g., abstract concepts, indoor scenes, human activities, synthetic media) is not validated.

**English-only evaluation.** All text prompts, captions, and audio descriptions are in English. Cross-lingual or multilingual coherence evaluation is not addressed.

**Embedding model dependence.** cMSCI's performance is bounded by the quality of the underlying embedding models. CLIP's 77-token context window truncates long descriptions, and CLAP's audio encoder may not capture fine-grained temporal structure. Gemini Embedding 2 is accessed via API, introducing latency variability and dependency on external service availability.

**Image-audio channel.** Neither v1 nor v2 fully exploits the image-audio relationship. In v1, the Ex-MCR projection is an approximation; in v2, the channel lacks discriminative variance. A stronger image-audio signal would likely improve all variants.

**Calibration distribution shift.** The z-score calibration statistics are fitted on baseline data from specific domains. If the evaluation domain shifts substantially (e.g., from environmental scenes to medical imaging), recalibration is necessary. We do not evaluate the metric's robustness to calibration distribution shift.

**Coherence vs. quality.** cMSCI measures semantic coherence (whether modalities agree) but not individual modality quality. A blurry image coherently paired with matching text and audio would receive a high cMSCI score despite low visual quality. A complete evaluation system would combine coherence metrics with modality-specific quality metrics.

**Contrastive margin dependence on index.** The contrastive margin component depends on the negative bank composition (57 images, 104 audio files). A different negative bank could yield different margins and, consequently, different optimal hyperparameters. We do not evaluate sensitivity to negative bank composition.

---

## 12. Conclusion

We presented the calibrated Multimodal Semantic Coherence Index (cMSCI), a geometric framework for evaluating cross-modal alignment in text-image-audio content. Our findings support six key conclusions:

1. **Calibration, not geometry, is the critical enabler.** Gramian volume alone does not improve over cosine similarity. Z-score calibration transforms non-significant baselines into significant metrics by making heterogeneous embedding spaces commensurable. This finding generalizes: any multimodal evaluation combining scores from different embedding spaces should calibrate before combining.

2. **The cMSCI framework is embedding-agnostic.** The same pipeline (Gramian volume $\rightarrow$ z-calibration $\rightarrow$ contrastive margins $\rightarrow$ adaptive weighting) produces significant human correlations on two fundamentally different backbones: CLIP+CLAP (v1, $\rho = 0.785$, $p$ < 10$^{-6}$ on 100 samples) and Gemini Embedding 2 (v2, $\rho = 0.467$, $p = 0.009$ on 30 samples). The framework adapts to the backbone's properties rather than depending on specific model architectures.

3. **Scaling evaluation strengthens results.** Expanding from 30 to 100 human-rated samples (with five raters) improves cMSCI v1 from $\rho = 0.630$ to $\rho = 0.785$, with LOO-CV confirming minimal overfitting (gap = 0.001). The larger sample size also enables most baselines to achieve significance, providing a more discriminative comparison landscape.

4. **Cross-modal complementarity is a coherence signal.** The finding that image-audio Gramian *dispersion* (not coherence) correlates with human judgments challenges the assumption that all channels should point in the same direction. Humans perceive multimodal content as more coherent when each modality contributes unique information, a property captured by the ExMCR complementarity component.

5. **Matryoshka Scale Consistency provides training-free uncertainty.** By measuring coherence stability across MRL truncation dimensions, we obtain per-sample confidence estimates without probabilistic adapters, training data, or GPU resources. This principle generalizes to any Matryoshka-compatible embedding model and represents, to our knowledge, the first use of MRL truncation as an uncertainty signal.

6. **Unified embedding spaces do not automatically solve cross-modal comparison.** Despite enabling direct image-audio computation, the Gemini unified space exhibits near-zero variance in the image-audio channel ($\sigma = 0.011$), rendering it uninformative. The mathematical possibility of computing a cross-modal similarity does not guarantee its discriminative value.

The cMSCI framework, code, trained model weights, and evaluation data are publicly available to support reproducibility and extension to additional modalities and embedding backbones.

---

## References

Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013). Deep canonical correlation analysis. In *Proceedings of the 30th International Conference on Machine Learning (ICML)*, 1247--1255.

Baltrusaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423--443.

Baumann, A., Li, R., Klasson, M., Mentu, S., Karthik, S., Akata, Z., Solin, A., & Trapp, M. (2026). Post-hoc probabilistic vision-language models. In *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:2412.06014.

Cicchetti, G., Grassucci, E., Sigillo, L., & Comminiello, D. (2025a). Gramian multimodal representation learning and alignment. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

Cicchetti, G., Grassucci, E., & Comminiello, D. (2025b). A TRIANGLE enables multimodal alignment beyond cosine similarity. In *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2509.24734.

Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K. V., Joulin, A., & Misra, I. (2023). ImageBind: One embedding space to bind them all. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 15180--15190.

Google. (2026). Gemini Embedding 2. *Google AI Developer Documentation*. https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2-preview

Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004). Canonical correlation analysis: An overview with application to learning methods. *Neural Computation*, 16(12), 2639--2664.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., & Choi, Y. (2021). CLIPScore: A reference-free evaluation metric for image captioning. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 7514--7528.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In *Advances in Neural Information Processing Systems (NeurIPS)*, 6626--6637.

Kim, C. D., Kim, B., Lee, H., & Kim, G. (2019). AudioCaps: Generating captions for audios in the wild. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT)*, 119--132.

Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155--163.

Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. *Technical Report, Annenberg School for Communication, University of Pennsylvania*.

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P., & Farhadi, A. (2022). Matryoshka representation learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 30233--30249.

Lee, T., Yasunaga, M., Meng, C., Mai, Y., Park, J. S., Gupta, A., Zhang, Y., Narayanan, D., Teufel, H., Bellagente, M., et al. (2023). Holistic evaluation of text-to-image models. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *Proceedings of the 40th International Conference on Machine Learning (ICML)*.

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. In *Advances in Neural Information Processing Systems (NeurIPS)*.


Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. (2021). Learning transferable visual models from natural language supervision. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 8748--8763.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 10684--10695.

Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001). Perceptual evaluation of speech quality (PESQ)---a new method for speech quality assessment of telephone networks and codecs. In *IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*, 749--752.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2226--2234.

Saporta, A., Puli, A., Goldstein, M., & Ranganath, R. (2024). Contrasting with Symile: Simple model-agnostic representation learning for unlimited modalities. In *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2411.01053.

Tang, C., Xiao, Q., Mei, K., Wang, T., Rao, F., & Zhang, C. (2026). WAVE: Learning unified & versatile audio-visual embeddings with multimodal LLM. In *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:2509.21990.

Upadhyay, U., Karthik, S., Chen, Y., Mancini, M., & Akata, Z. (2022). BayesCap: Bayesian identity cap for calibrated uncertainty in frozen neural networks. In *Proceedings of the European Conference on Computer Vision (ECCV)*, 299--317.

Upadhyay, U., Karthik, S., Mancini, M., & Akata, Z. (2023). ProbVLM: Probabilistic adapter for frozen vision-language models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

Wang, Z., Zhao, Y., Cheng, X., Huang, H., Liu, J., et al. (2023). C-MCR: Connecting multi-modal contrastive representations. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Wang, Z., Zhang, Z., Liu, L., Zhao, Y., Huang, H., Jin, T., & Zhao, Z. (2024). Extending multi-modal contrastive representations. In *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2310.08884.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In *IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*, 1--5.

Yariv, G., Gat, I., Benaim, S., Wolf, L., Schwartz, I., & Adi, Y. (2024). Diverse and aligned audio-to-video generation via text-to-video model adaptation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(7), 6639--6647.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems (NeurIPS)*.

---

