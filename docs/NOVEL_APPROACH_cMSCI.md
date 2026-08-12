# cMSCI: Calibrated Multimodal Semantic Coherence Index

## Novel Research Approach Document

**Date:** February 2026
**Status:** Locked — Ready for Implementation
**Authors:** [Your Name]

---

## 1. Executive Summary

We propose **cMSCI** (Calibrated Multimodal Semantic Coherence Index), an uncertainty-aware, geometrically-grounded evaluation metric for tri-modal (text-image-audio) coherence. cMSCI extends three published methods — GRAM, C-MCR, and ProbVLM — in ways their original authors did not address, and combines them into a unified coherence evaluation framework that does not exist in prior work.

The core idea: replace the current ad-hoc weighted average of pairwise cosine similarities with a **principled geometric measure** that operates across **heterogeneous embedding spaces**, provides **confidence intervals** via probabilistic embeddings, and is **calibrated** against contrastive negative banks.

---

## 2. Problem Statement

### 2.1 Current MSCI Formulation

The existing MSCI computes:

```
st_i = cos(text_clip, image_clip)       # CLIP space
st_a = cos(text_clap, audio_clap)       # CLAP space
MSCI = 0.45 * st_i + 0.45 * st_a       # weighted average (si_a omitted without bridge)
```

### 2.2 Three Fundamental Limitations

**Limitation 1: Incomparable Scales**

CLIP cosine similarities and CLAP cosine similarities have different distributions (different means, variances, and shapes). A CLIP score of 0.30 does not carry the same semantic meaning as a CLAP score of 0.30. Averaging them treats them as equivalent, which is statistically unprincipled.

This is a well-documented issue. Liang et al. (NeurIPS 2022) show that different modalities in CLIP occupy narrow cones separated by a "modality gap." CLIP and CLAP, being entirely different models, have different gap structures and temperature parameters, making their raw scores fundamentally non-comparable.

**Limitation 2: Point Estimates Without Confidence**

MSCI produces a single scalar (e.g., 0.72) with no indication of reliability. An ambiguous image that happens to score 0.72 is treated identically to a clear, unambiguous image that also scores 0.72. There is no way to distinguish confident evaluations from unreliable ones.

**Limitation 3: Pairwise Aggregation Misses Joint Alignment**

Symile (Patel et al., NeurIPS 2024) proves mathematically that pairwise contrastive alignment cannot capture joint three-way dependencies between modalities. Our weighted average of st_i and st_a is exactly this: two pairwise scores combined post-hoc. Information that exists only in the joint text-image-audio relationship is invisible to this formulation.

---

## 3. Proposed Solution: cMSCI

cMSCI addresses all three limitations through three layered extensions:

```
Layer 1: Heterogeneous-Space Geometric Alignment (extends GRAM)
Layer 2: Probabilistic Geometric Coherence (extends ProbVLM)
Layer 3: Contrastive Geometric Calibration (extends negCLIPLoss)
```

### 3.1 Architecture Overview

```
Input: (text, image, audio)
    |
    v
[Embedding Extraction]
    text_clip  = CLIP_text(text)          # 512-d, CLIP space
    text_clap  = CLAP_text(text)          # 512-d, CLAP space
    image_clip = CLIP_image(image)        # 512-d, CLIP space
    audio_clap = CLAP_audio(audio)        # 512-d, CLAP space
    |
    v
[Layer 1: Space Alignment + Normalization]
    audio_aligned = C-MCR(audio_clap)     # Project CLAP audio -> CLIP space
    text_norm     = z_normalize(text_clip, ref_dist_text)
    image_norm    = z_normalize(image_clip, ref_dist_image)
    audio_norm    = z_normalize(audio_aligned, ref_dist_audio)
    |
    v
[Layer 2: Probabilistic Embedding]
    text_dist  = ProbVLM(text_norm)       # mean + variance
    image_dist = ProbVLM(image_norm)      # mean + variance
    audio_dist = ProbVLM(audio_norm)      # mean + variance
    |
    v
[Layer 3: Geometric Coherence + Calibration]
    For i in 1..N (Monte Carlo samples):
        t_i ~ text_dist
        m_i ~ image_dist
        a_i ~ audio_dist
        G_i = gram_matrix([t_i, m_i, a_i])    # 3x3 Gram matrix
        vol_i = sqrt(det(G_i))                  # Gramian volume

    raw_coherence = 1 - mean(vol_i) / vol_max   # Normalize: 0 = no alignment, 1 = perfect
    confidence    = std(vol_i)                    # Uncertainty from MC sampling

    # Contrastive calibration
    neg_volumes = [gram_volume(text, neg_image_k, neg_audio_k) for k in neg_bank]
    margin      = raw_coherence - mean(neg_volumes)

    cMSCI = margin       # Final calibrated score
    cMSCI_ci = confidence # Confidence interval
    |
    v
Output: cMSCI score + confidence interval + sub-scores
```

---

## 4. Detailed Method Description

### 4.1 Layer 1: Heterogeneous-Space Geometric Alignment

#### Problem
GRAM (Cicchetti et al., ICLR 2025) computes the Gramian volume of the parallelotope spanned by modality vectors. The Gram matrix G for vectors v1, v2, ..., vk is:

```
G[i,j] = <vi, vj>    (dot product)
Volume = sqrt(det(G))
```

When all vectors are perfectly aligned, the volume approaches zero (parallelotope collapses). When vectors are orthogonal, the volume is maximized.

GRAM assumes all vectors live in the same embedding space. Our vectors don't: CLIP image embeddings and CLAP audio embeddings occupy different 512-d spaces with no shared geometry.

#### Our Extension

**Step 1: Cross-space projection via C-MCR**

C-MCR (Wang et al., NeurIPS 2023) connects CLIP and CLAP spaces using their shared text modality. Given:
- CLIP space: (text, image) pairs are aligned
- CLAP space: (text, audio) pairs are aligned
- Overlap: text exists in both spaces

C-MCR learns projections that map CLAP embeddings into CLIP space (or vice versa) without requiring any paired image-audio data. Pre-trained weights are available.

After projection, all three modalities (text, image, audio) live in a shared CLIP-derived space.

**Step 2: Distribution normalization**

Even after projection, the three channels may have different score distributions. We compute reference distributions from our experimental data (270 runs from RQ1):

```python
mu_text, sigma_text = mean(text_embeddings), std(text_embeddings)
mu_image, sigma_image = mean(image_embeddings), std(image_embeddings)
mu_audio, sigma_audio = mean(audio_projected), std(audio_projected)

text_norm = (text_emb - mu_text) / sigma_text
image_norm = (image_emb - mu_image) / sigma_image
audio_norm = (audio_projected - mu_audio) / sigma_audio
```

This ensures each channel contributes equally to the geometric computation regardless of its raw scale.

**Step 3: Gramian volume computation**

With all vectors in a shared, normalized space:

```python
G = np.array([
    [dot(text, text),   dot(text, image),  dot(text, audio)],
    [dot(image, text),  dot(image, image), dot(image, audio)],
    [dot(audio, text),  dot(audio, image), dot(audio, audio)]
])
volume = np.sqrt(np.abs(np.linalg.det(G)))
coherence = 1 - volume / volume_max  # Normalized to [0, 1]
```

Smaller volume = tighter alignment = higher coherence.

#### What Makes This Novel
- GRAM was designed as a **training loss** for learning aligned representations. We use it as an **evaluation metric** for scoring coherence of existing content.
- GRAM assumes a **homogeneous** embedding space. We extend it to **heterogeneous** spaces (CLIP + CLAP) via C-MCR projection + distribution normalization.
- The specific pipeline (project -> normalize -> geometric measure) for tri-modal coherence evaluation does not exist in prior work.

### 4.2 Layer 2: Probabilistic Geometric Coherence

#### Problem
GRAM volume on point embeddings gives a single number. We have no way to know if that number is reliable.

#### Our Extension

ProbVLM (Upadhyay et al., ICCV 2023) is a lightweight post-hoc adapter that converts frozen CLIP/CLAP point embeddings into probability distributions (Gaussian: mean + covariance). It requires no changes to the underlying models.

We wrap our embedders with ProbVLM adapters:

```python
# Instead of point embeddings:
text_point = clip.encode_text(text)       # Single 512-d vector

# We get distributions:
text_mean, text_var = probvlm.encode_text(text)  # Mean + variance in 512-d
```

Then compute the **expected GRAM volume** via Monte Carlo sampling:

```python
N = 1000  # Monte Carlo samples
volumes = []
for _ in range(N):
    t = sample(text_mean, text_var)
    m = sample(image_mean, image_var)
    a = sample(audio_mean, audio_var)
    G = gram_matrix([t, m, a])
    volumes.append(sqrt(abs(det(G))))

cMSCI_raw = 1 - mean(volumes) / vol_max
cMSCI_confidence = std(volumes)
```

This yields: **cMSCI = 0.82 +/- 0.03** (high confidence) vs **cMSCI = 0.81 +/- 0.22** (low confidence).

#### What Makes This Novel
- ProbVLM was designed for **classification and retrieval** uncertainty. We apply it to **coherence evaluation** — an entirely different task.
- The combination of probabilistic embeddings with geometric volume computation (**probabilistic Gramian coherence**) does not exist in any prior work.
- Confidence intervals on coherence scores enable downstream decisions (e.g., flagging unreliable evaluations, uncertainty-aware aggregation).

### 4.3 Layer 3: Contrastive Geometric Calibration

#### Problem
A GRAM volume of 0.15 is meaningless in isolation. Is 0.15 good? Bad? Average? The number depends on the domain, prompt complexity, and embedding model characteristics.

#### Our Extension

Inspired by negCLIPLoss (Li et al., NeurIPS 2024), which normalizes cosine similarity against contrastive pairs, we normalize the **geometric volume** against a negative bank.

**Step 1: Build domain-specific negative banks**

From our existing index (57 images, 104 audio files), grouped by domain:

```
nature_neg_images: [all nature images except the matched one]
nature_neg_audio:  [all nature audio except the matched one]
urban_neg_images:  [all urban images except the matched one]
urban_neg_audio:   [all urban audio except the matched one]
water_neg_images:  [all water images except the matched one]
water_neg_audio:   [all water audio except the matched one]
```

**Step 2: Compute contrastive margin**

```python
# Matched triple
matched_vol = gram_volume(text, matched_image, matched_audio)

# Top-k hardest negatives (same domain, highest individual similarities)
neg_volumes = []
for neg_img, neg_aud in top_k_negatives(text, domain, k=5):
    neg_volumes.append(gram_volume(text, neg_img, neg_aud))

# Contrastive calibrated score
cMSCI = (1 - matched_vol/vol_max) - mean(1 - neg_vol/vol_max for neg_vol in neg_volumes)
# Equivalently: cMSCI = mean(neg_volumes) - matched_vol  (lower volume = better)
```

Positive margin = genuinely coherent (matched triple is tighter than alternatives).
Near-zero margin = only generically similar (matched triple is no better than random same-domain alternatives).
Negative margin = likely incoherent (random alternatives are tighter).

#### What Makes This Novel
- negCLIPLoss applies contrastive normalization to **cosine similarity**. We apply it to **geometric volume measures** — a different mathematical object with different properties.
- Domain-specific negative banks for tri-modal geometric calibration do not exist in prior work.
- The margin-based formulation directly addresses the calibration problem that the modality gap literature identifies but does not solve for evaluation metrics.

---

## 5. Novelty Summary

### 5.1 What Is NOT Novel (we use these as-is)
- CLIP and CLAP pre-trained models (off-the-shelf)
- Cosine similarity as a sub-computation (standard)
- The retrieval and generative pipelines (engineering)
- Standard statistical tests (Wilcoxon, Cohen's d, etc.)
- Individual methods: C-MCR, GRAM, ProbVLM (published)

### 5.2 What IS Novel

| Contribution | Description | Why It's New |
|-------------|-------------|-------------|
| **Heterogeneous-Space GRAM** | Extending GRAM from single-space to cross-space (CLIP+CLAP) via projection + normalization | GRAM assumes homogeneous spaces; we solve the heterogeneous case |
| **Probabilistic Geometric Coherence** | Monte Carlo GRAM volume over ProbVLM uncertainty distributions | Combining probabilistic embeddings with geometric alignment measures is unprecedented |
| **Contrastive Geometric Calibration** | Margin-based normalization of GRAM volumes against domain-specific negative banks | Contrastive calibration has never been applied to geometric volume measures |
| **cMSCI Framework** | The unified pipeline: align -> normalize -> probabilistic geometry -> calibrate | This specific formulation for tri-modal coherence evaluation does not exist |
| **Tri-Modal Coherence Evaluation** | Systematic evaluation framework for text+image+audio coherence | Very few works address this; HiCAN (2025) is the only close competitor using attention networks, not geometry |

### 5.3 Differentiation from Closest Related Work

| Work | What They Do | How We Differ |
|------|-------------|--------------|
| **GRAM** (ICLR 2025) | Geometric alignment as training loss in shared space | We use it as evaluation metric in heterogeneous spaces |
| **C-MCR** (NeurIPS 2023) | Cross-space projection for retrieval | We use projection as preprocessing for geometric coherence, not retrieval |
| **ProbVLM** (ICCV 2023) | Uncertainty for classification/retrieval | We apply uncertainty to coherence scoring with geometric measures |
| **HiCAN** (Sci. Reports 2025) | Tri-modal alignment via attention networks | We use geometric methods (no training needed) with uncertainty + calibration |
| **negCLIPLoss** (NeurIPS 2024) | Contrastive normalization of cosine similarity | We extend contrastive calibration to geometric volume measures |
| **Symile** (NeurIPS 2024) | Joint multimodal learning via multilinear inner product | We address the evaluation problem (scoring), not the learning problem (training) |

---

## 6. Validation Plan

### 6.1 Re-run RQ1: Sensitivity Analysis

**Hypothesis:** cMSCI produces larger effect sizes than uncalibrated MSCI when distinguishing baseline from perturbed conditions.

**Method:**
- Same 30 prompts x 3 seeds x 3 conditions (baseline, wrong_image, wrong_audio)
- Compare Cohen's d for MSCI vs cMSCI
- Expect: d(cMSCI) > d(MSCI) > 2.2 (current)

### 6.2 Re-run RQ3: Human Correlation

**Hypothesis:** cMSCI correlates more strongly with human judgments than uncalibrated MSCI.

**Method:**
- Same 30 stratified samples rated by 3 raters
- Compare Spearman's rho: MSCI (current rho = 0.379) vs cMSCI (target rho > 0.50)
- Additionally report whether confidence intervals from Layer 2 predict when MSCI disagrees with humans (i.e., do uncertain scores have lower human correlation?)

### 6.3 Ablation Study

**Purpose:** Isolate the contribution of each layer.

| Variant | Layer 1 | Layer 2 | Layer 3 | Description |
|---------|---------|---------|---------|-------------|
| MSCI (baseline) | No | No | No | Current weighted average |
| MSCI-norm | Partial (z-score only, no GRAM) | No | No | Normalized cosine scores, still averaged |
| MSCI-geom | Yes (GRAM in aligned space) | No | No | Geometric alignment, no uncertainty |
| MSCI-prob | Yes | Yes (ProbVLM) | No | Probabilistic geometric, no calibration |
| **cMSCI** (full) | Yes | Yes | Yes | Full pipeline |

**Expected ordering:** cMSCI > MSCI-prob > MSCI-geom > MSCI-norm > MSCI (by human correlation)

### 6.4 Confidence Interval Analysis

**Purpose:** Validate that uncertainty estimates are meaningful.

- Bin evaluations by confidence (tight vs wide intervals)
- Show that tight-confidence scores have higher human agreement
- Show that flagging uncertain evaluations improves overall metric reliability

---

## 7. Implementation Plan

### Phase 1: Foundation (Space Alignment)
- Integrate C-MCR pre-trained projectors
- Implement z-score normalization with reference distributions from RQ1 data
- Validate: CKA score between projected audio and image embeddings should improve

### Phase 2: Geometric Scoring
- Implement GRAM volume computation (3x3 Gram matrix determinant)
- Replace weighted average with geometric coherence score
- Build negative banks per domain from existing indexes
- Implement contrastive calibration (margin over negatives)
- Validate: re-run sanity check, verify perturbation sensitivity

### Phase 3: Uncertainty
- Integrate ProbVLM adapters on CLIP and CLAP encoders
- Implement Monte Carlo GRAM volume sampling
- Validate: uncertainty correlates with human disagreement

### Phase 4: Experiments
- Re-run RQ1 with cMSCI
- Re-run RQ3 with cMSCI
- Run ablation study
- Generate updated figures

### Phase 5: Paper Update
- Update methodology section
- Update results
- Add ablation analysis
- Revise discussion with novelty contributions

---

## 8. Key References

1. **GRAM** — Cicchetti, G. et al. "Gramian Multimodal Representation Learning and Alignment." ICLR 2025. https://arxiv.org/abs/2412.11959

2. **C-MCR** — Wang, Z. et al. "Connecting Multi-Modal Contrastive Representations." NeurIPS 2023. https://arxiv.org/abs/2305.14381

3. **ProbVLM** — Upadhyay, U. et al. "ProbVLM: Probabilistic Adapter for Frozen Vision-Language Models." ICCV 2023. https://arxiv.org/abs/2307.00398

4. **Symile** — Patel, V. et al. "Symile: Multimodal Contrastive Learning Beyond Pairwise." NeurIPS 2024. https://arxiv.org/abs/2411.01053

5. **negCLIPLoss** — Li, H. et al. "CLIPLoss and Norm-Based Data Selection Methods for Multimodal Contrastive Learning." NeurIPS 2024. https://arxiv.org/abs/2405.19547

6. **Modality Gap** — Liang, W. et al. "Mind the Gap: Understanding the Modality Gap in Multi-Modal Contrastive Representation Learning." NeurIPS 2022. https://arxiv.org/abs/2203.02053

7. **TRIANGLE** — "TRI-modAl Neural Geometric LEarning." NeurIPS 2025. https://arxiv.org/abs/2509.24734

8. **Ex-MCR** — Wang, Z. et al. "Extending Multi-Modal Contrastive Representations." 2024. https://arxiv.org/abs/2310.08884

9. **HiCAN** — "Multimodal diffusion framework for collaborative text image audio generation." Scientific Reports 2025. https://www.nature.com/articles/s41598-025-05794-4

10. **Human-CLAP** — "Human-perception-based contrastive language-audio pretraining." 2025. https://arxiv.org/abs/2506.23553

---

## 9. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| C-MCR projection degrades audio embedding quality | Validate with CKA before/after; fall back to Ex-MCR if needed |
| ProbVLM adapter overfits on small dataset (57 img, 104 audio) | Use cross-validation; consider pre-trained ProbVLM weights |
| GRAM volume not sensitive enough to perturbations | Ablation will reveal this; can supplement with pairwise sub-scores |
| Negative bank too small for meaningful calibration | Use cross-domain negatives to expand bank; bootstrap confidence intervals |
| Computational overhead too high for MC sampling | Reduce N from 1000 to 100; profile and optimize |
| Human correlation doesn't improve over baseline | Report as informative result; analyze which layer helps/hurts |

---

## 10. Success Criteria

- [ ] cMSCI achieves higher Cohen's d than MSCI on RQ1 perturbation tests
- [ ] cMSCI achieves rho > 0.50 on RQ3 human correlation (currently 0.379)
- [ ] Ablation shows each layer contributes positively
- [ ] Confidence intervals predict human disagreement
- [ ] Full pipeline runs within 2x the inference time of current MSCI
