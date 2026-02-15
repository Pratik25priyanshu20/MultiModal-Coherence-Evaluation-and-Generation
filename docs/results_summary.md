# Experiment Results Summary

## Overview

This document reports the results of two controlled experiments evaluating the
Multimodal Semantic Coherence Index (MSCI), a metric for measuring cross-modal
alignment in multimodal generation systems.

- **RQ1**: Is MSCI sensitive to controlled semantic perturbations?
- **RQ2**: Does structured planning improve cross-modal alignment?

All experiments used 30 diverse prompts across 4 semantic domains (nature, urban,
water, mixed), each tested with 3 random seeds (42, 123, 7) for robustness.

---

## RQ1: MSCI Sensitivity to Perturbations

**Design**: 30 prompts x 3 seeds x 3 conditions = 270 runs

Each prompt tested under:
- **baseline**: semantically matched image + audio
- **wrong_image**: deliberately mismatched image (different domain)
- **wrong_audio**: deliberately mismatched audio (different domain)

### Descriptive Statistics (MSCI)

| Condition    |  N  |  Mean  |  Std   | Median |  Min   |  Max   |
|:-------------|:---:|:------:|:------:|:------:|:------:|:------:|
| baseline     | 30  | 0.3024 | 0.0677 | 0.2977 | 0.1662 | 0.4212 |
| wrong_image  | 30  | 0.2627 | 0.0656 | 0.2615 | 0.1423 | 0.3747 |
| wrong_audio  | 30  | 0.1694 | 0.0427 | 0.1546 | 0.0953 | 0.2830 |

### Sub-Metric Breakdown

| Condition    | st_i (text-image) | st_a (text-audio) |
|:-------------|:-----------------:|:-----------------:|
| baseline     |      0.2297       |      0.3750       |
| wrong_image  |      0.1505       |      0.3750       |
| wrong_audio  |      0.2297       |      0.1091       |

**Observation**: wrong_image selectively degrades st_i while preserving st_a.
wrong_audio selectively degrades st_a while preserving st_i. This confirms
MSCI's sub-metrics are modality-specific and independently sensitive.

### Hypothesis Tests

Paired t-tests (one-sided: baseline > perturbation), averaged across seeds per prompt.

| Comparison               | t(29)  | p-value       | Cohen's d | 95% CI            | Interpretation        |
|:-------------------------|:------:|:-------------:|:---------:|:-----------------:|:---------------------:|
| baseline vs wrong_image  | 12.804 | 9.22 x 10^-14 |   2.338   | [0.0333, 0.0459]  | Large positive effect |
| baseline vs wrong_audio  | 12.119 | 3.56 x 10^-13 |   2.213   | [0.1105, 0.1554]  | Large positive effect |

Both comparisons survive **Holm-Bonferroni correction** at alpha = 0.05.

### Power Analysis

| Metric                            | Value  |
|:----------------------------------|:------:|
| Observed average \|d\|            | 2.275  |
| Current N                         | 30     |
| Required N (power = 0.80)         | 2      |
| Adequately powered                | Yes    |

The experiment is massively overpowered. The effect sizes are so large that
only 2 prompts would suffice for 80% power.

### RQ1 Verdict

> **SUPPORTED** -- MSCI is sensitive to controlled semantic perturbations
> (p < 0.05, |d| >= 0.5 for both comparisons).

MSCI reliably drops when either the image or audio modality is deliberately
mismatched. The mean drop for wrong_image is +0.040 (13% relative decrease),
and for wrong_audio is +0.133 (44% relative decrease). The larger audio effect
is consistent with the CLAP embedding space having stronger discriminative power
for cross-domain mismatches than CLIP in our retrieval pool.

---

## RQ2: Effect of Structured Planning on Alignment

**Design**: 30 prompts x 3 seeds x 4 modes = 360 runs

Each prompt tested under:
- **direct**: raw prompt passed directly to generators (no planning)
- **planner**: 1 LLM call producing a unified semantic plan
- **council**: 3 independent LLM calls merged via council voting
- **extended_prompt**: 1 LLM call with 3x token budget (controls for token count)

### Descriptive Statistics (MSCI)

| Mode             |  N  |  Mean  |  Std   | Median |  Min   |  Max   |
|:-----------------|:---:|:------:|:------:|:------:|:------:|:------:|
| direct           | 30  | 0.2966 | 0.0695 | 0.3087 | 0.1417 | 0.4234 |
| extended_prompt  | 30  | 0.2940 | 0.0605 | 0.3063 | 0.1680 | 0.4302 |
| planner          | 30  | 0.2790 | 0.0733 | 0.2866 | 0.0491 | 0.4080 |
| council          | 30  | 0.2749 | 0.0744 | 0.2747 | 0.0699 | 0.3985 |

### Sub-Metric Breakdown

| Mode             | st_i (text-image) | st_a (text-audio) |
|:-----------------|:-----------------:|:-----------------:|
| direct           |      0.2129       |      0.3803       |
| extended_prompt  |      0.2195       |      0.3685       |
| planner          |      0.2103       |      0.3476       |
| council          |      0.2111       |      0.3386       |

**Observation**: st_i is nearly identical across all modes (~0.21). The variance
comes from st_a, where planning modes slightly reduce text-audio alignment.
This suggests the LLM planner produces audio descriptions that are less
retrieval-friendly than the raw prompt.

### Hypothesis Tests

Paired t-tests (two-sided), averaged across seeds per prompt.

| Comparison                   | t(29)  | Raw p   | Adj. p  | Cohen's d | Interpretation          |
|:-----------------------------|:------:|:-------:|:-------:|:---------:|:-----------------------:|
| council vs direct            | -2.199 | 0.0360  | 0.1081  |  -0.401   | Small negative effect   |
| planner vs direct            | -1.797 | 0.0827  | 0.1081  |  -0.328   | Small negative effect   |
| extended_prompt vs direct    | -0.270 | 0.7893  | 0.7893  |  -0.049   | Negligible              |
| council vs extended_prompt   | -2.266 | 0.0311  | 0.1081  |  -0.414   | Small negative effect   |

After **Holm-Bonferroni correction**, no comparison reaches significance at alpha = 0.05.

### RQ2 Verdict

> **NOT SUPPORTED** -- Structured planning does not significantly improve
> cross-modal alignment.

None of the three planning modes (planner, council, extended_prompt) produced
significantly higher MSCI scores than the direct baseline after multiple
comparison correction. In fact, the trend is slightly negative: planning modes
tend to reduce alignment, with the council mode showing the largest (though
non-significant) decrease (d = -0.40).

### Interpretation of the Negative Result

The null result for RQ2 is informative rather than disappointing. We identify
the **retrieval pool ceiling** as the primary explanation: with only 21 images
and 60 audio files in the retrieval index, the best-matching asset for any
prompt is constrained by pool diversity rather than prompt quality. A more
sophisticated prompt (from the planner) cannot improve retrieval if the pool
does not contain a better match.

Furthermore, the LLM planner (qwen2:7b) may introduce semantic drift --
transforming a concrete prompt like "ocean waves crashing on rocks" into a more
abstract plan that retrieves less precisely. The st_a degradation in planning
modes supports this hypothesis: planner-generated audio descriptions may be
more stylistically rich but less retrieval-effective in CLAP's embedding space.

The **extended_prompt control** is revealing: giving the LLM 3x the token budget
without structured output produces virtually identical results to direct mode
(d = -0.05, p = 0.79). This confirms that the limiting factor is not planning
effort or token budget, but retrieval ceiling.

---

## Summary of Findings

| Research Question | Verdict        | Key Evidence                                           |
|:------------------|:---------------|:-------------------------------------------------------|
| RQ1: Sensitivity  | **Supported**  | p < 10^-13, Cohen's d > 2.2 for both perturbations    |
| RQ2: Planning     | **Not Supported** | No mode significantly improves over direct (adj. p > 0.05) |

### Effect Size Classification (Cohen, 1988)

| Effect Size |  |d| Range  | Observed in RQ1 | Observed in RQ2 |
|:------------|:-----------:|:---------------:|:---------------:|
| Small       | 0.2 - 0.5  |        --       | d = 0.33 - 0.41 |
| Medium      | 0.5 - 0.8  |        --       |        --        |
| Large       |    > 0.8   | d = 2.21 - 2.34 |        --        |

---

## Experimental Parameters

| Parameter              | RQ1               | RQ2                                    |
|:-----------------------|:------------------|:---------------------------------------|
| Prompts                | 30                | 30                                     |
| Seeds                  | [42, 123, 7]      | [42, 123, 7]                           |
| Conditions/Modes       | 3 conditions      | 4 modes                                |
| Total runs             | 270               | 360                                    |
| Runtime                | 85 min            | 226 min                                |
| Text generation        | Skip (prompt=text)| Ollama qwen2:7b                        |
| Image retrieval        | CLIP (0.20 threshold) | CLIP (0.20 threshold)              |
| Audio retrieval        | CLAP (0.10 threshold) | CLAP (0.10 threshold)              |
| Statistical test       | Paired t-test     | Paired t-test                          |
| Multiple comparison    | Holm-Bonferroni   | Holm-Bonferroni                        |
| Significance level     | alpha = 0.05      | alpha = 0.05                           |

---

## Retrieval Bottleneck Analysis

The RQ2 null result is best explained by a quantifiable retrieval ceiling. We
analyzed similarity distributions across all 630 retrieval events (270 RQ1 +
360 RQ2).

### Pool Coverage

| Metric                       | Image           | Audio           |
|:-----------------------------|:---------------:|:---------------:|
| Index size                   | 21 files        | 60 files        |
| Unique files ever selected   | 15 / 21 (71%)  | 7 / 60 (12%)   |
| Shannon entropy              | 3.21 / 3.91    | 2.21 / 2.81    |
| Most-selected file           | forest_mountain01.png (29%) | forest_birds_wind_01.wav (41%) |

Audio retrieval is severely concentrated: 7 files out of 60 cover all 630
retrievals, and a single file accounts for 41% of selections.

### Similarity Distributions

| Metric                         | Image     | Audio     |
|:-------------------------------|:---------:|:---------:|
| Mean cosine similarity         | 0.2288    | 0.3777    |
| Median                         | 0.2215    | 0.3715    |
| % below 0.25 threshold        | 72.6%     | 17.0%     |
| Top-5 gap (mean)              | 0.0605    | 0.2505    |

72.6% of image retrievals fall below 0.25 cosine similarity — a regime where
CLIP text-image alignment is essentially at floor level. The top-5 gap for
images (0.06) indicates a flat similarity landscape: the best match is barely
better than the 5th-best match, leaving no room for planning to improve
selection.

### Per-Domain Breakdown

| Domain  | RQ1 Image | RQ2 Image | RQ1 Audio | RQ2 Audio |
|:--------|:---------:|:---------:|:---------:|:---------:|
| nature  |   0.223   |   0.230   |   0.409   |   0.371   |
| urban   |   0.255   |   0.252   |   0.295   |   0.259   |
| water   |   0.234   |   0.226   |   0.512   |   0.498   |
| mixed   |   0.204   |   0.202   |   0.370   |   0.366   |

Water/nature domains achieve the highest audio similarity (>0.37), consistent
with the audio pool being biased toward nature sounds. Urban and mixed domains
are systematically disadvantaged, further explaining within-domain variance in
MSCI.

### Mode Comparison (Direct vs Planner)

| Mode    | Image mean | Audio mean |
|:--------|:----------:|:----------:|
| direct  | 0.2297     | 0.3936     |
| planner | 0.2270     | 0.3460     |

The planner produces **lower** audio similarity than direct (-0.048), confirming
that LLM-generated audio descriptions are less retrieval-friendly in CLAP's
embedding space. Image similarity is virtually identical across modes, consistent
with the ceiling hypothesis.

---

## Threats to Validity

### Internal Validity

1. **Retrieval pool ceiling (primary threat)**. With only 21 images and 60 audio
   files, the best-matching asset for any prompt is constrained by pool diversity
   rather than pipeline quality. This is the dominant confound for RQ2: planning
   cannot improve retrieval if no better match exists. We quantify this threat
   above — 72.6% of image retrievals fall below the 0.25 similarity threshold,
   and only 7 unique audio files are ever selected.

2. **Embedding model dependency**. MSCI is defined by CLIP (ViT-B/32) and CLAP
   (HTSAT-BERT) embeddings. Results may not generalize to other embedding models.
   Both models are frozen (no fine-tuning), and our projection heads are identity
   when in_dim == out_dim. This preserves pre-trained alignment but limits
   adaptation to our domain.

3. **No cross-modal joint training**. CLIP (text-image) and CLAP (text-audio)
   occupy separate embedding spaces trained on different data. Image-audio
   similarity (si_a) is omitted from MSCI for this reason, but the text anchoring
   approach (comparing text-image and text-audio independently) may still conflate
   lexical overlap with semantic alignment.

4. **Perturbation design (RQ1)**. Wrong-image and wrong-audio perturbations swap
   assets across domains (e.g., nature → urban). Cross-domain swaps produce large
   effects (d > 2.2) but do not test within-domain sensitivity (e.g., two
   different forest images). The effect sizes may overestimate sensitivity to
   subtle misalignments.

### External Validity

5. **Retrieval-only pipeline**. Images and audio are retrieved from fixed pools,
   not generated by diffusion models (Stable Diffusion, DALL-E) or neural audio
   synthesis (AudioLDM). Results may not transfer to generative pipelines where
   the output space is unbounded.

6. **Single planner model**. RQ2 uses qwen2:7b (7B parameter local model) for
   all planning modes. A larger or better-prompted model might produce more
   retrieval-effective descriptions. The negative planning result is specific to
   this model + retrieval pool combination.

7. **Prompt domain coverage**. All 30 prompts belong to 4 domains (nature, urban,
   water, mixed). Results may not generalize to abstract concepts, emotional
   content, or action scenes that require temporal understanding.

8. **MSCI weight assumptions**. The 0.45/0.45/0.10 weights for st_i/st_a/si_a
   are hypothesized, not empirically optimized. RQ3 human evaluation will
   partially address this by testing whether the weighted composite correlates
   with human perception.

### Statistical Conclusion Validity

9. **Multiple testing**. Both RQ1 and RQ2 apply Holm-Bonferroni correction at
   alpha = 0.05. RQ1 survives correction easily (p < 10^-13). RQ2 raw p-values
   (0.031–0.036 for council/planner vs direct) do not survive correction,
   appropriately preventing false positive claims.

10. **Seed variance**. Three seeds per prompt (42, 123, 7) provide limited
    sampling of stochastic variation. We average across seeds before hypothesis
    testing (paired by prompt), which is conservative but reduces within-prompt
    variance. Adding more seeds would improve precision of prompt-level
    estimates.
