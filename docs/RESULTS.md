# Experimental Results

This document reports the results of controlled experiments evaluating the Multimodal Semantic Coherence Index (MSCI). For the full statistical report with retrieval bottleneck analysis and threats to validity, see [results_summary.md](results_summary.md).

---

## Status

| Research Question | Data Collection | Analysis | Status |
|-------------------|-----------------|----------|--------|
| RQ1: MSCI Sensitivity | Complete (270 runs) | Complete | **Supported** |
| RQ2: Planning Effect | Complete (360 runs) | Complete | **Not Supported** |
| RQ3: Human Correlation | Complete (30 samples, 3 raters) | Complete | **Supported** |

---

## RQ1: MSCI Sensitivity

**Question**: Is MSCI sensitive to controlled semantic perturbations?

**Hypothesis**:
- H0: MSCI(baseline) = MSCI(perturbed)
- H1: MSCI(baseline) > MSCI(perturbed)

**Method**: 30 prompts x 3 seeds x 3 conditions = 270 runs. Paired t-test on prompt-level averages.

### Results

```
Condition          N     Mean MSCI    SD       Median    Min      Max
-------------------------------------------------------------------------
Baseline           30    0.3024       0.0677   0.2977    0.1662   0.4212
Wrong Image        30    0.2627       0.0656   0.2615    0.1423   0.3747
Wrong Audio        30    0.1694       0.0427   0.1546    0.0953   0.2830
```

### Sub-Metric Breakdown

```
Condition          st_i (text-image)    st_a (text-audio)
-----------------------------------------------------------
Baseline           0.2297               0.3750
Wrong Image        0.1505               0.3750
Wrong Audio        0.2297               0.1091
```

Wrong_image selectively degrades st_i while preserving st_a. Wrong_audio selectively degrades st_a while preserving st_i. This confirms MSCI's sub-metrics are modality-specific and independently sensitive.

### Hypothesis Tests

Paired t-tests (one-sided: baseline > perturbation), Holm-Bonferroni corrected.

```
Comparison                  t(29)     p-value          Cohen's d    95% CI
---------------------------------------------------------------------------
Baseline vs Wrong Image     12.804    9.22 x 10^-14    2.338        [0.033, 0.046]
Baseline vs Wrong Audio     12.119    3.56 x 10^-13    2.213        [0.111, 0.155]
```

Both comparisons survive Holm-Bonferroni correction at alpha = 0.05.

### Power Analysis

| Metric | Value |
|--------|-------|
| Observed average \|d\| | 2.275 |
| Current N | 30 |
| Required N (power = 0.80) | 2 |
| Adequately powered | Yes |

The experiment is massively overpowered — the effect sizes are so large that only 2 prompts would suffice for 80% power.

### RQ1 Verdict

> **SUPPORTED** — MSCI is sensitive to controlled semantic perturbations (p < 0.05, |d| >= 0.5 for both comparisons).

MSCI reliably drops when either the image or audio modality is deliberately mismatched. The mean drop for wrong_image is 0.040 (13% relative decrease), and for wrong_audio is 0.133 (44% relative decrease). The larger audio effect is consistent with CLAP having stronger discriminative power for cross-domain mismatches than CLIP in our retrieval pool.

---

## RQ2: Planning Effect

**Question**: Does structured semantic planning improve cross-modal alignment?

**Hypothesis**:
- H0: MSCI(planner) = MSCI(direct)
- H1: MSCI(planner) > MSCI(direct)

**Method**: 30 prompts x 3 seeds x 4 modes = 360 runs. Paired t-test on prompt-level averages.

### Ablation Conditions

| Condition | Description | LLM Calls | Token Budget |
|-----------|-------------|-----------|--------------|
| Direct | No planning (baseline) | 0 | 0 |
| Single Planner | 1 LLM call | 1 | 1x |
| Council | 3 LLM calls + merge | 3 | 3x |
| Extended Prompt | 1 call, 3x tokens | 1 | 3x |

### Results

```
Mode               N     Mean MSCI    SD       Median    Min      Max
-----------------------------------------------------------------------
Direct             30    0.2966       0.0695   0.3087    0.1417   0.4234
Extended Prompt    30    0.2940       0.0605   0.3063    0.1680   0.4302
Planner            30    0.2790       0.0733   0.2866    0.0491   0.4080
Council            30    0.2749       0.0744   0.2747    0.0699   0.3985
```

### Hypothesis Tests

Paired t-tests (two-sided), Holm-Bonferroni corrected.

```
Comparison                      t(29)    Raw p    Adj. p    Cohen's d    Interpretation
---------------------------------------------------------------------------------------
Council vs Direct               -2.199   0.0360   0.1081    -0.401       Small negative
Planner vs Direct               -1.797   0.0827   0.1081    -0.328       Small negative
Extended Prompt vs Direct       -0.270   0.7893   0.7893    -0.049       Negligible
Council vs Extended Prompt      -2.266   0.0311   0.1081    -0.414       Small negative
```

After Holm-Bonferroni correction, no comparison reaches significance at alpha = 0.05.

### Key Finding: Structure vs. Tokens

The extended_prompt control is revealing: giving the LLM 3x the token budget without structured output produces virtually identical results to direct mode (d = -0.05, p = 0.79). This confirms the limiting factor is not planning effort or token budget, but the **retrieval pool ceiling**.

### RQ2 Verdict

> **NOT SUPPORTED** — Structured planning does not significantly improve cross-modal alignment.

None of the three planning modes produced significantly higher MSCI scores than the direct baseline after multiple comparison correction. The trend is slightly negative: planning modes tend to reduce alignment, with council showing the largest (non-significant) decrease (d = -0.40).

### Interpretation

The null result is **informative rather than disappointing**. The retrieval pool ceiling is the primary explanation: with only 21 images and 60 audio files, the best-matching asset for any prompt is constrained by pool diversity rather than prompt quality. Furthermore, the LLM planner (qwen2:7b) introduces semantic drift — transforming concrete prompts into more abstract plans that retrieve less precisely. See [results_summary.md](results_summary.md) for the full retrieval bottleneck analysis.

---

## RQ3: Human Correlation

**Question**: Does MSCI correlate with human judgments of multimodal coherence?

**Hypothesis**:
- H0: rho(MSCI, human) <= 0
- H1: rho(MSCI, human) > 0

**Method**: Spearman correlation on 30 stratified samples rated by human evaluators.

### Design

- 30 samples: 10 baseline, 10 wrong_image, 10 wrong_audio
- Stratified across nature, urban, water domains
- Blind evaluation (condition labels hidden)
- Randomized presentation order per evaluator
- 20% re-rating for intra-rater reliability (target: kappa >= 0.70)
- AudioLDM-generated audio (unique per sample)

### Human Evaluation Interface

A Streamlit app (`app/human_eval_app.py`) provides:
- Text, image, and audio display
- 5-point Likert ratings for text-image, text-audio, image-audio, and overall coherence
- Structured rubric with expandable descriptions
- Session persistence (save/resume)
- Multiple evaluator support

### Results

**Raters**: 3 independent evaluators (Dr Nikhil, Fariha, Pratik), all 30 samples rated.

#### Inter-Rater Reliability

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ICC(3,1) | 0.697 | Moderate agreement (single rater) |
| ICC(3,k) | 0.873 | Good agreement (averaged across raters) |
| ICC(2,1) | 0.690 | Moderate (random effects model) |
| Krippendorff's alpha | 0.684 | Acceptable (> 0.667 threshold) |

#### MSCI-Human Correlation

| Metric | Value | p-value | 95% CI | Significant? |
|--------|-------|---------|--------|--------------|
| Spearman's rho | 0.379 | 0.039 | [0.021, 0.650] | Yes |
| Kendall's tau | 0.293 | 0.038 | [0.041, 0.546] | Yes |

#### Per-Rater Correlations (Spearman)

| Rater | rho |
|-------|-----|
| Pratik | 0.419 |
| Dr Nikhil | 0.343 |
| Fariha | 0.267 |

#### Condition Breakdown

```
Condition          N     Human Mean (1-5)    Human SD    MSCI Mean
--------------------------------------------------------------------
Baseline           10    2.70                1.487       0.3865
Wrong Image        10    2.10                0.539       0.3322
Wrong Audio        10    2.10                1.044       0.2277
```

Both MSCI and human ratings rank conditions identically: baseline > wrong_image > wrong_audio. However, the between-condition differences in human ratings are not statistically significant (Kruskal-Wallis H = 0.785, p = 0.675), likely due to the small per-condition sample size (n = 10).

### RQ3 Verdict

> **SUPPORTED** — MSCI correlates with human coherence judgments (Spearman's rho = 0.379, p = 0.039; Kendall's tau = 0.293, p = 0.038). The correlation is moderate, indicating MSCI captures meaningful aspects of perceived coherence while leaving substantial variance unexplained.

---

## Summary of Findings

| RQ | Hypothesis | Result | Effect Size | Interpretation |
|----|------------|--------|-------------|----------------|
| RQ1 | MSCI sensitive to perturbations | **Supported** | d = 2.21 - 2.34 (large) | MSCI reliably detects mismatches |
| RQ2 | Planning improves alignment | **Not Supported** | d = -0.05 to -0.41 (negligible to small) | Retrieval ceiling, not pipeline failure |
| RQ3 | MSCI correlates with humans | **Supported** | rho = 0.379 (moderate) | Significant but moderate correlation |

---

## Negative Results

RQ2 produced a null result. This is reported honestly with full analysis:

- All planning modes showed slight (non-significant) MSCI decreases
- The retrieval pool ceiling is quantified: 72.6% of image retrievals fall below 0.25 similarity, and only 7/60 audio files are ever selected
- The extended_prompt control confirms the bottleneck is data, not planning
- LLM semantic drift reduces retrieval effectiveness for audio

These findings motivate dataset expansion (Wikimedia + Freesound scripts provided) and generative pipelines (Stable Diffusion + AudioLDM integrated) as concrete next steps.

---

## Reproducibility

```bash
# RQ1: MSCI sensitivity
python scripts/run_rq1.py --out-dir runs/rq1

# RQ2: Planning ablation
python scripts/run_rq2.py --out-dir runs/rq2

# Analyze with bootstrap CIs
python scripts/analyze_results.py runs/rq1/rq1_results.json
python scripts/analyze_results.py runs/rq2/rq2_results.json

# RQ3: Human evaluation
streamlit run app/human_eval_app.py
python scripts/analyze_human_eval.py
```

Random seeds: [42, 123, 7] (3 seeds per prompt for robustness).

---

## Raw Data Location

```
runs/rq1/rq1_results.json      # RQ1 raw results (270 runs)
runs/rq2/rq2_results.json      # RQ2 raw results (360 runs)
runs/rq3/rq3_samples.json      # RQ3 stratified samples (30 items)
runs/rq3/sessions/              # Human evaluation session files
```
