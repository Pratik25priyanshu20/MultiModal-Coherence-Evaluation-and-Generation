# Controlled Experiment Guide

## Overview

This guide explains the controlled experimental design implemented to address the critical question: **"Is this structure or just more prompting?"**

## Experimental Design

### 6 Experimental Conditions

| Prompt Mode | Condition | Description |
|-------------|----------|-------------|
| Direct | Baseline | Raw prompt → generators (no planner) |
| Direct | Wrong Image | Direct mode + mismatched image |
| Direct | Wrong Audio | Direct mode + mismatched audio |
| Planner | Baseline | Planner → generators |
| Planner | Wrong Image | Planner mode + mismatched image |
| Planner | Wrong Audio | Planner mode + mismatched audio |

### Why This Design?

1. **Controlled Baseline**: Direct mode provides a fair comparison without planner structure
2. **Perturbation Tests**: Wrong image/audio test metric sensitivity
3. **Paired Comparison**: Same prompt, same seed, only difference is planner presence
4. **Reproducibility**: Same models, same generators, same datasets

## Usage

### Step 1: Run Controlled Experiment

```bash
# Single prompt
python3 scripts/run_controlled_experiment.py --prompt "A calm foggy forest at dawn"

# Multiple prompts from file
python3 scripts/run_controlled_experiment.py --prompts-file prompts.json

# Custom output directory
python3 scripts/run_controlled_experiment.py --prompt "..." --out-dir runs/my_experiment
```

### Step 2: Analyze Results

```bash
python3 scripts/analyze_controlled_experiment.py runs/controlled_experiment/controlled_experiment_results.json
```

This will:
- Compute statistics for each condition
- Calculate separation between baseline and perturbed
- Compare planner vs direct mode
- Save analysis to JSON

## What Gets Measured

### Metrics (Reused, Not New)

- **MSCI**: Multimodal Semantic Coherence Index
- **st_i**: Text-Image similarity
- **st_a**: Text-Audio similarity
- **si_a**: Image-Audio similarity

### Analysis Focus

1. **Mean & Variance Shifts**: Do conditions show different distributions?
2. **Separation**: Can metrics distinguish baseline from perturbed?
3. **Planner vs Direct**: Paired comparison showing planner effect

## Important Limitations

### What We DO NOT Claim

- ❌ Alignment with "true meaning"
- ❌ Human-level semantic correctness
- ❌ Generative quality improvements
- ❌ Absolute correctness of MSCI

### What We DO Claim

- ✅ This evaluates model-to-model semantic consistency
- ✅ MSCI measures relative cross-modal alignment
- ✅ Results are comparative, not absolute
- ✅ Planner vs direct comparison shows structural effects

## Interpretation Guidelines

### Reading Results

1. **Separation > 0**: Metric can distinguish baseline from perturbed (good)
2. **Planner vs Direct difference**: Shows structural effect (if consistent)
3. **Mean shifts**: Indicates different alignment patterns

### What Results Mean

- **Higher separation in planner mode**: Structure helps metric sensitivity
- **Different means**: Planner changes alignment patterns
- **Consistent differences**: Structural effect is measurable

### What Results Do NOT Mean

- Higher MSCI ≠ "better" semantic understanding
- Separation ≠ human-verifiable correctness
- Differences ≠ absolute quality improvements

## Code Changes

### Pipeline (`src/pipeline/generate_and_evaluate.py`)

**Added:**
- `mode` parameter: "direct" | "planner"
- `condition` parameter: "baseline" | "wrong_image" | "wrong_audio"
- Direct mode path (bypasses planner)
- Condition-based perturbations
- Enhanced metadata logging

**Unchanged:**
- Embedders
- Generators
- MSCI formulation
- Council-Lite internals

### New Scripts

1. **`scripts/run_controlled_experiment.py`**
   - Runs all 6 conditions
   - Ensures reproducibility (same seed, models)
   - Saves structured results

2. **`scripts/analyze_controlled_experiment.py`**
   - Comparative analysis only
   - No new metrics
   - Focus on separation and differences

## Example Workflow

```bash
# 1. Run experiment
python3 scripts/run_controlled_experiment.py \
    --prompt "A rainy neon-lit city street at night" \
    --out-dir runs/controlled_experiment \
    --seed 42

# 2. Analyze results
python3 scripts/analyze_controlled_experiment.py \
    runs/controlled_experiment/controlled_experiment_results.json \
    --output runs/controlled_experiment/analysis.json

# 3. Review analysis
cat runs/controlled_experiment/analysis.json | jq '.planner_vs_direct.msci'
```

## Output Structure

### Results File (`controlled_experiment_results.json`)

```json
{
  "direct_baseline": [
    {
      "run_id": "...",
      "mode": "direct",
      "condition": "baseline",
      "scores": {"msci": 0.5, "st_i": 0.6, ...},
      "meta": {"seed": 42, ...}
    }
  ],
  "planner_baseline": [...],
  ...
}
```

### Analysis File (`controlled_experiment_analysis.json`)

```json
{
  "metrics": {
    "msci": {
      "direct_baseline": {"mean": 0.5, "std": 0.1, ...},
      ...
    }
  },
  "comparisons": {
    "msci": {
      "direct_baseline_vs_wrong_image": {
        "separation": 0.8,
        "overlap": 0.2
      }
    }
  },
  "planner_vs_direct": {
    "msci": {
      "baseline": {"separation": 0.3, ...}
    }
  }
}
```

## Academic Standards

This design follows academic best practices:

1. **Controlled Variables**: Only planner presence varies
2. **Reproducibility**: Same seed, models, generators
3. **Explicit Limitations**: Clear about what we don't claim
4. **Comparative Analysis**: Focus on differences, not absolutes
5. **No Methodological Overreach**: Reuses existing metrics

## Next Steps

1. ✅ Run controlled experiments
2. ✅ Analyze results
3. 📊 Document findings
4. 📝 Prepare for academic review

---

**Status**: Implementation complete. Ready for controlled experiments.
