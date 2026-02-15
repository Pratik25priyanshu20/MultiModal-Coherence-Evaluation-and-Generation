# Implementation Plan - Multimodal Coherence AI

## Executive Summary

I've created a comprehensive execution plan and implemented **Phases 1-4** of your roadmap. Here's what's been built:

## ✅ Completed Phases

### Phase 1: Dataset-Driven Evaluation Loop
**Status**: ✅ Complete

**What was built**:
- `scripts/run_laion_eval.py` - Main evaluation script that:
  - Loads LAION captions (hard cap 500, deterministic shuffle)
  - For each caption: semantic plan → generate → evaluate → store
  - Produces `raw_results.json` and `summary_msci.json`

- Enhanced `src/data/laion_loader.py` with:
  - Deterministic seeding support
  - Shuffle option for reproducible subsets

**Usage**:
```bash
python scripts/run_laion_eval.py
```

**Output**: `runs/laion_eval/raw_results.json` + `summary_msci.json`

---

### Phase 2: Automatic Failure Diagnostics
**Status**: ✅ Complete

**What was built**:
- `scripts/analyze_laion_failures.py` - Failure analysis that:
  - Analyzes all results for failure patterns
  - Categorizes failures (AUDIO_ALIGNMENT_FAILURE, PLANNER_UNDERSPECIFIED, etc.)
  - Saves top-10 worst examples with metadata
  - Produces `diagnostics.json` with failure mode counts

**Failure Modes Detected**:
- `AUDIO_ALIGNMENT_FAILURE` - Audio not matching text/image
- `PLANNER_AUDIO_UNDERSPECIFIED` - Plan missing audio elements
- `PLANNER_VISUAL_UNDERSPECIFIED` - Plan missing visual elements
- `TEXT_IMAGE_MISMATCH` - Text-image similarity very low
- `AUDIO_GENERATION_DRIFT` - Audio drifts from plan
- `IMAGE_GENERATION_DRIFT` - Image drifts from plan
- `GLOBAL_COHERENCE_FAILURE` - Negative MSCI

**Usage**:
```bash
python scripts/analyze_laion_failures.py
```

**Output**: `runs/laion_eval/diagnostics.json` + `worst_examples/` directory

---

### Phase 3: Generator Conditioning Consistency (Option B)
**Status**: ✅ Complete

**What was built**:
- `src/pipeline/conditioning_validator.py` - Validation module that:
  - Detects prompt leakage (keywords from original prompt not in plan)
  - Checks plan completeness
  - Validates that all modality prompts derive from plan only

- `scripts/verify_conditioning.py` - Verification script to audit batches

- Integrated validation into `generate_and_evaluate()`:
  - Automatically validates conditioning strictness
  - Stores results in `bundle.meta["conditioning_validation"]`
  - Logs warnings for violations (non-fatal)

**Usage**:
```bash
python scripts/verify_conditioning.py
```

**Output**: `runs/laion_eval/conditioning_analysis.json`

---

### Phase 4: Metric Calibration (Option D)
**Status**: ✅ Complete (needs integration)

**What was built**:
- `scripts/calibrate_metrics.py` - Calibration script that:
  - Runs batch perturbation experiments (baseline vs wrong modalities)
  - Computes normalization parameters (mean, std) from baseline distributions
  - Analyzes separation between baseline and wrong-modality distributions
  - Derives thresholds based on separation analysis
  - Produces `calibration_config.json`

- `src/coherence/normalized_scorer.py` - Normalized scoring module:
  - Z-score normalization: `(value - mean) / std`
  - Calibrated threshold classification
  - Can be integrated into coherence engine

**Usage**:
```bash
python scripts/calibrate_metrics.py
```

**Output**: `runs/calibration/calibration_config.json`

**Next Step**: Integrate `NormalizedScorer` into `CoherenceEngine` (see below)

---

## 📋 Remaining Work

### Phase 4 Integration (High Priority)
**Status**: ⏳ Pending

**What needs to be done**:
1. Integrate `NormalizedScorer` into `CoherenceEngine`
2. Update `CoherenceScorer` to use calibrated thresholds
3. Re-run evaluation with normalized metrics
4. Verify MSCI distribution improves

**Files to modify**:
- `src/coherence/coherence_engine.py` - Add normalization option
- `src/coherence/scorer.py` - Use calibrated thresholds
- `src/coherence/thresholds.py` - Load from calibration config

---

### Phase 5: Caching & Reproducibility
**Status**: ⏳ Pending

**What needs to be built**:
1. Embedding cache (avoid recomputing embeddings for same inputs)
2. Generated artifact cache (reuse images/audio when possible)
3. Deterministic LAION subset caching
4. Seed management for full reproducibility

---

## 🚀 Quick Start

### 1. Run Full Evaluation Pipeline

```bash
# Step 1: Evaluate LAION dataset
python scripts/run_laion_eval.py

# Step 2: Analyze failures
python scripts/analyze_laion_failures.py

# Step 3: Verify conditioning
python scripts/verify_conditioning.py
```

### 2. Calibrate Metrics

```bash
# Run calibration experiments (takes time - runs multiple perturbation experiments)
python scripts/calibrate_metrics.py

# Review calibration config
cat runs/calibration/calibration_config.json
```

### 3. View Results

```bash
# Summary statistics
cat runs/laion_eval/summary_msci.json

# Failure diagnostics
cat runs/laion_eval/diagnostics.json

# Worst examples
ls runs/laion_eval/worst_examples/
```

---

## 📊 Expected Outcomes

### After Phase 1-2:
- ✅ 500 LAION captions evaluated
- ✅ Failure modes automatically categorized
- ✅ Top-10 worst examples identified

### After Phase 3:
- ✅ Conditioning validation working
- ✅ Prompt leakage detected
- ✅ Plan completeness checked

### After Phase 4 (when integrated):
- ⏳ Normalized similarity scores
- ⏳ Calibrated thresholds from perturbation data
- ⏳ Improved MSCI distribution (should shift upward)

---

## 🔍 Key Insights from Implementation

### Current Architecture Strengths:
1. **Semantic Planning Layer**: Already producing structured plans
2. **Modality Prompts**: Already derived from plan (not raw prompt)
3. **Evaluation Pipeline**: End-to-end pipeline working

### Identified Issues:
1. **Audio Alignment**: `st_a` and `si_a` often low/negative
   - Likely cause: Audio generation not matching semantic plan well
   - Solution: Improve audio generator conditioning (future work)

2. **MSCI Negative Means**: Indicates embedding-space misalignment
   - Likely cause: CLIP/CLAP embedding spaces not aligned
   - Solution: Normalization should help (Phase 4)

3. **Plan Completeness**: Some plans missing audio/visual elements
   - Detected by: `analyze_laion_failures.py`
   - Solution: Improve planner prompts (future work)

---

## 📁 File Structure

```
scripts/
├── run_laion_eval.py              # Phase 1: Dataset evaluation
├── analyze_laion_failures.py      # Phase 2: Failure diagnostics
├── verify_conditioning.py         # Phase 3: Conditioning validation
└── calibrate_metrics.py           # Phase 4: Metric calibration

src/
├── pipeline/
│   ├── generate_and_evaluate.py  # Main pipeline (updated with validation)
│   └── conditioning_validator.py  # Phase 3: Validation module
├── coherence/
│   └── normalized_scorer.py       # Phase 4: Normalized scoring
└── data/
    └── laion_loader.py             # Phase 1: Enhanced loader

docs/
└── execution_plan.md              # Detailed execution plan
```

---

## 🎯 Next Steps (Priority Order)

1. **Integrate Phase 4** (Normalized Scorer into Coherence Engine)
   - Modify `coherence_engine.py` to use `NormalizedScorer`
   - Update thresholds to use calibration config
   - Re-run evaluation and verify improvement

2. **Run Full Evaluation**
   - Execute `run_laion_eval.py` on 500 LAION captions
   - Run `analyze_laion_failures.py` to identify patterns
   - Review worst examples to understand failure modes

3. **Calibrate Metrics**
   - Run `calibrate_metrics.py` (may take time)
   - Review calibration config
   - Integrate into evaluation pipeline

4. **Implement Phase 5** (Caching)
   - Add embedding cache
   - Add artifact cache
   - Improve reproducibility

---

## 📝 Notes

- All scripts use deterministic seeding (seed=42) for reproducibility
- LAION subset is deterministically shuffled before capping
- Calibration thresholds are derived from perturbation distributions, not hand-picked
- Conditioning validation is non-fatal (warnings only) to allow experimentation

---

## 🆘 Troubleshooting

### If `run_laion_eval.py` fails:
- Check that LAION samples.json exists: `data/laion/samples.json`
- Verify Ollama is running (for text generation)
- Check that image index exists: `data/embeddings/image_index.npz`

### If `calibrate_metrics.py` is slow:
- Reduce `N_PER_PROMPT` in the script (default: 5)
- Reduce number of prompts in `CALIBRATION_PROMPTS`

### If conditioning validation shows many violations:
- Review `conditioning_analysis.json` for patterns
- Check if planner is producing complete plans
- Verify `plan_to_prompts()` is working correctly

---

## 📚 Documentation

- **Detailed Plan**: See `docs/execution_plan.md`
- **Architecture**: See `docs/architecture/` directory
- **Metrics**: See `docs/metrics/` directory

---

**Status**: Phases 1-4 implemented. Ready for integration and testing.
