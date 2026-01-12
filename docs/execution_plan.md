# Multimodal Coherence AI - Execution Plan

## Overview

This document outlines the execution plan for completing the multimodal coherence evaluation system. The project is organized into 5 phases, with clear deliverables and success criteria.

## Architecture Layers

The system follows a layered pipeline architecture:

- **Layer 0 - Inputs**: Manual prompts or dataset prompts (LAION captions)
- **Layer 1 - Semantic Planning**: Converts raw prompt → structured semantic plan
- **Layer 2 - Generation**: Generates text, image, and audio conditioned on plan
- **Layer 3 - Embedding**: Converts modalities to vectors (CLIP/CLAP)
- **Layer 4 - Scoring**: Computes similarity scores (st_i, st_a, si_a, msci)
- **Layer 5 - Classification**: Diagnoses failure modes and classifies coherence

## Phase Status

### ✅ Phase 1: Dataset-Driven Evaluation Loop (COMPLETED)

**Goal**: Operationalize dataset evaluation with LAION captions

**Deliverables**:
- `scripts/run_laion_eval.py` - Main evaluation script
- `src/data/laion_loader.py` - Enhanced with deterministic seeding
- `runs/laion_eval/raw_results.json` - Structured results
- `runs/laion_eval/summary_msci.json` - Summary statistics

**Usage**:
```bash
python scripts/run_laion_eval.py
```

**Features**:
- Hard cap 500 samples with deterministic shuffle (seed=42)
- Semantic plan → generate → evaluate → store loop
- Per-caption statistics and aggregate summaries

---

### ✅ Phase 2: Automatic Failure Diagnostics (COMPLETED)

**Goal**: Systematic failure analysis with taxonomy

**Deliverables**:
- `scripts/analyze_laion_failures.py` - Failure analysis script
- `runs/laion_eval/diagnostics.json` - Failure mode counts and flags
- `runs/laion_eval/worst_examples/` - Top-10 worst examples with metadata

**Usage**:
```bash
python scripts/analyze_laion_failures.py
```

**Failure Modes Detected**:
- `AUDIO_ALIGNMENT_FAILURE` - Audio not matching text/image
- `PLANNER_AUDIO_UNDERSPECIFIED` - Plan missing audio elements
- `PLANNER_VISUAL_UNDERSPECIFIED` - Plan missing visual elements
- `TEXT_IMAGE_MISMATCH` - Text-image similarity very low
- `AUDIO_GENERATION_DRIFT` - Audio drifts from plan
- `IMAGE_GENERATION_DRIFT` - Image drifts from plan
- `GLOBAL_COHERENCE_FAILURE` - Negative MSCI

---

### ✅ Phase 3: Generator Conditioning Consistency (COMPLETED)

**Goal**: Ensure all generators use ONLY semantic plan, no raw prompt leakage

**Deliverables**:
- `src/pipeline/conditioning_validator.py` - Validation module
- `scripts/verify_conditioning.py` - Verification script
- Integrated validation in `generate_and_evaluate()`

**Usage**:
```bash
python scripts/verify_conditioning.py
```

**Validation Checks**:
- Keyword extraction from original prompt vs plan
- Detection of leaked keywords in modality prompts
- Plan completeness checks (scene, visual, audio, entities)

**Integration**:
- Validation runs automatically in `generate_and_evaluate()`
- Results stored in `bundle.meta["conditioning_validation"]`
- Warnings logged for non-fatal violations

---

### 🔄 Phase 4: Metric Calibration (IN PROGRESS)

**Goal**: Normalize similarity scores and calibrate thresholds from perturbation data

**Deliverables**:
- `scripts/calibrate_metrics.py` - Calibration script
- `src/coherence/normalized_scorer.py` - Normalized scoring module
- `runs/calibration/calibration_config.json` - Calibration parameters

**Usage**:
```bash
# Step 1: Run calibration experiments
python scripts/calibrate_metrics.py

# Step 2: Apply normalization to existing results
python scripts/apply_normalization.py  # (to be created)
```

**Calibration Process**:
1. Run batch perturbation experiments (baseline vs wrong modalities)
2. Compute normalization parameters (mean, std) from baseline distributions
3. Analyze separation between baseline and wrong-modality distributions
4. Derive thresholds based on separation analysis
5. Save calibration config for use in evaluation

**Next Steps**:
- Integrate normalized scorer into coherence engine
- Update thresholds.py to use calibration config
- Re-run evaluation with normalized metrics

---

### ⏳ Phase 5: Caching & Reproducibility (PENDING)

**Goal**: Add caching for embeddings, generated artifacts, and deterministic subsets

**Planned Deliverables**:
- Embedding cache (avoid recomputing embeddings for same inputs)
- Generated artifact cache (reuse generated images/audio when possible)
- Deterministic LAION subset caching
- Seed management for full reproducibility

---

## Quick Start Guide

### 1. Run Dataset Evaluation

```bash
# Evaluate 500 LAION captions
python scripts/run_laion_eval.py

# Analyze failures
python scripts/analyze_laion_failures.py

# Verify conditioning consistency
python scripts/verify_conditioning.py
```

### 2. Calibrate Metrics

```bash
# Run perturbation experiments and compute calibration
python scripts/calibrate_metrics.py

# Review calibration_config.json
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

## Key Files Reference

### Scripts
- `scripts/run_laion_eval.py` - Main dataset evaluation
- `scripts/analyze_laion_failures.py` - Failure diagnostics
- `scripts/verify_conditioning.py` - Conditioning validation
- `scripts/calibrate_metrics.py` - Metric calibration

### Core Modules
- `src/pipeline/generate_and_evaluate.py` - Main pipeline
- `src/pipeline/conditioning_validator.py` - Conditioning checks
- `src/coherence/normalized_scorer.py` - Normalized scoring
- `src/data/laion_loader.py` - LAION dataset loader

### Output Directories
- `runs/laion_eval/` - Dataset evaluation results
- `runs/calibration/` - Calibration experiments and config
- `runs/unified/` - Single-run results

## Success Criteria

### Phase 1 ✅
- [x] 500 LAION captions evaluated
- [x] Results stored in structured format
- [x] Summary statistics computed

### Phase 2 ✅
- [x] Failure modes automatically detected
- [x] Top-10 worst examples identified
- [x] Diagnostics JSON generated

### Phase 3 ✅
- [x] Conditioning validation implemented
- [x] Prompt leakage detection working
- [x] Validation integrated into pipeline

### Phase 4 🔄
- [ ] Calibration config generated
- [ ] Normalized scorer integrated
- [ ] Thresholds recalibrated
- [ ] MSCI distribution shifts upward

### Phase 5 ⏳
- [ ] Embedding cache implemented
- [ ] Artifact cache implemented
- [ ] Full reproducibility achieved

## Known Issues & Next Steps

### Current Issues
1. **Audio alignment**: st_a and si_a often low/negative → audio generation mismatch
2. **MSCI negative means**: Indicates embedding-space misalignment or generation drift
3. **AudioLDM warnings**: Missing safetensors, fallback to pickle loading

### Immediate Next Steps
1. Complete Phase 4 integration (normalized scorer in coherence engine)
2. Run full calibration and re-evaluate with normalized metrics
3. Analyze if normalization improves MSCI distribution
4. Investigate audio generation alignment issues

### Long-term Improvements
1. Improve audio generator conditioning on semantic plan
2. Consider alternative audio embedding models if CLAP misaligned
3. Add image generation (diffusion) instead of retrieval
4. Implement Phase 5 caching for speed

## Notes

- All scripts use deterministic seeding (seed=42) for reproducibility
- LAION subset is deterministically shuffled before capping
- Calibration thresholds are derived from perturbation distributions, not hand-picked
- Conditioning validation is non-fatal (warnings only) to allow experimentation
