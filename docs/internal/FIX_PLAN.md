# Fix Plan: Improving Coherence Scores

## Issues Identified

1. **`plan_to_prompts` using wrong fields** - Not extracting from nested UnifiedPlan structure
2. **Image prompts too generic** - Not using plan details effectively
3. **Audio prompts too generic** - Not specific enough for AudioLDM
4. **No similarity filtering** - Image retrieval accepts any match
5. **No normalization** - Raw scores not calibrated

## Fixes Applied

### ✅ Fix 1: Correct Field Extraction in `plan_to_prompts`

**Problem**: Looking for `primary_entities`, `audio_elements`, `visual_attributes` at top level, but they're nested.

**Solution**: Extract from correct nested structure:
- `core_semantics.main_subjects` → primary entities
- `audio_constraints.sound_sources` + `ambience` → audio elements
- `style_controls.visual_style` + `image_constraints.objects` → visual attributes

**File**: `src/planner/schema_to_text.py`

### ✅ Fix 2: Improved Image Prompts

**Problem**: Image prompts too generic, not using plan details.

**Solution**: 
- Build richer prompts with scene_summary + subjects + visual details + style + mood
- Include core semantics (setting, time_of_day, weather)
- More specific for better retrieval matching

**File**: `src/planner/schema_to_text.py` (image_prompt section)

### ✅ Fix 3: Improved Audio Prompts

**Problem**: Audio prompts too generic, AudioLDM not getting enough detail.

**Solution**:
- Build specific prompts: scene + sound_sources + ambience + audio_intent
- Include weather/setting context
- Add tempo from audio_constraints
- Format for AudioLDM (concise, specific)

**File**: `src/planner/schema_to_text.py` (audio_prompt section)

### ✅ Fix 4: Better Image Retrieval

**Problem**: No similarity thresholding, accepts any match.

**Solution**: 
- Added `ImprovedImageRetrievalGenerator` with min_similarity threshold
- Filter results by similarity (default: 0.15)
- Warn if similarity is very low (< 0.2)

**File**: `src/generators/image/generator_improved.py` (new)

---

## Next Steps to Complete Fixes

### Step 1: Integrate Improved Image Generator

Update `src/pipeline/generate_and_evaluate.py` to use improved generator:

```python
# Replace:
from src.generators.image.generator import generate_image

# With:
from src.generators.image.generator_improved import generate_image_improved as generate_image
```

### Step 2: Add Normalization to Coherence Engine

Update `src/coherence/coherence_engine.py` to use normalized scores:

```python
from src.coherence.normalized_scorer import NormalizedScorer

class CoherenceEngine:
    def __init__(self, target_dim: int = 512, use_normalization: bool = True):
        self.embedder = AlignedEmbedder(target_dim=target_dim)
        self.scorer = CoherenceScorer()
        self.normalizer = NormalizedScorer() if use_normalization else None
    
    def evaluate(self, ...):
        # ... existing code ...
        
        # Apply normalization if enabled
        if self.normalizer and self.normalizer.is_calibrated():
            normalized_scores = self.normalizer.normalize_scores(scores)
            # Use normalized scores for classification
            # ...
```

### Step 3: Re-run Evaluation

After fixes, re-run evaluation:

```bash
# Re-run LAION evaluation
python scripts/run_laion_eval.py

# Re-analyze failures
python scripts/analyze_laion_failures.py
```

### Step 4: Verify Improvements

Check if:
- Mean MSCI improves from 0.0013 → > 0.1 (target: > 0.3)
- Image drift decreases from 100% → < 50%
- Audio drift decreases from 100% → < 50%
- Similarity scores improve (st_i, st_a, si_a > 0.1)

---

## Expected Improvements

### Before Fixes:
- Mean MSCI: 0.0013
- Image drift: 100% high
- Audio drift: 100% high
- All similarities: ~0.0

### After Fixes (Expected):
- Mean MSCI: 0.1-0.3 (10-100x improvement)
- Image drift: < 50% high
- Audio drift: < 50% high
- Similarities: 0.1-0.3 (should be positive)

---

## Testing Strategy

1. **Test on small subset first**:
   ```bash
   # Modify run_laion_eval.py to use MAX_SAMPLES = 10
   python scripts/run_laion_eval.py
   ```

2. **Compare before/after**:
   - Check if prompts are more specific
   - Check if image retrieval finds better matches
   - Check if audio prompts are more detailed

3. **Full evaluation**:
   - Run on full 500 samples
   - Compare metrics to baseline

---

## Rollback Plan

If fixes cause issues:
1. Revert `schema_to_text.py` changes
2. Keep improved generator as optional
3. Use feature flags to enable/disable improvements

---

## Additional Improvements (Future)

1. **Image Generation**: Replace retrieval with diffusion generation
2. **Better Audio Models**: Try alternative audio generation models
3. **Embedding Alignment**: Align CLIP and CLAP embedding spaces
4. **Plan Validation**: Reject incomplete plans before generation
