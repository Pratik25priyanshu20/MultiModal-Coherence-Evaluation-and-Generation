# How to Fix the Coherence Problems

## ✅ Fixes Applied

I've implemented **4 critical fixes** to address the coherence issues:

### Fix 1: Correct Field Extraction ✅
**Problem**: `plan_to_prompts` was looking for wrong field names (top-level `primary_entities`, `audio_elements`, etc.) but the actual plan has nested structure.

**Solution**: Updated `src/planner/schema_to_text.py` to extract from correct nested fields:
- `core_semantics.main_subjects` → primary entities
- `audio_constraints.sound_sources` + `ambience` → audio elements  
- `style_controls.visual_style` + `image_constraints.objects` → visual attributes

**Impact**: Prompts now use actual plan details instead of being empty/generic.

---

### Fix 2: Improved Image Prompts ✅
**Problem**: Image prompts were too generic ("A single high-quality image of the following scene").

**Solution**: Build richer prompts with:
- Scene summary + main subjects + visual details
- Style, mood, setting, time_of_day, weather
- More specific for better retrieval matching

**Impact**: Image retrieval should find better matches.

---

### Fix 3: Improved Audio Prompts ✅
**Problem**: Audio prompts were too generic, AudioLDM wasn't getting enough detail.

**Solution**: Build specific prompts with:
- Scene + sound_sources + ambience + audio_intent
- Weather/setting context
- Tempo from audio_constraints
- Formatted for AudioLDM (concise, specific)

**Impact**: Audio generation should better match the semantic plan.

---

### Fix 4: Better Image Retrieval ✅
**Problem**: No similarity thresholding - accepts any match even if similarity is 0.0.

**Solution**: Created `ImprovedImageRetrievalGenerator` with:
- Minimum similarity threshold (default: 0.15)
- Filters low-similarity matches
- Warns if similarity is very low

**Impact**: Only retrieves images that actually match the prompt.

---

## 🧪 How to Test the Fixes

### Step 1: Test on Small Sample

```bash
python scripts/apply_fixes_and_test.py
```

This will:
- Test 3 sample prompts
- Show generated prompts (should be more specific now)
- Show scores (should be better)
- Save results to `runs/fix_test/test_results.json`

**What to check**:
- Image prompts should include subjects, style, mood
- Audio prompts should include sound sources, ambience
- Scores should be > 0.01 (better than baseline 0.0013)

---

### Step 2: Re-run Diagnostics (Fixed)

```bash
# Re-run with fixed diagnostics script
python scripts/analyze_laion_failures.py
```

This will now correctly check plan completeness using the right field names.

---

### Step 3: Full Evaluation (Optional)

If small test looks good, re-run full evaluation:

```bash
# Re-run on 500 LAION captions
python scripts/run_laion_eval.py

# Re-analyze
python scripts/analyze_laion_failures.py
```

**Expected improvements**:
- Mean MSCI: 0.0013 → 0.1-0.3 (10-100x improvement)
- Image drift: 100% → < 50%
- Audio drift: 100% → < 50%
- Similarities: 0.0 → 0.1-0.3

---

## 📊 What Changed in Code

### Files Modified:
1. `src/planner/schema_to_text.py` - Fixed field extraction, improved prompts
2. `src/pipeline/generate_and_evaluate.py` - Uses improved image generator
3. `scripts/analyze_laion_failures.py` - Fixed plan completeness checking

### Files Created:
1. `src/generators/image/generator_improved.py` - Better retrieval with filtering
2. `scripts/apply_fixes_and_test.py` - Test script
3. `FIX_PLAN.md` - Detailed fix documentation

---

## 🔍 Verification Checklist

After running tests, verify:

1. **Prompts are specific**:
   - Open `runs/fix_test/*/bundle.json`
   - Check `meta.modality_prompts.image_prompt` - should include subjects, style, mood
   - Check `meta.modality_prompts.audio_prompt` - should include sound sources, ambience

2. **Scores improved**:
   - MSCI should be > 0.01 (vs baseline 0.0013)
   - Similarities (st_i, st_a, si_a) should be > 0.05

3. **Drift decreased**:
   - Image drift should be < 0.8 (vs baseline ~1.0)
   - Audio drift should be < 0.8 (vs baseline ~1.0)

---

## 🚨 If Fixes Don't Work

### Issue: Prompts still generic
- Check if plan structure is correct in `bundle.json`
- Verify `core_semantics`, `style_controls`, `audio_constraints` exist

### Issue: Scores still low
- Check if image index has relevant images
- Verify audio generation is using the prompt
- Check embedding alignment (CLIP vs CLAP)

### Issue: Image retrieval fails
- Check if `data/embeddings/image_index.npz` exists
- Verify images in `data/processed/images/` match LAION captions
- May need to rebuild image index with more diverse images

---

## 📈 Next Steps After Fixes

1. **Apply Normalization** (if calibration config exists):
   - Integrate `NormalizedScorer` into `CoherenceEngine`
   - This will normalize scores using calibration data

2. **Improve Image Dataset**:
   - Add more diverse images to `data/processed/images/`
   - Rebuild image index
   - Consider image generation instead of retrieval

3. **Improve Audio Generation**:
   - Test different audio models
   - Improve audio prompt templates
   - Add audio quality validation

4. **Plan Quality**:
   - Add validation to reject incomplete plans
   - Improve planner prompts to ensure all fields populated

---

## 💡 Key Insights

The main problem was **prompt generation not using plan details**. The fixes ensure:
1. ✅ Correct field extraction from nested plan structure
2. ✅ Rich, specific prompts for each modality
3. ✅ Better filtering/validation in retrieval

These should significantly improve coherence scores!
