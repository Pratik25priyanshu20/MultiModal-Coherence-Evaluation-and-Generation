# Phase 1 Semantic Backbone (Task 1)

## Purpose of the Semantic Plan

The semantic plan is the single source of truth for meaning. It is shared across
text, image, and audio (and later video). The plan is human-readable,
machine-verifiable, and directly comparable against outputs. If something is not
in the plan, it is not expected in the output.

This contract enables drift detection, explainability, regeneration, and MSCI
evaluation.

## Semantic Plan V1 Schema (Locked)

Core intent:
- `scene_summary`: One-sentence description of the scene or idea.
- `domain`: The usage domain (marketing, education, entertainment, etc.).

Entities and objects:
- `primary_entities`: Must-appear subjects or objects.
- `secondary_entities`: Optional background elements.

Attributes and visual traits:
- `visual_attributes`: Colors, materials, lighting, time of day.
- `style`: Cinematic, realistic, minimal, abstract, etc.

Emotional and narrative semantics:
- `mood_emotion`: Calm, tense, joyful, mysterious, etc.
- `narrative_tone`: Hopeful, dramatic, playful, etc.

Audio semantics:
- `audio_intent`: Ambient, energetic, slow, rhythmic, etc.
- `audio_elements`: Rain, wind, soft synths, orchestral pads, etc.

Constraints:
- `must_include`: Non-negotiable concepts.
- `must_avoid`: Forbidden elements.

Metadata:
- `complexity_level`: `low`, `medium`, or `high`.
- `risk_flags`: Ambiguity, abstract concepts, emotional conflict, etc.

## Council-Lite Design (Locked)

Two mandatory planners plus one optional audio planner:

- Planner A (Core Semantics): scene, entities, constraints (conservative, literal).
- Planner B (Mood and Style): emotion, tone, aesthetics (catches emotional drift).
- Planner C (Optional, Audio Bias): soundscape and rhythm.

Each planner returns a full SemanticPlan plus confidence scores per section.

## Merge Logic (Locked)

1. Field-wise comparison by section.
2. Agreement detection:
   - Exact match -> accept.
   - Partial overlap -> merge.
   - Conflict -> flag.
3. Conflict resolution:
   - Factual fields (entities, constraints, core intent) -> Planner A wins.
   - Stylistic or emotional fields -> Planner B wins.
   - Audio fields -> Planner C wins if present, otherwise Planner B.
   - Unresolved -> mark as uncertain.
4. Final output includes resolved fields plus uncertainty annotations.

## Semantic Agreement Score

Agreement score (0-1) is derived from per-field agreement, averaged per section,
with penalties for:
- Conflicting entities
- Conflicting moods
- Contradictory constraints

We store:
- Global agreement score
- Per-section agreement score
- Per-field agreement score

## Validation Rules

Hard validation (errors):
- Empty `scene_summary`
- No `primary_entities`
- Missing `mood_emotion`
- Missing `audio_intent`

Soft validation (warnings):
- Too many entities
- Conflicting emotions
- High abstraction + low agreement

Failures are logged and surfaced for explainability.

## Output Artifacts (Phase 1)

For each prompt/run, we produce:
1. Final merged Semantic Plan (JSON-like structure)
2. Planner A and B raw plans (Planner C optional)
3. Agreement scores (global + per-section + per-field)
4. Conflict annotations and uncertainty fields

## Implementation References

- Schema: `src/planner/schema.py`
- Council roles: `src/planner/council.py`
- Merge logic: `src/planner/merge_logic.py`
- Validation rules: `src/planner/validators.py`

## Phase 1 – Semantic Backbone Evaluation

### Objective
Validate whether a council-based semantic plan combined with joint embeddings
can measure multimodal coherence reliably.

### Components Evaluated
- Semantic Planning Council (3 planners)
- CLIP + CLAP embedding bridge
- MSCI v0
- Drift detection logic

### Dataset
- Gold Benchmark v0 (Mixed General)
- ~XX samples (pilot)

### Results
- MSCI distribution summary
- Human vs MSCI alignment
- Drift frequency by modality

### Key Observations
- Text–Image alignment is strongest
- Audio drift is more frequent in abstract scenes
- Council disagreement correlates with lower MSCI

### Limitations
- No fine-tuning
- Small gold dataset
- CPU inference

### Next Steps
- Generation pipeline (Phase-2)
- Self-correction loop
- Larger benchmark
