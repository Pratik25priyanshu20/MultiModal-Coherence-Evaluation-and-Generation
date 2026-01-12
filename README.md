# Multimodal Coherence Generation & Evaluation Framework

This project presents a **research-driven system for analyzing, diagnosing, and improving semantic coherence in multimodal generative AI systems**, with a focus on **text–image–audio generation pipelines**.

Unlike prior work that evaluates modalities independently, this framework treats **coherence as a first-class systems problem**, introducing structured planning, diagnostic metrics, and controlled experiments to understand *why* multimodal generation fails and how it can be fixed.

---

## Problem Statement

Modern multimodal generative models can produce convincing outputs per modality, yet frequently fail at **cross-modal semantic alignment**.

Typical failure cases include:
- Images that do not reflect the textual intent
- Audio that is ambient but semantically irrelevant
- Compounding drift across generation stages

**Key question:**  
How can we systematically measure, diagnose, and improve semantic coherence across text, image, and audio generation?

---

## System Overview (Layered Pipeline)

```text
Caption / Prompt
  ↓
Semantic Planning (LLM)
  ↓
Modality-Specific Prompt Generation
  ├─ Image Prompt
  └─ Audio Prompt
  ↓
Multimodal Generation
  ├─ Text Generator
  ├─ Image Generator
  └─ Audio Generator
  ↓
Embedding & Similarity Analysis
  ├─ CLIP (Text–Image)
  ├─ CLAP (Text–Audio)
  └─ Image–Audio Consistency
  ↓
MSCI Scoring + Failure Diagnostics
  ↓
Semantic Drift Analysis (Plan → Outputs)
```

---

## Core Components

### 1) Semantic Planner
- Uses an LLM to decompose input captions into structured semantics.
- Extracts:
  - Scene
  - Time
  - Environment
  - Visual elements
  - Audio elements
  - Mood and motion

This converts free-form text into machine-interpretable intent.

### 2) Canonical Semantic Plan (CSP)
All modality prompts are derived exclusively from a canonical plan, not the raw prompt.

Example schema (simplified):
```text
Scene {
  setting
  time
  weather
  primary_entities
  visual_elements
  audio_elements
  mood
  motion
}
```

Why this matters:
- Prevents prompt drift
- Enforces shared semantics
- Creates a single source of truth across modalities

### 3) Multimodal Generation
- Image generation conditioned on CSP-derived visual prompts
- Audio generation conditioned on CSP-derived audio prompts
- Text generation uses canonical plan text (no raw prompt)

Generation is modular and swappable. Alignment is measured, not trusted.

### 4) Embedding & Similarity Models
- CLIP → Text–Image similarity
- CLAP → Text–Audio similarity
- Image–Audio similarity → Cross-modal consistency

All similarity scores are logged per run.

### 5) Multimodal Semantic Coherence Index (MSCI)
MSCI is a unified coherence score derived from:
- Text–Image similarity
- Text–Audio similarity
- Image–Audio similarity

It enables:
- Dataset-level evaluation
- Threshold-based classification
- Failure mode detection

### 6) Failure Diagnostics
Each run is automatically classified into:
- GLOBAL_FAILURE
- AUDIO_ALIGNMENT_FAILURE
- LOCAL_MODALITY_WEAKNESS
- MODALITY_FAILURE
- HIGH_COHERENCE

Planner-level diagnostics detect:
- Missing visual elements
- Missing audio elements
- Underspecified semantics

This converts qualitative failure into actionable engineering signals.

### 7) Semantic Drift Analysis
We explicitly measure drift between:
- Canonical plan embedding
- Generated text embedding
- Generated image embedding
- Generated audio embedding

This highlights which modality deviates most from the plan.

### 8) Council-Lite Semantic Planning (Implemented)
To reduce planner underspecification, the project includes **Council-Lite**:
- Multiple lightweight planners generate independent semantic plans
- Plans are merged via intersection + confidence weighting
- Only shared, high-confidence semantics propagate

Benefits:
- Reduces hallucination
- Improves robustness
- Avoids heavy multi-agent orchestration

---

## Deterministic Baseline Mode (Stability First)

This is mandatory before adding data, learning weights, or UI.

Goal:  
**Same prompt → same outputs → same coherence score (± tiny noise)**

Determinism enforced via:
- Global seed for Python, NumPy, Torch
- Deterministic flags for text/audio generation

See:
- `src/utils/seed.py`
- `src/pipeline/generate_and_evaluate.py`

---

## Dataset-Scale Evaluation (LAION Prompts)

We provide a streaming subset builder for 500 captions:
- `scripts/build_laion_subset.py`

This enables automated generation → scoring → logging.

Example observed behavior (when running 500 captions):
- Mean MSCI near 0
- High variance
- Weak separation between correct vs. perturbed samples

This indicates coherence failure is systemic, not random noise.

---

## Metric Calibration (Perturbation Tests)

To validate MSCI reliability:
- Baseline generations are compared against:
  - Wrong-image substitutions
  - Wrong-audio substitutions
- Distribution overlap and separation are measured
- Normalized thresholds are derived

Outputs:
- `artifacts/coherence_stats.json`
- `artifacts/thresholds_frozen.json`
- `runs/perturbation/perturbation_results.json`

---

## Quickstart

### 1) Single run
```bash
python scripts/run_unified.py
```

### 2) Batch run
```bash
python scripts/run_unified_batch.py
```

### 3) Dataset eval (wikimedia / audiocaps / laion)
```bash
python scripts/run_dataset_eval.py wikimedia
```

### 4) LAION prompt subset (500)
```bash
python scripts/build_laion_subset.py
```

---

## Key Artifacts

Per run:
- `bundle.json` (plan, prompts, outputs, scores, drift)
- `run.json` (legacy logs)

Analytics:
- `artifacts/coherence_stats.json`
- `artifacts/dataset_summary.json`
- `artifacts/failure_analysis.json`
- `artifacts/unified_batch_report.json`

---

## Project Structure (Selected)

- `src/planner/` – planning, schema, council-lite, prompt compilation
- `src/generators/` – text, image, audio generators
- `src/embeddings/` – CLIP/CLAP embedding + projection
- `src/coherence/` – MSCI, thresholds, classifier, drift
- `src/explainability/` – diagnostics, drift analysis
- `scripts/` – pipeline runners and evaluation tools
- `data/` – datasets and processed assets

---

## Future Enhancements

- Extend to video generation and video-text/audio coherence
- Cross-modal attention constraints during generation
- Human-aligned coherence evaluation and calibration

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Ollama (local LLM) is recommended for deterministic planning.

---

## License

See `LICENSE`.
