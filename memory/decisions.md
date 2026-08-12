# Decisions Log

Architectural, design, and strategic decisions made throughout the project.

---

## Embedding Architecture
- **CLIP** (openai/clip-vit-base-patch32, 512-d) for text-image shared space
- **CLAP** (laion/clap-htsat-unfused, 512-d) for text-audio shared space
- CLIP and CLAP are **separate spaces** — cross-space comparison requires trained bridge
- ProjectionHead uses **identity** when in_dim == out_dim (preserves pre-trained alignment)

## Scoring Metric: cMSCI
- Replaced simple MSCI (weighted cosine average) with calibrated cMSCI pipeline
- Pipeline: Gramian volume → z-norm → contrastive margin → Ex-MCR complementarity → ProbVLM adaptive weighting
- Variant F is the final production metric
- Optimized hyperparameters (LOO-CV, 100 samples): ALPHA=2, W_TI=0.15, W_3D=0.15, GAMMA=0.6, CAL_MODE="cosine"
- Previous 30-sample params: ALPHA=7, W_TI=0.30, W_3D=0.35, GAMMA=0.4, CAL_MODE="gram"

## Cross-Space Bridge
- Trained 590K-param bridge (CLIP→CLAP) on 10,255 pairs
- Data sources: OmniBench + domain-matched + AudioCaps + augmented
- Bridge enables image-audio similarity (si_a), previously omitted

## Probabilistic Adapters
- BayesCap-style adapters for CLIP (592K params) and CLAP (592K params)
- Enable per-sample uncertainty estimation and adaptive channel weighting
- Trained on same expanded dataset

## Paper Framing
- Paper reframed around **generative + hybrid** approaches (retrieval removed from main analysis)
- Three RQs: perturbation sensitivity, planning effectiveness, human alignment

## Data Strategy
- 100 human-rated samples, 5 raters (8 on original 30-sample subset)
- Raters: Pratik, Dr Nikhil, Fariha, Jimmy, Satyam Shivam, Priyanshu Priyam, Nitish Kumar, Aston John
- Dev/test split: 70 dev / 30 test, stratified by domain × condition, seed=2024
- LOO-CV on all 100 samples for optimization; test set for additional validation
- Centralized sample paths in `src/config/settings.py` (RQ3_SAMPLES_PATH, RQ3_SAMPLES_EXTENDED_PATH)
- Indexes: 57 images, 104 audio files across nature/urban/water/other domains

## Audio Processing
- AUDIO_USE_WINDOWED = False (paper setting — single-clip CLAP)
- Silence detection at -40 dB threshold, 70% fraction = mostly silent

## Key Finding: Planning Hurts Coherence
- RQ2 showed structured planning **reduces** coherence (d = -0.82 to -1.51)
- Root cause: CLIP's 77-token context window truncates verbose planned prompts

## cMSCI v2 (Gemini Embedding 2)
- Added Gemini Embedding 2 as a second embedding backend to prove cMSCI is embedding-agnostic
- Unified 3072-d space eliminates bridge, ExMCR, and ProbVLM requirements
- Novel contribution: Matryoshka Scale Consistency (training-free uncertainty via MRL truncation)
- 5 variants (A-E) vs v1's 6 (A-F) — simpler pipeline, same calibration methodology
- 3-channel contrastive margins and exact 3D Gramian are now possible
- Regardless of outcome (v2>v1, v2~v1, v2<v1), it strengthens the paper

## Pip Package: cmsci-package/ (2026-08-02)
- cMSCI released as installable package in dedicated `cmsci-package/` folder (self-contained, does not touch research code)
- Curated copy of scoring modules (coherence + embeddings + config + utils), imports renamed src.* -> cmsci.*
- Trained assets (~9 MB: calibration, Ex-MCR, bridge, prob adapters, negative-bank indexes) bundled as package data
- Public API: `from cmsci import CoherenceScorer`; CLI: `cmsci --text ... --image ... --audio ...`
- Install: `pip install "cmsci @ git+<repo-url>#subdirectory=cmsci-package"`
- Root .gitignore got negation rules so bundled assets are not excluded by `models/` pattern
- v2 (Gemini) intentionally NOT in v0.1 package — v1 CLIP+CLAP pipeline only
