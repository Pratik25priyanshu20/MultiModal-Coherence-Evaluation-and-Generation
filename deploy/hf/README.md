---
title: Multimodal Coherence AI
emoji: "\U0001f3a8"
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: "1.41.0"
app_file: app.py
pinned: false
license: mit
short_description: Coherent text + image + audio with MSCI
---

# Multimodal Coherence AI

Generate semantically coherent **text + image + audio** bundles and evaluate
cross-modal alignment using the **Multimodal Semantic Coherence Index (MSCI)**.

## How it works

1. **Text** — generated via HF Inference API
2. **Image** — retrieved from a curated index using CLIP (ViT-B/32) embeddings
3. **Audio** — retrieved from a curated index using CLAP (HTSAT-unfused) embeddings
4. **MSCI** — computed as `0.45 * cos_sim(text, image) + 0.45 * cos_sim(text, audio)`

## Research

This demo accompanies a study evaluating multimodal semantic coherence across
three research questions:

- **RQ1**: Is MSCI sensitive to controlled semantic perturbations? (Supported, d > 2.0)
- **RQ2**: Does structured planning improve cross-modal alignment? (Not supported)
- **RQ3**: Does MSCI correlate with human coherence judgments? (Supported, rho = 0.379)
