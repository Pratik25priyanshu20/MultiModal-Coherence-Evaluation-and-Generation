---
title: Multimodal Coherence Evaluation
emoji: "\U0001F3AF"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: false
license: mit
short_description: Rate text+image+audio coherence bundles
---

# Multimodal Coherence Evaluation

Human evaluation interface for rating text + image + audio coherence bundles.

## How to use

1. Enter your name
2. Select dataset size (30 or 100 samples)
3. Click "Start New Session"
4. Rate each sample on 4 dimensions (text-image, text-audio, image-audio, overall)
5. **Download your session JSON when done** (data is ephemeral)

## For researchers

Session files are JSON and can be downloaded individually or as a ZIP archive
from the admin panel on the login page.
