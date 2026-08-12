# cMSCI — Calibrated Multimodal Semantic Coherence Index

[![PyPI](https://img.shields.io/pypi/v/cmsci)](https://pypi.org/project/cmsci/)
[![Python](https://img.shields.io/pypi/pyversions/cmsci)](https://pypi.org/project/cmsci/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Score how semantically coherent a **text / image / audio** triple is, with a single
calibrated metric.

cMSCI evaluates coherence using pre-aligned embedding spaces (CLIP for text–image,
CLAP for text–audio), Gramian-volume geometry, distribution calibration, contrastive
margins against hard negatives, a trained cross-space bridge for the image–audio
channel, and probabilistic uncertainty weighting.

On a 100-sample human evaluation (5 raters), cMSCI correlates with human coherence
judgments at **Spearman ρ = 0.785** (p < 1e-6), versus 0.558 for the uncalibrated
MSCI baseline.

## Install

```bash
pip install cmsci
```

Or the latest development version straight from GitHub:

```bash
pip install "cmsci @ git+https://github.com/Pratik25priyanshu20/MultiModal-Coherence-Evaluation-and-Generation.git#subdirectory=cmsci-package"
```

All trained artifacts (calibration statistics, Ex-MCR projector, cross-space
bridge, probabilistic adapters, negative-bank indexes — ~9 MB) are bundled;
the CLIP and CLAP backbones download automatically from Hugging Face on first use.

## Usage

### Python

```python
from cmsci import CoherenceScorer

scorer = CoherenceScorer()
result = scorer.score(
    text="rain falling in a dense forest",
    image="scene.png",     # optional
    audio="rain.wav",      # optional
    domain="nature",       # optional hint for harder negatives
)

print(result.cmsci)           # calibrated coherence score
print(result.msci)            # legacy weighted-cosine baseline
print(result.st_i, result.st_a, result.si_a)   # channel similarities
print(result.variant_scores)  # ablation variants A–F
print(result.uncertainty)     # probabilistic uncertainty estimates
```

Batch scoring:

```python
results = scorer.score_batch([
    {"text": "waves on a beach", "image": "beach.jpg", "audio": "waves.wav"},
    {"text": "a busy city street", "image": "street.jpg", "audio": "traffic.wav"},
])
```

### Command line

```bash
cmsci --text "rain falling in a dense forest" --image scene.png --audio rain.wav
cmsci --text "..." --image scene.png --json     # full result as JSON
```

## How it works

| Stage | Component |
| --- | --- |
| Embedding | CLIP (`openai/clip-vit-base-patch32`) for text/image, CLAP (`laion/clap-htsat-unfused`) for text/audio |
| Geometry | Gramian volume of the embedding set — 0 = aligned, 1 = orthogonal |
| Calibration | Per-channel z-score normalization from baseline statistics |
| Contrastive | Margin vs. hard negatives from the bundled embedding indexes |
| Cross-space | Trained 590K-param bridge enables the image–audio channel |
| Uncertainty | ProbVLM-style adapters adaptively weight channels |

Scores are calibrated against the paper's baseline distribution. For very
different domains, recalibration is possible — point `CMSCI_ASSETS_DIR` at a
directory with your own `artifacts/cmsci_calibration.json` (same layout as the
bundled assets).

## Configuration

| Env var | Effect |
| --- | --- |
| `CMSCI_ASSETS_DIR` | Use a custom assets directory instead of the bundled one |
| `CMSCI_CACHE_DIR` | Embedding cache location (default `~/.cache/cmsci`) |

## Notes

- Missing assets degrade gracefully: without the bridge the `si_a` channel is
  omitted; without calibration, raw uncalibrated scores are returned.
- CPU works fine for scoring; GPU is only beneficial for large batches.

## Citation

If you use cMSCI in academic work, please cite the accompanying paper
(reference to be added upon publication).

## License

MIT
