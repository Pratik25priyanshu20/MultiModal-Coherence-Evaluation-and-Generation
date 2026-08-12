"""
cMSCI — Calibrated Multimodal Semantic Coherence Index.

Evaluates the semantic coherence of text / image / audio triples using
pre-aligned embedding spaces (CLIP, CLAP), Gramian-volume geometry,
distribution calibration, contrastive margins, and probabilistic
uncertainty weighting.

Quickstart:
    from cmsci import CoherenceScorer

    scorer = CoherenceScorer()
    result = scorer.score(text="waves crashing on a beach",
                          image="beach.jpg", audio="waves.wav")
    print(result.cmsci)
"""

__version__ = "0.1.0"

__all__ = ["CoherenceScorer", "ScoreResult", "resolve_assets", "AssetPaths"]


def __getattr__(name):
    # Lazy imports keep `import cmsci` light (no torch/transformers load).
    if name in ("CoherenceScorer", "ScoreResult"):
        from cmsci import api

        return getattr(api, name)
    if name in ("resolve_assets", "AssetPaths"):
        from cmsci import assets

        return getattr(assets, name)
    raise AttributeError(f"module 'cmsci' has no attribute {name!r}")
