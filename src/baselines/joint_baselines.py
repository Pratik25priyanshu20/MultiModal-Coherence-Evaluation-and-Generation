"""
Joint Embedding Baselines for Multimodal Coherence.

Implements linear projection baselines:
1. CCA (Canonical Correlation Analysis): Learn projections maximizing correlation
   between CLIP and CLAP embedding spaces
2. Multi-view CCA: Regularized CCA for small sample sizes

These methods learn a shared space from training data, then measure coherence
as the correlation between projected embeddings.

Usage:
    from src.baselines.joint_baselines import CCABaseline
    cca = CCABaseline()
    cca.fit(train_embeddings)
    result = cca.score(text, image_path, audio_path)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from sklearn.cross_decomposition import CCA

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity

logger = logging.getLogger(__name__)


class CCABaseline:
    """CCA-based multimodal coherence scoring.

    Learns linear projections that maximize correlation between
    CLIP (text-image) and CLAP (text-audio) embedding spaces.

    Score = cosine similarity in CCA-projected space.
    """

    def __init__(self, n_components: int = 10, regularization: float = 0.1):
        self.n_components = n_components
        self.regularization = regularization
        self.embedder = AlignedEmbedder(target_dim=512)
        self._cca_ti = None  # Text-image CCA
        self._cca_ta = None  # Text-audio CCA
        self._fitted = False

    def fit(
        self,
        samples: list,
        exclude_ids: Optional[set] = None,
    ) -> None:
        """Fit CCA on training samples.

        Args:
            samples: List of sample dicts with prompt_text, image_path, audio_path.
            exclude_ids: Sample IDs to exclude (e.g., test set).
        """
        if exclude_ids is None:
            exclude_ids = set()

        text_clip_embs = []
        image_embs = []
        text_clap_embs = []
        audio_embs = []

        for s in samples:
            if s.get("sample_id") in exclude_ids:
                continue

            text_clip = self.embedder.embed_text(s["prompt_text"]).squeeze()
            text_clap = self.embedder.embed_text_for_audio(s["prompt_text"]).squeeze()
            image = self.embedder.embed_image(s["image_path"]).squeeze()
            audio = self.embedder.embed_audio(s["audio_path"]).squeeze()

            text_clip_embs.append(text_clip)
            image_embs.append(image)
            text_clap_embs.append(text_clap)
            audio_embs.append(audio)

        X_ti = np.stack(text_clip_embs)
        Y_ti = np.stack(image_embs)
        X_ta = np.stack(text_clap_embs)
        Y_ta = np.stack(audio_embs)

        # Fit CCA — n_components must be <= min(n_samples, n_features)
        n_comp = min(self.n_components, len(X_ti), X_ti.shape[1])

        self._cca_ti = CCA(n_components=n_comp)
        self._cca_ti.fit(X_ti, Y_ti)

        self._cca_ta = CCA(n_components=n_comp)
        self._cca_ta.fit(X_ta, Y_ta)

        self._fitted = True
        logger.info("CCA fitted with %d components on %d samples", n_comp, len(X_ti))

    def score(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Compute CCA-based coherence score."""
        if not self._fitted:
            return {"method": "CCA", "score": None, "error": "not_fitted"}

        text_clip = self.embedder.embed_text(text).reshape(1, -1)
        text_clap = self.embedder.embed_text_for_audio(text).reshape(1, -1)
        image = self.embedder.embed_image(image_path).reshape(1, -1)
        audio = self.embedder.embed_audio(audio_path).reshape(1, -1)

        # Project into CCA space
        tc_proj, img_proj = self._cca_ti.transform(text_clip, image)
        ta_proj, aud_proj = self._cca_ta.transform(text_clap, audio)

        # Cosine similarity in CCA space
        cos_ti = float(np.dot(tc_proj[0], img_proj[0]) / (
            np.linalg.norm(tc_proj[0]) * np.linalg.norm(img_proj[0]) + 1e-12
        ))
        cos_ta = float(np.dot(ta_proj[0], aud_proj[0]) / (
            np.linalg.norm(ta_proj[0]) * np.linalg.norm(aud_proj[0]) + 1e-12
        ))

        score = (cos_ti + cos_ta) / 2.0

        return {
            "method": "CCA",
            "score": float(score),
            "cos_ti_cca": float(cos_ti),
            "cos_ta_cca": float(cos_ta),
        }


class RegularizedCCABaseline:
    """Multi-view CCA with ridge regularization.

    Better suited for small sample sizes (n << d) by adding
    regularization to prevent overfitting of CCA projections.

    Uses manual implementation since sklearn CCA doesn't support
    explicit regularization parameter.
    """

    def __init__(self, n_components: int = 10, alpha: float = 1.0):
        self.n_components = n_components
        self.alpha = alpha
        self.embedder = AlignedEmbedder(target_dim=512)
        self._W_text = None
        self._W_image = None
        self._W_text_a = None
        self._W_audio = None
        self._fitted = False

    def _rcca_fit(
        self, X: np.ndarray, Y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit regularized CCA.

        Solves: max corr(X @ W_x, Y @ W_y) subject to ridge penalty.

        Args:
            X: (n, d1) first view.
            Y: (n, d2) second view.

        Returns:
            (W_x, W_y): Projection matrices.
        """
        n = X.shape[0]
        d1 = X.shape[1]
        d2 = Y.shape[1]

        # Center
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

        # Regularized covariance matrices
        Cxx = (X.T @ X) / n + self.alpha * np.eye(d1)
        Cyy = (Y.T @ Y) / n + self.alpha * np.eye(d2)
        Cxy = (X.T @ Y) / n

        # Solve via generalized eigenvalue problem
        # Cxx^{-1} Cxy Cyy^{-1} Cyx w = lambda^2 w
        Cxx_inv = np.linalg.solve(Cxx, np.eye(d1))
        Cyy_inv = np.linalg.solve(Cyy, np.eye(d2))

        M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T

        # Use SVD for numerical stability
        n_comp = min(self.n_components, n, d1, d2)
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        W_x = U[:, :n_comp]

        # Corresponding Y projections
        W_y = Cyy_inv @ Cxy.T @ W_x
        # Normalize columns
        for j in range(W_y.shape[1]):
            norm = np.linalg.norm(W_y[:, j])
            if norm > 1e-12:
                W_y[:, j] /= norm

        return W_x, W_y

    def fit(
        self,
        samples: list,
        exclude_ids: Optional[set] = None,
    ) -> None:
        """Fit regularized CCA on training samples."""
        if exclude_ids is None:
            exclude_ids = set()

        text_clip_embs = []
        image_embs = []
        text_clap_embs = []
        audio_embs = []

        for s in samples:
            if s.get("sample_id") in exclude_ids:
                continue

            text_clip_embs.append(self.embedder.embed_text(s["prompt_text"]).squeeze())
            image_embs.append(self.embedder.embed_image(s["image_path"]).squeeze())
            text_clap_embs.append(self.embedder.embed_text_for_audio(s["prompt_text"]).squeeze())
            audio_embs.append(self.embedder.embed_audio(s["audio_path"]).squeeze())

        X_ti = np.stack(text_clip_embs)
        Y_ti = np.stack(image_embs)
        X_ta = np.stack(text_clap_embs)
        Y_ta = np.stack(audio_embs)

        self._W_text, self._W_image = self._rcca_fit(X_ti, Y_ti)
        self._W_text_a, self._W_audio = self._rcca_fit(X_ta, Y_ta)
        self._X_ti_mean = X_ti.mean(axis=0)
        self._Y_ti_mean = Y_ti.mean(axis=0)
        self._X_ta_mean = X_ta.mean(axis=0)
        self._Y_ta_mean = Y_ta.mean(axis=0)

        self._fitted = True
        logger.info("Regularized CCA fitted on %d samples (alpha=%.2f)", len(X_ti), self.alpha)

    def score(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Compute regularized CCA coherence score."""
        if not self._fitted:
            return {"method": "RegCCA", "score": None, "error": "not_fitted"}

        text_clip = self.embedder.embed_text(text).squeeze() - self._X_ti_mean
        text_clap = self.embedder.embed_text_for_audio(text).squeeze() - self._X_ta_mean
        image = self.embedder.embed_image(image_path).squeeze() - self._Y_ti_mean
        audio = self.embedder.embed_audio(audio_path).squeeze() - self._Y_ta_mean

        # Project
        tc_proj = text_clip @ self._W_text
        img_proj = image @ self._W_image
        ta_proj = text_clap @ self._W_text_a
        aud_proj = audio @ self._W_audio

        cos_ti = float(np.dot(tc_proj, img_proj) / (
            np.linalg.norm(tc_proj) * np.linalg.norm(img_proj) + 1e-12
        ))
        cos_ta = float(np.dot(ta_proj, aud_proj) / (
            np.linalg.norm(ta_proj) * np.linalg.norm(aud_proj) + 1e-12
        ))

        score = (cos_ti + cos_ta) / 2.0

        return {
            "method": "RegCCA",
            "score": float(score),
            "cos_ti_rcca": float(cos_ti),
            "cos_ta_rcca": float(cos_ta),
        }


def evaluate_joint_baselines(
    samples: list,
    human_scores: Optional[dict] = None,
    test_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Run joint embedding baselines with proper train/test split.

    Args:
        samples: List of all sample dicts.
        human_scores: Optional human score dict.
        test_ids: Sample IDs for test set (excluded from CCA fitting).

    Returns:
        Dict with per-method results and correlations.
    """
    if test_ids is None:
        test_ids = set()

    baselines = {
        "CCA": CCABaseline(n_components=10),
        "RegCCA": RegularizedCCABaseline(n_components=10, alpha=1.0),
    }

    # Fit on training data only
    for name, bl in baselines.items():
        print(f"  Fitting {name}...")
        bl.fit(samples, exclude_ids=test_ids)

    # Evaluate on all samples
    all_results = {m: [] for m in baselines}
    for i, s in enumerate(samples):
        for method_name, bl in baselines.items():
            try:
                result = bl.score(
                    text=s["prompt_text"],
                    image_path=s.get("image_path", ""),
                    audio_path=s.get("audio_path", ""),
                )
                result["sample_id"] = s["sample_id"]
                all_results[method_name].append(result)
            except Exception as e:
                all_results[method_name].append({
                    "method": method_name,
                    "sample_id": s["sample_id"],
                    "score": None,
                    "error": str(e),
                })
        print(f"  [{i+1}/{len(samples)}] {s['sample_id']}", end="\r")

    print()

    # Compute correlations
    correlations = {}
    if human_scores is not None:
        for method_name, results in all_results.items():
            scores = []
            humans = []
            for r in results:
                sid = r["sample_id"]
                if sid in human_scores and r.get("score") is not None:
                    scores.append(r["score"])
                    humans.append(human_scores[sid]["weighted_score"]["mean"])

            if len(scores) >= 5:
                rho, p = sp_stats.spearmanr(scores, humans)
                correlations[method_name] = {
                    "rho": float(rho),
                    "p": float(p),
                    "n": len(scores),
                    "significant": p < 0.05,
                }

    return {
        "results": all_results,
        "correlations": correlations,
    }
