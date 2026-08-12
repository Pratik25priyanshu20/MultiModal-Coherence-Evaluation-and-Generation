"""
Simple Baselines for Multimodal Coherence.

Implements four simple baselines for comparison with cMSCI:
1. Raw cosine similarity: cos(text_CLIP, image_CLIP) + cos(text_CLAP, audio_CLAP)
2. Cosine + z-norm: Same but z-score normalized
3. Retrieval rank: 1/rank of matched item in retrieval results
4. Concatenated + cosine: Concat all embeddings, compute pairwise cosine

Usage:
    from src.baselines.simple_baselines import SimpleBaselines
    baselines = SimpleBaselines()
    results = baselines.evaluate_all(samples)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.embeddings.similarity import cosine_similarity


class SimpleBaselines:
    """Simple baseline methods for multimodal coherence scoring."""

    def __init__(
        self,
        calibration_path: Optional[str] = None,
        negative_bank_enabled: bool = True,
    ):
        self.embedder = AlignedEmbedder(target_dim=512)

        # Load calibration for z-norm baseline
        self._calibration = None
        if calibration_path and Path(calibration_path).exists():
            with open(calibration_path) as f:
                self._calibration = json.load(f)

        # Load negative bank for retrieval rank baseline
        self._neg_bank = None
        if negative_bank_enabled:
            try:
                from src.coherence.negative_bank import NegativeBank
                self._neg_bank = NegativeBank()
            except Exception:
                pass

    def raw_cosine(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Baseline 1: Raw cosine similarity average.

        Score = (cos(text_CLIP, image_CLIP) + cos(text_CLAP, audio_CLAP)) / 2
        """
        emb_text_clip = self.embedder.embed_text(text)
        emb_text_clap = self.embedder.embed_text_for_audio(text)
        emb_image = self.embedder.embed_image(image_path)
        emb_audio = self.embedder.embed_audio(audio_path)

        st_i = cosine_similarity(emb_text_clip, emb_image)
        st_a = cosine_similarity(emb_text_clap, emb_audio)
        score = (st_i + st_a) / 2.0

        return {
            "method": "raw_cosine",
            "score": float(score),
            "st_i": float(st_i),
            "st_a": float(st_a),
        }

    def cosine_znorm(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Baseline 2: Z-normalized cosine similarity.

        Z-normalize each channel then sigmoid to [0,1].
        Shows the effect of calibration alone (without Gramian or contrastive).
        """
        emb_text_clip = self.embedder.embed_text(text)
        emb_text_clap = self.embedder.embed_text_for_audio(text)
        emb_image = self.embedder.embed_image(image_path)
        emb_audio = self.embedder.embed_audio(audio_path)

        st_i = cosine_similarity(emb_text_clip, emb_image)
        st_a = cosine_similarity(emb_text_clap, emb_audio)

        if self._calibration is not None:
            z_i = (st_i - self._calibration["st_i"]["mean"]) / max(self._calibration["st_i"]["std"], 1e-10)
            z_a = (st_a - self._calibration["st_a"]["mean"]) / max(self._calibration["st_a"]["std"], 1e-10)
            z_mean = (z_i + z_a) / 2.0
            score = float(1.0 / (1.0 + np.exp(-z_mean)))
        else:
            score = (st_i + st_a) / 2.0
            z_i = z_a = None

        return {
            "method": "cosine_znorm",
            "score": float(score),
            "st_i": float(st_i),
            "st_a": float(st_a),
            "z_i": float(z_i) if z_i is not None else None,
            "z_a": float(z_a) if z_a is not None else None,
        }

    def retrieval_rank(
        self, text: str, image_path: str, audio_path: str, domain: str = "",
    ) -> Dict[str, Any]:
        """Baseline 3: Reciprocal retrieval rank.

        Rank the matched image/audio among all candidates by text similarity.
        Score = (1/rank_image + 1/rank_audio) / 2.
        """
        if self._neg_bank is None:
            return {"method": "retrieval_rank", "score": None, "error": "no_neg_bank"}

        emb_text_clip = self.embedder.embed_text(text)
        emb_text_clap = self.embedder.embed_text_for_audio(text)
        emb_image = self.embedder.embed_image(image_path)
        emb_audio = self.embedder.embed_audio(audio_path)

        # Compute similarity of matched image vs all images in bank
        matched_sim_i = cosine_similarity(emb_text_clip, emb_image)
        if hasattr(self._neg_bank, '_image_embs') and self._neg_bank._image_embs is not None:
            all_sims_i = self._neg_bank._image_embs @ (emb_text_clip.squeeze() / (np.linalg.norm(emb_text_clip.squeeze()) + 1e-12))
            rank_i = int(np.sum(all_sims_i >= matched_sim_i)) + 1
        else:
            rank_i = 1

        matched_sim_a = cosine_similarity(emb_text_clap, emb_audio)
        if hasattr(self._neg_bank, '_audio_embs') and self._neg_bank._audio_embs is not None:
            all_sims_a = self._neg_bank._audio_embs @ (emb_text_clap.squeeze() / (np.linalg.norm(emb_text_clap.squeeze()) + 1e-12))
            rank_a = int(np.sum(all_sims_a >= matched_sim_a)) + 1
        else:
            rank_a = 1

        rr_i = 1.0 / rank_i
        rr_a = 1.0 / rank_a
        score = (rr_i + rr_a) / 2.0

        return {
            "method": "retrieval_rank",
            "score": float(score),
            "rank_image": rank_i,
            "rank_audio": rank_a,
            "rr_image": float(rr_i),
            "rr_audio": float(rr_a),
        }

    def concatenated_cosine(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Baseline 4: Concatenated embedding cosine.

        Concatenate [text_CLIP, text_CLAP] and [image_CLIP, audio_CLAP],
        then compute single cosine similarity.
        """
        emb_text_clip = self.embedder.embed_text(text)
        emb_text_clap = self.embedder.embed_text_for_audio(text)
        emb_image = self.embedder.embed_image(image_path)
        emb_audio = self.embedder.embed_audio(audio_path)

        # Concatenate: text representation = [text_CLIP || text_CLAP]
        text_concat = np.concatenate([emb_text_clip.squeeze(), emb_text_clap.squeeze()])
        media_concat = np.concatenate([emb_image.squeeze(), emb_audio.squeeze()])

        score = cosine_similarity(text_concat, media_concat)

        return {
            "method": "concatenated_cosine",
            "score": float(score),
        }

    def evaluate_sample(
        self, text: str, image_path: str, audio_path: str, domain: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """Run all simple baselines on a single sample.

        Returns:
            Dict mapping method name to result dict.
        """
        return {
            "raw_cosine": self.raw_cosine(text, image_path, audio_path),
            "cosine_znorm": self.cosine_znorm(text, image_path, audio_path),
            "retrieval_rank": self.retrieval_rank(text, image_path, audio_path, domain),
            "concatenated_cosine": self.concatenated_cosine(text, image_path, audio_path),
        }

    def evaluate_all(
        self,
        samples: list,
        human_scores: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Run all baselines on a list of samples and optionally compute correlations.

        Args:
            samples: List of sample dicts with prompt_text, image_path, audio_path, domain.
            human_scores: Optional dict mapping sample_id to human score dict.

        Returns:
            Dict with per-sample results and optional correlation analysis.
        """
        all_results = []

        for i, s in enumerate(samples):
            result = self.evaluate_sample(
                text=s["prompt_text"],
                image_path=s.get("image_path", ""),
                audio_path=s.get("audio_path", ""),
                domain=s.get("domain", ""),
            )
            result["sample_id"] = s["sample_id"]
            all_results.append(result)
            print(f"  [{i+1}/{len(samples)}] {s['sample_id']}", end="\r")

        print()

        output = {"results": all_results}

        # Compute correlations if human scores available
        if human_scores is not None:
            methods = ["raw_cosine", "cosine_znorm", "retrieval_rank", "concatenated_cosine"]
            correlations = {}

            for method in methods:
                scores = []
                humans = []
                for r in all_results:
                    sid = r["sample_id"]
                    if sid in human_scores and r[method].get("score") is not None:
                        scores.append(r[method]["score"])
                        humans.append(human_scores[sid]["weighted_score"]["mean"])

                if len(scores) >= 5:
                    rho, p = sp_stats.spearmanr(scores, humans)
                    correlations[method] = {
                        "rho": float(rho),
                        "p": float(p),
                        "n": len(scores),
                        "significant": p < 0.05,
                    }

            output["correlations"] = correlations

        return output
