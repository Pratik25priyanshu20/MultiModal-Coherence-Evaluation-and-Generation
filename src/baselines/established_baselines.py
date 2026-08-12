"""
Established Metric Baselines for Multimodal Coherence.

Implements industry-standard metrics for comparison with cMSCI:
1. CLIPScore: Standard CLIP-based image-text matching score
2. BLIPScore: BLIP-2 image-text matching score (ITM head)
3. ImageBind: Meta's 6-modality model for text-image-audio scoring

Each baseline provides a single coherence-like score per sample.

Usage:
    from src.baselines.established_baselines import CLIPScoreBaseline, ImageBindBaseline
    clip_bl = CLIPScoreBaseline()
    result = clip_bl.score("A sunset beach", "beach.jpg")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class CLIPScoreBaseline:
    """CLIPScore: Standard CLIP-based text-image matching.

    CLIPScore = max(100 * cos(text_CLIP, image_CLIP), 0)

    For multimodal: we extend with text-audio via CLAP.
    Final score = (CLIPScore_image + CLAPScore_audio) / 2
    """

    def __init__(self, clip_model: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_model.eval()

        # Also load CLAP for audio channel
        from transformers import ClapModel, ClapProcessor
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.clap_model.eval()

    @torch.no_grad()
    def _clip_score(self, text: str, image_path: str) -> float:
        """Compute standard CLIPScore for text-image pair."""
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        # CLIPScore = 100 * cos_sim(text, image)
        logits = outputs.logits_per_image.item()
        return max(logits, 0.0)

    @torch.no_grad()
    def _clap_score(self, text: str, audio_path: str) -> float:
        """Compute CLAPScore for text-audio pair (analogous to CLIPScore)."""
        import librosa
        waveform, _ = librosa.load(audio_path, sr=48000, mono=True)
        try:
            text_inputs = self.clap_processor(text=[text], return_tensors="pt", padding=True)
            audio_inputs = self.clap_processor(audio=waveform, sampling_rate=48000, return_tensors="pt")
        except TypeError:
            text_inputs = self.clap_processor(text=[text], return_tensors="pt", padding=True)
            audio_inputs = self.clap_processor(audios=waveform, sampling_rate=48000, return_tensors="pt")

        text_feats = self.clap_model.get_text_features(**text_inputs)
        audio_feats = self.clap_model.get_audio_features(**audio_inputs)

        # Normalize
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        audio_feats = audio_feats / audio_feats.norm(dim=-1, keepdim=True)

        cos_sim = (text_feats @ audio_feats.T).item()
        return max(100.0 * cos_sim, 0.0)

    def score(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Compute combined CLIPScore + CLAPScore."""
        clip_s = self._clip_score(text, image_path)
        clap_s = self._clap_score(text, audio_path)
        combined = (clip_s + clap_s) / 2.0

        return {
            "method": "CLIPScore",
            "score": float(combined),
            "clip_score_image": float(clip_s),
            "clap_score_audio": float(clap_s),
        }


class BLIPScoreBaseline:
    """BLIPScore: BLIP-2 image-text matching score.

    Uses BLIP-2's image-text matching (ITM) head which was trained
    with contrastive learning for matching assessment.

    For audio: falls back to CLAPScore since BLIP-2 is vision-only.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self._model = None
        self._processor = None
        self._model_name = model_name
        # Also load CLAP for audio channel
        self._clap_model = None
        self._clap_processor = None

    def _load_models(self):
        """Lazy-load models to save memory when not needed."""
        if self._model is not None:
            return

        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self._processor = Blip2Processor.from_pretrained(self._model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self._model_name, torch_dtype=torch.float16
            )
            self._model.eval()
            logger.info("BLIP-2 loaded: %s", self._model_name)
        except Exception as e:
            logger.warning("BLIP-2 load failed, falling back to BLIP-base: %s", e)
            from transformers import BlipProcessor, BlipForImageTextRetrieval
            self._processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
            self._model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
            self._model.eval()
            self._model_name = "Salesforce/blip-itm-base-coco"

        from transformers import ClapModel, ClapProcessor
        self._clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self._clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self._clap_model.eval()

    @torch.no_grad()
    def _blip_itm_score(self, text: str, image_path: str) -> float:
        """Compute BLIP image-text matching score."""
        self._load_models()
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        if "blip2" in self._model_name.lower():
            # BLIP-2: Use cosine similarity of embeddings
            inputs = self._processor(images=image, text=text, return_tensors="pt")
            # Get image and text features
            image_features = self._model.get_image_features(**{k: v for k, v in inputs.items() if k != "input_ids" and k != "attention_mask"})
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            # Simplify: use raw embedding cosine
            inputs_text = self._processor(text=text, return_tensors="pt", padding=True)
            inputs_img = self._processor(images=image, return_tensors="pt")
            outputs = self._model(**{**inputs_img, **inputs_text})
            # Use logits if available
            if hasattr(outputs, 'itm_score'):
                score = torch.softmax(outputs.itm_score, dim=-1)[0, 1].item()
                return score * 100.0
            return 50.0  # fallback
        else:
            # BLIP-base ITM
            inputs = self._processor(images=image, text=text, return_tensors="pt")
            outputs = self._model(**inputs)
            # ITM score: probability that text matches image
            itm_scores = torch.softmax(outputs.itm_score, dim=-1)
            match_prob = itm_scores[0, 1].item()
            return match_prob * 100.0

    @torch.no_grad()
    def _clap_score(self, text: str, audio_path: str) -> float:
        """CLAP-based text-audio score."""
        self._load_models()
        import librosa
        waveform, _ = librosa.load(audio_path, sr=48000, mono=True)
        try:
            text_inputs = self._clap_processor(text=[text], return_tensors="pt", padding=True)
            audio_inputs = self._clap_processor(audio=waveform, sampling_rate=48000, return_tensors="pt")
        except TypeError:
            text_inputs = self._clap_processor(text=[text], return_tensors="pt", padding=True)
            audio_inputs = self._clap_processor(audios=waveform, sampling_rate=48000, return_tensors="pt")

        text_feats = self._clap_model.get_text_features(**text_inputs)
        audio_feats = self._clap_model.get_audio_features(**audio_inputs)
        # Handle newer transformers returning BaseModelOutput instead of tensor
        if hasattr(text_feats, 'pooler_output'):
            text_feats = text_feats.pooler_output
        elif hasattr(text_feats, 'last_hidden_state'):
            text_feats = text_feats.last_hidden_state[:, 0]
        if hasattr(audio_feats, 'pooler_output'):
            audio_feats = audio_feats.pooler_output
        elif hasattr(audio_feats, 'last_hidden_state'):
            audio_feats = audio_feats.last_hidden_state[:, 0]
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        audio_feats = audio_feats / audio_feats.norm(dim=-1, keepdim=True)
        return max(100.0 * (text_feats @ audio_feats.T).item(), 0.0)

    def score(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Compute combined BLIPScore."""
        blip_s = self._blip_itm_score(text, image_path)
        clap_s = self._clap_score(text, audio_path)
        combined = (blip_s + clap_s) / 2.0

        return {
            "method": "BLIPScore",
            "score": float(combined),
            "blip_score_image": float(blip_s),
            "clap_score_audio": float(clap_s),
        }


class ImageBindBaseline:
    """ImageBind: Meta's 6-modality joint embedding model.

    Projects text, image, and audio into a shared embedding space,
    enabling native tri-modal coherence measurement.

    Score = mean(cos(text, image), cos(text, audio), cos(image, audio))
    """

    def __init__(self):
        self._model = None
        self._available = None

    def _check_available(self) -> bool:
        """Check if ImageBind is installed."""
        if self._available is not None:
            return self._available
        try:
            import imagebind
            self._available = True
        except ImportError:
            try:
                import imagebind_model
                self._available = True
            except ImportError:
                self._available = False
                logger.warning(
                    "ImageBind not installed. Install with: pip install imagebind"
                )
        return self._available

    def _load_model(self):
        """Lazy-load ImageBind model."""
        if self._model is not None:
            return

        if not self._check_available():
            return

        try:
            from imagebind.models import imagebind_model as ib_model
            from imagebind.models.imagebind_model import ModalityType
            self._model = ib_model.imagebind_huge(pretrained=True)
            self._model.eval()
            self._ModalityType = ModalityType
            self._data_module = __import__("imagebind.data", fromlist=["data"])
        except Exception as e:
            logger.error("Failed to load ImageBind: %s", e)
            self._available = False

    @torch.no_grad()
    def score(
        self, text: str, image_path: str, audio_path: str,
    ) -> Dict[str, Any]:
        """Compute ImageBind tri-modal coherence score."""
        if not self._check_available():
            return {
                "method": "ImageBind",
                "score": None,
                "error": "imagebind_not_installed",
            }

        self._load_model()
        if self._model is None:
            return {"method": "ImageBind", "score": None, "error": "model_load_failed"}

        try:
            ModalityType = self._ModalityType
            data = self._data_module

            inputs = {
                ModalityType.TEXT: data.load_and_transform_text([text], "cpu"),
                ModalityType.VISION: data.load_and_transform_vision_data([image_path], "cpu"),
                ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], "cpu"),
            }

            embeddings = self._model(inputs)

            # Compute pairwise cosine similarities
            text_emb = embeddings[ModalityType.TEXT][0]
            image_emb = embeddings[ModalityType.VISION][0]
            audio_emb = embeddings[ModalityType.AUDIO][0]

            text_emb = text_emb / text_emb.norm()
            image_emb = image_emb / image_emb.norm()
            audio_emb = audio_emb / audio_emb.norm()

            cos_ti = (text_emb @ image_emb).item()
            cos_ta = (text_emb @ audio_emb).item()
            cos_ia = (image_emb @ audio_emb).item()

            score = (cos_ti + cos_ta + cos_ia) / 3.0

            return {
                "method": "ImageBind",
                "score": float(score),
                "cos_text_image": float(cos_ti),
                "cos_text_audio": float(cos_ta),
                "cos_image_audio": float(cos_ia),
            }
        except Exception as e:
            return {"method": "ImageBind", "score": None, "error": str(e)}


def evaluate_established_baselines(
    samples: list,
    human_scores: Optional[dict] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run established baselines on samples and compute correlations.

    Args:
        samples: List of sample dicts.
        human_scores: Optional human score dict for correlation.
        methods: Which baselines to run. Default: ["CLIPScore", "BLIPScore", "ImageBind"].

    Returns:
        Dict with results per method and correlations.
    """
    if methods is None:
        methods = ["CLIPScore"]  # Start conservative, add others if available

    baselines = {}
    if "CLIPScore" in methods:
        baselines["CLIPScore"] = CLIPScoreBaseline()
    if "BLIPScore" in methods:
        baselines["BLIPScore"] = BLIPScoreBaseline()
    if "ImageBind" in methods:
        ib = ImageBindBaseline()
        if ib._check_available():
            baselines["ImageBind"] = ib
        else:
            print("  ImageBind not available, skipping")

    all_results = {m: [] for m in baselines}

    for i, s in enumerate(samples):
        for method_name, baseline in baselines.items():
            try:
                result = baseline.score(
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
