"""
Gemini Embedding 2 Unified Embedder.

Uses Google's gemini-embedding-2-preview model which places text, image,
and audio in a single unified 3072-d embedding space. This eliminates the
CLIP/CLAP separate-space problem entirely.

Key differences from CLIP+CLAP:
- ONE text encoder (not two separate ones)
- Image-audio similarity is directly meaningful (no bridge needed)
- 3072-d output with Matryoshka support (768, 1536, 3072)
- All pairwise and 3-way Gramian volumes are exact (no ExMCR approximation)

Usage:
    embedder = GeminiEmbedder()
    t = embedder.embed_text("A beach at sunset")
    i = embedder.embed_image("beach.jpg")
    a = embedder.embed_audio("waves.wav")
    # All three are in the same 3072-d space!

    # Multi-scale for Matryoshka uncertainty:
    scales = embedder.embed_all_scales("text", "img.jpg", "aud.wav")
    # scales[768]["text"], scales[1536]["image"], etc.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import numpy as np

from src.config.settings import (
    GEMINI_API_KEY,
    GEMINI_CACHE_DIR,
    GEMINI_MODEL_ID,
    GEMINI_OUTPUT_DIM,
    GEMINI_TASK_TYPE,
    MATRYOSHKA_DIMS,
)

logger = logging.getLogger(__name__)

# Rate limiting constants
_MAX_RETRIES = 5
_BASE_DELAY = 1.0  # seconds
_MAX_DELAY = 60.0  # seconds


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector."""
    v = v.astype(np.float64)
    norm = np.linalg.norm(v) + eps
    return v / norm


def _cache_key(prefix: str, content_id: str) -> str:
    """Generate a deterministic cache key."""
    h = hashlib.sha256(f"{prefix}:{content_id}".encode()).hexdigest()[:16]
    return f"{prefix}_{h}"


class GeminiEmbedder:
    """
    Unified text/image/audio embedder using Gemini Embedding 2.

    All outputs are in the same 3072-d space and L2-normalized.
    Supports Matryoshka truncation for multi-scale analysis.
    """

    _instance: Optional["GeminiEmbedder"] = None
    _lock = Lock()

    def __init__(
        self,
        model_id: str = GEMINI_MODEL_ID,
        output_dim: int = GEMINI_OUTPUT_DIM,
        api_key: str = "",
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
        task_type: str = "",
    ):
        self._model_id = model_id
        self._output_dim = output_dim
        self._api_key = api_key or GEMINI_API_KEY
        self._cache_dir = Path(cache_dir) if cache_dir else GEMINI_CACHE_DIR
        self._enable_cache = enable_cache
        self._task_type = task_type or GEMINI_TASK_TYPE
        self._client = None

        if self._enable_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def shared(cls) -> "GeminiEmbedder":
        """Return a singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_client(self):
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Run: pip install google-genai"
                )
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set. "
                    "Get a key at https://aistudio.google.com/apikey"
                )
            self._client = genai.Client(api_key=self._api_key)
            logger.info("Gemini client initialized (model=%s)", self._model_id)
        return self._client

    def _call_api_with_retry(self, contents, task_type: str = "") -> np.ndarray:
        """Call Gemini embed API with exponential backoff."""
        from google.genai import types

        task_type = task_type or self._task_type
        client = self._get_client()
        delay = _BASE_DELAY

        for attempt in range(_MAX_RETRIES):
            try:
                result = client.models.embed_content(
                    model=self._model_id,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._output_dim,
                    ),
                )
                emb = np.array(result.embeddings[0].values, dtype=np.float64)
                return _l2_normalize(emb)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < _MAX_RETRIES - 1:
                        jitter = delay * (0.5 + np.random.random())
                        logger.warning(
                            "Rate limited (attempt %d/%d), retrying in %.1fs",
                            attempt + 1, _MAX_RETRIES, jitter,
                        )
                        time.sleep(jitter)
                        delay = min(delay * 2, _MAX_DELAY)
                        continue
                raise

        raise RuntimeError(f"Gemini API failed after {_MAX_RETRIES} retries")

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        """Load cached embedding if available."""
        if not self._enable_cache:
            return None
        cache_path = self._cache_dir / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cache(self, key: str, emb: np.ndarray) -> None:
        """Save embedding to cache."""
        if not self._enable_cache:
            return
        cache_path = self._cache_dir / f"{key}.npy"
        np.save(cache_path, emb)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text into the unified 3072-d Gemini space.

        Unlike CLIP+CLAP, there is only ONE text encoder — the same
        embedding works for both image and audio comparison.

        Args:
            text: Input text string.

        Returns:
            L2-normalized 3072-d numpy array.
        """
        key = _cache_key("text", text)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        emb = self._call_api_with_retry(text)
        self._save_cache(key, emb)
        return emb

    def embed_image(self, path: str) -> np.ndarray:
        """
        Embed an image into the unified 3072-d Gemini space.

        Args:
            path: Path to image file (jpg, png, webp).

        Returns:
            L2-normalized 3072-d numpy array.
        """
        from google.genai import types

        path = str(path)
        key = _cache_key("image", path)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        image_bytes = Path(path).read_bytes()
        mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        emb = self._call_api_with_retry(part)
        self._save_cache(key, emb)
        return emb

    def embed_audio(self, path: str) -> np.ndarray:
        """
        Embed audio into the unified 3072-d Gemini space.

        Args:
            path: Path to audio file (wav, mp3, flac, ogg).

        Returns:
            L2-normalized 3072-d numpy array.
        """
        from google.genai import types

        path = str(path)
        key = _cache_key("audio", path)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        audio_bytes = Path(path).read_bytes()
        mime_type = mimetypes.guess_type(path)[0] or "audio/wav"
        part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        emb = self._call_api_with_retry(part)
        self._save_cache(key, emb)
        return emb

    def embed_all_scales(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        dims: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Embed at full dimension, then truncate+renormalize for each MRL scale.

        This is efficient: we make at most 3 API calls (text, image, audio)
        regardless of how many truncation dimensions are requested. The
        Matryoshka property means prefix truncation preserves semantic meaning.

        Args:
            text: Input text.
            image_path: Optional path to image.
            audio_path: Optional path to audio.
            dims: Truncation dimensions (default: MATRYOSHKA_DIMS).

        Returns:
            Dict mapping dim -> {"text": emb, "image": emb, "audio": emb}.
            Missing modalities have None values.
        """
        if dims is None:
            dims = MATRYOSHKA_DIMS

        # Embed at full resolution (3 API calls max)
        text_full = self.embed_text(text)
        image_full = self.embed_image(image_path) if image_path else None
        audio_full = self.embed_audio(audio_path) if audio_path else None

        result = {}
        for dim in sorted(dims):
            entry = {}
            entry["text"] = _l2_normalize(text_full[:dim])
            if image_full is not None:
                entry["image"] = _l2_normalize(image_full[:dim])
            else:
                entry["image"] = None
            if audio_full is not None:
                entry["audio"] = _l2_normalize(audio_full[:dim])
            else:
                entry["audio"] = None
            result[dim] = entry

        return result

    def embed_text_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts (sequential with caching)."""
        return [self.embed_text(t) for t in texts]

    def embed_image_batch(self, paths: List[str]) -> List[np.ndarray]:
        """Embed a batch of images (sequential with caching)."""
        return [self.embed_image(p) for p in paths]

    def embed_audio_batch(self, paths: List[str]) -> List[np.ndarray]:
        """Embed a batch of audio files (sequential with caching)."""
        return [self.embed_audio(p) for p in paths]
