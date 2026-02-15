"""
Caching utilities for embeddings and generation results.

Implements:
- Embedding cache (text, image, audio)
- Generation result cache
- Content-based caching with similarity matching
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from joblib import Memory

from src.embeddings.similarity import cosine_similarity


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""

    def __init__(self, cache_dir: str = ".cache/embeddings", similarity_threshold: float = 0.99):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        
        # In-memory cache for faster access
        self._memory_cache: Dict[str, np.ndarray] = {}
        
        # Persistent cache using joblib Memory
        self._memory = Memory(location=str(self.cache_dir / "joblib"), verbose=0)

    def _get_key(self, content: str, modality: str) -> str:
        """Generate cache key from content."""
        content_hash = hashlib.sha256(f"{modality}:{content}".encode()).hexdigest()
        return f"{modality}_{content_hash[:16]}"

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.npy"

    def get(self, content: str, modality: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding."""
        key = self._get_key(content, modality)
        
        # Check in-memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            embedding = np.load(cache_path)
            self._memory_cache[key] = embedding
            return embedding
        
        return None

    def set(self, content: str, modality: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._get_key(content, modality)
        cache_path = self._get_cache_path(key)
        
        # Store in memory
        self._memory_cache[key] = embedding
        
        # Store on disk
        np.save(cache_path, embedding)

    def get_similar(self, query_embedding: np.ndarray, modality: str) -> Optional[Tuple[str, np.ndarray, float]]:
        """Find similar cached embedding using cosine similarity."""
        # Load all cached embeddings for this modality
        pattern = f"{modality}_*.npy"
        cached_files = list(self.cache_dir.glob(pattern))
        
        best_match: Optional[Tuple[str, np.ndarray, float]] = None
        best_similarity = -1.0
        
        for cache_file in cached_files:
            try:
                cached_embedding = np.load(cache_file)
                similarity = cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    # Extract content from cache file metadata if available
                    content_id = cache_file.stem
                    best_match = (content_id, cached_embedding, similarity)
            except Exception:
                continue
        
        return best_match


class GenerationCache:
    """Cache for generation results (images, audio, text)."""

    def __init__(self, cache_dir: str = ".cache/generations"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._metadata_file = self.cache_dir / "metadata.json"
        
        # Load existing metadata
        self._metadata: Dict[str, Dict[str, Any]] = {}
        if self._metadata_file.exists():
            try:
                with self._metadata_file.open("r") as f:
                    self._metadata = json.load(f)
            except Exception:
                self._metadata = {}

    def _get_key(self, prompt: str, modality: str, generator_config: Optional[Dict] = None) -> str:
        """Generate cache key from prompt and generator config."""
        config_str = json.dumps(generator_config or {}, sort_keys=True)
        content = f"{modality}:{prompt}:{config_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, prompt: str, modality: str, generator_config: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached generation result."""
        key = self._get_key(prompt, modality, generator_config)
        
        # Check in-memory cache
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check metadata and disk
        if key in self._metadata:
            entry = self._metadata[key]
            result_path = self.cache_dir / entry["file_path"]
            
            if result_path.exists():
                result = {
                    "file_path": str(result_path),
                    "modality": entry["modality"],
                    "prompt": entry["prompt"],
                    "metadata": entry.get("metadata", {}),
                }
                self._memory_cache[key] = result
                return result
        
        return None

    def set(
        self,
        prompt: str,
        modality: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict] = None,
    ) -> str:
        """Store generation result in cache."""
        key = self._get_key(prompt, modality, generator_config)
        
        # Copy file to cache directory if it's not already there
        source_path = Path(file_path)
        if str(source_path.parent) != str(self.cache_dir):
            cache_file_path = self.cache_dir / f"{key}_{modality}{source_path.suffix}"
            import shutil
            shutil.copy2(source_path, cache_file_path)
            file_path = str(cache_file_path)
        else:
            file_path = str(source_path)
        
        entry = {
            "file_path": file_path,
            "modality": modality,
            "prompt": prompt,
            "metadata": metadata or {},
            "generator_config": generator_config or {},
        }
        
        # Update metadata
        self._metadata[key] = entry
        self._memory_cache[key] = {
            "file_path": file_path,
            "modality": modality,
            "prompt": prompt,
            "metadata": metadata or {},
        }
        
        # Save metadata
        with self._metadata_file.open("w") as f:
            json.dump(self._metadata, f, indent=2)
        
        return key

    def find_similar(self, prompt: str, modality: str, similarity_threshold: float = 0.85) -> list[Dict[str, Any]]:
        """Find similar cached results using prompt similarity (simple keyword matching)."""
        # Simple implementation: exact prompt match
        # Can be enhanced with semantic similarity using embeddings
        matches = []
        prompt_lower = prompt.lower()
        
        for entry in self._metadata.values():
            if entry["modality"] == modality:
                cached_prompt_lower = entry["prompt"].lower()
                # Simple overlap check (can be enhanced)
                words = set(prompt_lower.split())
                cached_words = set(cached_prompt_lower.split())
                overlap = len(words & cached_words) / max(len(words | cached_words), 1)
                
                if overlap >= similarity_threshold:
                    result_path = self.cache_dir / entry["file_path"]
                    if result_path.exists():
                        matches.append({
                            "file_path": str(result_path),
                            "prompt": entry["prompt"],
                            "similarity": overlap,
                            "metadata": entry.get("metadata", {}),
                        })
        
        return sorted(matches, key=lambda x: x["similarity"], reverse=True)


class ContentBasedCache:
    """Content-based cache that uses semantic similarity for matching."""

    def __init__(
        self,
        cache_dir: str = ".cache/content",
        embedding_cache: Optional[EmbeddingCache] = None,
        similarity_threshold: float = 0.90,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.embedding_cache = embedding_cache or EmbeddingCache()
        
        self._index_file = self.cache_dir / "content_index.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        
        if self._index_file.exists():
            try:
                with self._index_file.open("r") as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}

    def _get_content_key(self, content: str) -> str:
        """Generate key for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add(self, content: str, result: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> None:
        """Add content and result to cache."""
        key = self._get_content_key(content)
        
        # Store embedding if provided
        if embedding is not None:
            self.embedding_cache.set(content, "content", embedding)
        
        entry = {
            "content": content,
            "result": result,
            "embedding_key": self.embedding_cache._get_key(content, "content") if embedding is not None else None,
        }
        
        self._index[key] = entry
        
        # Save index
        with self._index_file.open("w") as f:
            json.dump(self._index, f, indent=2)

    def find_similar(self, query: str, query_embedding: Optional[np.ndarray] = None) -> list[Tuple[Dict[str, Any], float]]:
        """Find similar cached content using semantic similarity."""
        if query_embedding is None:
            # Try to get embedding from cache
            cached_emb = self.embedding_cache.get(query, "content")
            if cached_emb is None:
                return []
            query_embedding = cached_emb
        
        matches = []
        for key, entry in self._index.items():
            if entry.get("embedding_key"):
                cached_emb = self.embedding_cache.get(entry["content"], "content")
                if cached_emb is not None:
                    similarity = cosine_similarity(query_embedding, cached_emb)
                    if similarity >= self.similarity_threshold:
                        matches.append((entry["result"], similarity))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
