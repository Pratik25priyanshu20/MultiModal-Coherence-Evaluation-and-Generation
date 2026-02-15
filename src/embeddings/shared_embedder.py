"""
Thread-safe singleton for AlignedEmbedder.

When running experiments with --parallel N, each thread would normally
create its own AlignedEmbedder, loading CLIP + CLAP models redundantly.
This module provides a shared, thread-safe instance that all threads use.

Usage:
    from src.embeddings.shared_embedder import get_shared_embedder
    embedder = get_shared_embedder()  # Returns the same instance every time
"""

from __future__ import annotations

import threading
from typing import Optional

from src.embeddings.aligned_embeddings import AlignedEmbedder

_lock = threading.Lock()
_instance: Optional[AlignedEmbedder] = None


def get_shared_embedder(
    target_dim: int = 512,
    enable_cache: bool = True,
    cache_dir: str = ".cache/embeddings",
) -> AlignedEmbedder:
    """
    Get or create the shared AlignedEmbedder instance.

    Thread-safe: uses double-checked locking. The first call creates the
    instance; subsequent calls return the same object immediately.

    The underlying CLIP/CLAP models are read-only at inference time, so
    sharing across threads is safe. The EmbeddingCache uses file-based
    storage which handles concurrent access.

    Args:
        target_dim: Embedding dimension (only used on first call)
        enable_cache: Whether to enable disk caching (only used on first call)
        cache_dir: Cache directory path (only used on first call)

    Returns:
        Shared AlignedEmbedder instance
    """
    global _instance
    if _instance is not None:
        return _instance

    with _lock:
        # Double-checked locking
        if _instance is not None:
            return _instance
        _instance = AlignedEmbedder(
            target_dim=target_dim,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )
        return _instance


def reset_shared_embedder() -> None:
    """
    Reset the singleton (for testing or reconfiguration).

    Not thread-safe — call only when no other threads are using the embedder.
    """
    global _instance
    with _lock:
        _instance = None
