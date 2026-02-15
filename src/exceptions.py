"""
Custom exception hierarchy for MultiModal Coherence AI.

Provides structured error types so callers can distinguish between
retrieval failures, generation failures, and validation errors
instead of catching bare Exception everywhere.
"""


class MultiModalError(Exception):
    """Base exception for all project-specific errors."""


class RetrievalError(MultiModalError):
    """Raised when image or audio retrieval fails.

    Attributes:
        query: The query text that failed retrieval.
        modality: 'image' or 'audio'.
        best_similarity: The best similarity score found (if any).
    """

    def __init__(self, message: str, *, query: str = "", modality: str = "", best_similarity: float | None = None):
        super().__init__(message)
        self.query = query
        self.modality = modality
        self.best_similarity = best_similarity


class GenerationError(MultiModalError):
    """Raised when text, image, or audio generation fails.

    Attributes:
        modality: 'text', 'image', or 'audio'.
        backend: The backend that failed (e.g., 'ollama', 'audioldm', 'fallback_ambient').
    """

    def __init__(self, message: str, *, modality: str = "", backend: str = ""):
        super().__init__(message)
        self.modality = modality
        self.backend = backend


class ValidationError(MultiModalError):
    """Raised when input validation or conditioning checks fail.

    Attributes:
        field: The field or parameter that failed validation.
    """

    def __init__(self, message: str, *, field: str = ""):
        super().__init__(message)
        self.field = field


class EmbeddingError(MultiModalError):
    """Raised when embedding computation fails.

    Attributes:
        modality: The modality that failed ('text', 'image', 'audio').
        model: The model that failed (e.g., 'clip', 'clap').
    """

    def __init__(self, message: str, *, modality: str = "", model: str = ""):
        super().__init__(message)
        self.modality = modality
        self.model = model


class IndexError_(MultiModalError):
    """Raised when an embedding index is missing, empty, or corrupt."""

    def __init__(self, message: str, *, index_path: str = ""):
        super().__init__(message)
        self.index_path = index_path


class PlanningError(MultiModalError):
    """Raised when the semantic planner fails to produce a valid plan."""
