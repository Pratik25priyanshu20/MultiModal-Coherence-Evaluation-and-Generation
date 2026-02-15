"""
Tests for src/exceptions.py — custom exception hierarchy.

Verifies that all custom exceptions inherit from MultiModalError,
store their extra attributes, and can be caught with a single
except MultiModalError clause.
"""

from __future__ import annotations

import pytest

from src.exceptions import (
    EmbeddingError,
    GenerationError,
    IndexError_,
    MultiModalError,
    PlanningError,
    RetrievalError,
    ValidationError,
)


class TestExceptionHierarchy:
    """All custom exceptions should be subclasses of MultiModalError."""

    @pytest.mark.parametrize("exc_cls", [
        RetrievalError,
        GenerationError,
        ValidationError,
        EmbeddingError,
        IndexError_,
        PlanningError,
    ])
    def test_is_multimodal_error(self, exc_cls):
        """Every project exception should be a MultiModalError subclass."""
        assert issubclass(exc_cls, MultiModalError)

    def test_catch_all_as_multimodal_error(self):
        """All custom exceptions should be catchable with except MultiModalError."""
        exceptions_to_test = [
            RetrievalError("retrieval failed", query="test", modality="image"),
            GenerationError("gen failed", modality="audio", backend="audioldm"),
            ValidationError("bad input", field="prompt"),
            EmbeddingError("embed failed", modality="text", model="clip"),
            IndexError_("index missing", index_path="/data/index.npz"),
            PlanningError("plan failed"),
        ]
        for exc in exceptions_to_test:
            with pytest.raises(MultiModalError):
                raise exc


class TestRetrievalError:
    """Tests for RetrievalError attributes."""

    def test_stores_query_and_modality(self):
        """RetrievalError should store query, modality, and best_similarity."""
        err = RetrievalError(
            "No image found",
            query="sunset over mountains",
            modality="image",
            best_similarity=0.15,
        )
        assert err.query == "sunset over mountains"
        assert err.modality == "image"
        assert err.best_similarity == 0.15
        assert str(err) == "No image found"

    def test_default_attributes(self):
        """Default attribute values should be empty/None."""
        err = RetrievalError("fail")
        assert err.query == ""
        assert err.modality == ""
        assert err.best_similarity is None

    def test_is_multimodal_error(self):
        """RetrievalError should be an instance of MultiModalError."""
        err = RetrievalError("fail")
        assert isinstance(err, MultiModalError)


class TestGenerationError:
    """Tests for GenerationError attributes."""

    def test_stores_modality_and_backend(self):
        """GenerationError should store modality and backend."""
        err = GenerationError(
            "Audio generation failed",
            modality="audio",
            backend="audioldm",
        )
        assert err.modality == "audio"
        assert err.backend == "audioldm"
        assert str(err) == "Audio generation failed"

    def test_default_attributes(self):
        """Default attribute values should be empty strings."""
        err = GenerationError("fail")
        assert err.modality == ""
        assert err.backend == ""


class TestValidationError:
    """Tests for ValidationError attributes."""

    def test_stores_field(self):
        """ValidationError should store the field name."""
        err = ValidationError(
            "Prompt cannot be empty",
            field="prompt",
        )
        assert err.field == "prompt"
        assert str(err) == "Prompt cannot be empty"

    def test_default_field(self):
        """Default field should be empty string."""
        err = ValidationError("fail")
        assert err.field == ""


class TestEmbeddingError:
    """Tests for EmbeddingError attributes."""

    def test_stores_modality_and_model(self):
        """EmbeddingError should store modality and model."""
        err = EmbeddingError(
            "CLIP inference failed",
            modality="image",
            model="clip",
        )
        assert err.modality == "image"
        assert err.model == "clip"


class TestIndexError_:
    """Tests for IndexError_ attributes."""

    def test_stores_index_path(self):
        """IndexError_ should store the index_path."""
        err = IndexError_(
            "Index file not found",
            index_path="/data/embeddings/image_index.npz",
        )
        assert err.index_path == "/data/embeddings/image_index.npz"


class TestPlanningError:
    """Tests for PlanningError."""

    def test_is_multimodal_error(self):
        """PlanningError should be a MultiModalError."""
        err = PlanningError("Planner returned empty plan")
        assert isinstance(err, MultiModalError)
        assert str(err) == "Planner returned empty plan"
