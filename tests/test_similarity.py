"""
Tests for src/embeddings/similarity.py

Covers l2_normalize and cosine_similarity with edge cases including
identical, orthogonal, opposite, and zero vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.similarity import cosine_similarity, l2_normalize


class TestL2Normalize:
    """Tests for the l2_normalize helper."""

    def test_produces_unit_vector(self, random_vector_512: np.ndarray):
        """Normalised vector should have L2 norm very close to 1.0."""
        normed = l2_normalize(random_vector_512)
        assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-5)

    def test_unit_vector_unchanged(self, unit_vector_512: np.ndarray):
        """Already-unit vector should stay (approximately) the same."""
        normed = l2_normalize(unit_vector_512)
        assert np.allclose(normed, unit_vector_512, atol=1e-6)

    def test_output_dtype_is_float32(self):
        """Output should always be float32 regardless of input dtype."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        normed = l2_normalize(vec)
        assert normed.dtype == np.float32

    def test_direction_preserved(self, rng: np.random.Generator):
        """Normalisation should not flip the sign of any component."""
        vec = rng.standard_normal(128).astype(np.float32)
        normed = l2_normalize(vec)
        # Signs should match (allowing for tiny eps perturbation on near-zero)
        signs_match = np.sign(vec) == np.sign(normed)
        near_zero = np.abs(vec) < 1e-6
        assert np.all(signs_match | near_zero)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors_equal_one(self, random_vector_512: np.ndarray):
        """cos(v, v) == 1.0 for any non-zero vector."""
        sim = cosine_similarity(random_vector_512, random_vector_512)
        assert pytest.approx(sim, abs=1e-5) == 1.0

    def test_orthogonal_vectors_equal_zero(self):
        """cos(e1, e2) == 0.0 for orthogonal basis vectors."""
        e1 = np.zeros(512, dtype=np.float32)
        e2 = np.zeros(512, dtype=np.float32)
        e1[0] = 1.0
        e2[1] = 1.0
        sim = cosine_similarity(e1, e2)
        assert pytest.approx(sim, abs=1e-5) == 0.0

    def test_opposite_vectors_equal_neg_one(self, random_vector_512: np.ndarray):
        """cos(v, -v) == -1.0 for any non-zero vector."""
        sim = cosine_similarity(random_vector_512, -random_vector_512)
        assert pytest.approx(sim, abs=1e-5) == -1.0

    def test_zero_vector_handled_gracefully(self, zero_vector_512: np.ndarray, unit_vector_512: np.ndarray):
        """
        Zero vector should not cause division by zero.
        The eps in l2_normalize prevents NaN; the result should be
        a finite number close to 0.
        """
        sim = cosine_similarity(zero_vector_512, unit_vector_512)
        assert np.isfinite(sim)
        # Zero vector normalized is ~0 everywhere, so dot with anything is ~0
        assert pytest.approx(sim, abs=1e-3) == 0.0

    def test_result_in_valid_range(self, rng: np.random.Generator):
        """cosine_similarity should always return a value in [-1.0, 1.0]."""
        for _ in range(50):
            a = rng.standard_normal(512).astype(np.float32)
            b = rng.standard_normal(512).astype(np.float32)
            sim = cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0

    def test_symmetry(self, rng: np.random.Generator):
        """cos(a, b) == cos(b, a)."""
        a = rng.standard_normal(256).astype(np.float32)
        b = rng.standard_normal(256).astype(np.float32)
        assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == cosine_similarity(b, a)
