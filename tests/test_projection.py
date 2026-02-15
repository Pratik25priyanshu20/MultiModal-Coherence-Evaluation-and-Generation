"""
Tests for src/embeddings/projection.py — ProjectionHead.

Key invariant: when in_dim == out_dim, ProjectionHead uses identity
pass-through to preserve pre-trained alignment (CLIP / CLAP).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.projection import ProjectionHead


class TestProjectionHeadIdentity:
    """Tests for identity (pass-through) projection when in_dim == out_dim."""

    def test_identity_flag_when_same_dim(self):
        """512 -> 512 should use identity (no linear layer)."""
        head = ProjectionHead(in_dim=512, out_dim=512)
        assert head._identity is True
        assert head.layer is None

    def test_identity_preserves_values_exactly(self, random_vector_512: np.ndarray):
        """Identity projection must return the exact same values (as float32)."""
        head = ProjectionHead(in_dim=512, out_dim=512)
        projected = head.project(random_vector_512)
        assert projected.dtype == np.float32
        np.testing.assert_array_equal(projected, random_vector_512.astype(np.float32))

    def test_identity_preserves_zeros(self, zero_vector_512: np.ndarray):
        """Identity projection of zeros should be zeros."""
        head = ProjectionHead(in_dim=512, out_dim=512)
        projected = head.project(zero_vector_512)
        np.testing.assert_array_equal(projected, np.zeros(512, dtype=np.float32))

    def test_identity_with_various_matching_dims(self):
        """Identity should work for any matching in/out dim, not just 512."""
        for dim in [64, 128, 256, 768, 1024]:
            head = ProjectionHead(in_dim=dim, out_dim=dim)
            assert head._identity is True
            assert head.layer is None
            vec = np.random.randn(dim).astype(np.float32)
            projected = head.project(vec)
            np.testing.assert_array_equal(projected, vec)


class TestProjectionHeadLinear:
    """Tests for non-identity (linear) projection when in_dim != out_dim."""

    def test_non_identity_creates_linear_layer(self):
        """768 -> 512 should use a linear layer, not identity."""
        head = ProjectionHead(in_dim=768, out_dim=512)
        assert head._identity is False
        assert head.layer is not None

    def test_output_shape_matches_out_dim(self):
        """Output dimension should match the specified out_dim."""
        head = ProjectionHead(in_dim=768, out_dim=512)
        vec = np.random.randn(768).astype(np.float32)
        projected = head.project(vec)
        assert projected.shape == (512,)

    def test_output_dtype_is_float32(self):
        """Projected output should be float32."""
        head = ProjectionHead(in_dim=256, out_dim=512)
        vec = np.random.randn(256).astype(np.float64)
        projected = head.project(vec)
        assert projected.dtype == np.float32

    def test_linear_layer_has_no_bias(self):
        """Linear projection should have bias=False as per source."""
        head = ProjectionHead(in_dim=768, out_dim=512)
        assert head.layer.bias is None

    def test_linear_layer_in_eval_mode(self):
        """Linear layer should be in eval mode (no dropout effects etc.)."""
        head = ProjectionHead(in_dim=768, out_dim=512)
        assert not head.layer.training
