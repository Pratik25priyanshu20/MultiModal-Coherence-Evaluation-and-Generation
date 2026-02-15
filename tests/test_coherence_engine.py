"""
Tests for MSCI (Multimodal Semantic Coherence Index) computation logic.

We do NOT import CoherenceEngine directly because it would load heavy ML
models (CLIP/CLAP).  Instead we reimplement the MSCI formula and test the
scoring math in isolation.

MSCI formula (from src/coherence/coherence_engine.py):
    weights = {"st_i": 0.45, "st_a": 0.45, "si_a": 0.10}
    available = {k: v for k, v in scores.items() if v is not None}
    total_weight = sum(weights[k] for k in available)
    msci = sum(score * weights[k] for k, score in available.items()) / total_weight
"""

from __future__ import annotations

from typing import Dict, Optional

import pytest


# ---------------------------------------------------------------------------
# Reimplementation of MSCI formula for isolated testing
# ---------------------------------------------------------------------------

MSCI_WEIGHTS: Dict[str, float] = {
    "st_i": 0.45,
    "st_a": 0.45,
    "si_a": 0.10,
}


def compute_msci(
    st_i: Optional[float] = None,
    st_a: Optional[float] = None,
    si_a: Optional[float] = None,
) -> Optional[float]:
    """
    Compute MSCI exactly as CoherenceEngine does,
    without importing it (avoids loading CLIP/CLAP).
    """
    scores: Dict[str, Optional[float]] = {
        "st_i": st_i,
        "st_a": st_a,
        "si_a": si_a,
    }
    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        return None

    total_weight = sum(MSCI_WEIGHTS[k] for k in available)
    msci = sum(v * MSCI_WEIGHTS[k] for k, v in available.items()) / total_weight
    return msci


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMSCIComputation:
    """Tests for the MSCI scoring formula."""

    def test_msci_both_st_i_and_st_a(self):
        """
        With st_i=0.8, st_a=0.6, si_a=None (cross-space guard):
          total_weight = 0.45 + 0.45 = 0.90
          msci = (0.8*0.45 + 0.6*0.45) / 0.90
               = (0.36 + 0.27) / 0.90
               = 0.63 / 0.90
               = 0.70
        """
        msci = compute_msci(st_i=0.8, st_a=0.6, si_a=None)
        assert msci is not None
        assert pytest.approx(msci, abs=1e-6) == 0.7

    def test_msci_equal_weights_when_both_available(self):
        """st_i and st_a have equal weight (0.45 each), so if both are the
        same value the MSCI should equal that value."""
        msci = compute_msci(st_i=0.5, st_a=0.5, si_a=None)
        assert msci is not None
        assert pytest.approx(msci, abs=1e-6) == 0.5

    def test_msci_only_st_i(self):
        """
        With only st_i=0.8:
          total_weight = 0.45
          msci = 0.8*0.45 / 0.45 = 0.8
        """
        msci = compute_msci(st_i=0.8, st_a=None, si_a=None)
        assert msci is not None
        assert pytest.approx(msci, abs=1e-6) == 0.8

    def test_msci_only_st_a(self):
        """
        With only st_a=0.6:
          total_weight = 0.45
          msci = 0.6*0.45 / 0.45 = 0.6
        """
        msci = compute_msci(st_i=None, st_a=0.6, si_a=None)
        assert msci is not None
        assert pytest.approx(msci, abs=1e-6) == 0.6

    def test_msci_none_when_no_scores(self):
        """MSCI should be None when all component scores are None."""
        msci = compute_msci(st_i=None, st_a=None, si_a=None)
        assert msci is None

    def test_si_a_always_none_cross_space_guard(self):
        """
        In the production system, si_a is always None because CLIP image
        embeddings and CLAP audio embeddings live in different spaces.
        With si_a=None the weight renormalises to 0.45+0.45=0.90.
        Verify the formula behaves identically to (st_i+st_a)/2 in this case.
        """
        for st_i_val in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for st_a_val in [0.0, 0.3, 0.5, 0.8, 1.0]:
                msci = compute_msci(st_i=st_i_val, st_a=st_a_val, si_a=None)
                expected = (st_i_val + st_a_val) / 2.0
                assert msci is not None
                assert pytest.approx(msci, abs=1e-6) == expected

    def test_msci_all_three_scores(self):
        """
        When all three scores are provided (hypothetical trained bridge):
          st_i=0.8, st_a=0.6, si_a=0.4
          total_weight = 0.45 + 0.45 + 0.10 = 1.00
          msci = 0.8*0.45 + 0.6*0.45 + 0.4*0.10
               = 0.36 + 0.27 + 0.04
               = 0.67
        """
        msci = compute_msci(st_i=0.8, st_a=0.6, si_a=0.4)
        assert msci is not None
        assert pytest.approx(msci, abs=1e-6) == 0.67

    def test_msci_perfect_scores(self):
        """All scores 1.0 should yield MSCI = 1.0 regardless of weighting."""
        msci = compute_msci(st_i=1.0, st_a=1.0, si_a=1.0)
        assert pytest.approx(msci, abs=1e-6) == 1.0

        msci_partial = compute_msci(st_i=1.0, st_a=1.0, si_a=None)
        assert pytest.approx(msci_partial, abs=1e-6) == 1.0

    def test_msci_zero_scores(self):
        """All scores 0.0 should yield MSCI = 0.0."""
        msci = compute_msci(st_i=0.0, st_a=0.0, si_a=None)
        assert pytest.approx(msci, abs=1e-6) == 0.0
