"""
Common fixtures for MultiModal Coherence AI test suite.

Provides reusable numpy vectors, evaluation objects, and helper utilities
so individual test modules stay focused on their assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.human_eval_schema import (
    EvaluationSample,
    EvaluationSession,
    HumanEvaluation,
)


# ---------------------------------------------------------------------------
# Numpy vector fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_vector_512() -> np.ndarray:
    """A deterministic 512-d unit vector (first basis vector)."""
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0
    return vec


@pytest.fixture
def random_vector_512(rng: np.random.Generator) -> np.ndarray:
    """A random 512-d vector (not normalised)."""
    return rng.standard_normal(512).astype(np.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def zero_vector_512() -> np.ndarray:
    """All-zeros 512-d vector."""
    return np.zeros(512, dtype=np.float32)


# ---------------------------------------------------------------------------
# Human-eval schema fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_evaluation() -> HumanEvaluation:
    """A valid HumanEvaluation with middle-of-range scores."""
    return HumanEvaluation(
        sample_id="test-001",
        evaluator_id="evaluator-A",
        text_image_coherence=4,
        text_audio_coherence=3,
        image_audio_coherence=2,
        overall_coherence=3,
        confidence=4,
        notes="Test fixture evaluation",
    )


@pytest.fixture
def sample_evaluation_sample() -> EvaluationSample:
    """A valid EvaluationSample for round-trip tests."""
    return EvaluationSample(
        sample_id="sample-001",
        text_content="A quiet forest at dawn with birdsong",
        image_path="/data/images/forest_dawn.jpg",
        audio_path="/data/audio/birdsong_morning.wav",
        condition="planner_baseline",
        mode="planner",
        perturbation="baseline",
        msci_score=0.72,
        run_id="run-20240601",
        original_prompt="A quiet forest at dawn with birdsong",
    )


@pytest.fixture
def evaluation_session(sample_evaluation_sample: EvaluationSample) -> EvaluationSession:
    """An EvaluationSession with 5 identical samples, no evaluations yet."""
    samples = []
    for i in range(5):
        s = EvaluationSample(
            sample_id=f"sample-{i:03d}",
            text_content=sample_evaluation_sample.text_content,
            image_path=sample_evaluation_sample.image_path,
            audio_path=sample_evaluation_sample.audio_path,
        )
        samples.append(s)

    return EvaluationSession(
        session_id="session-test",
        evaluator_id="evaluator-A",
        samples=samples,
    )
