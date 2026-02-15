"""
Tests for src/evaluation/human_eval_schema.py

Covers HumanEvaluation validation, scoring methods, EvaluationSession
progress tracking, and EvaluationSample round-trip serialisation.
"""

from __future__ import annotations

import pytest

from src.evaluation.human_eval_schema import (
    EvaluationSample,
    EvaluationSession,
    HumanEvaluation,
)


class TestHumanEvaluationValidation:
    """Tests for HumanEvaluation __post_init__ validation."""

    def test_rejects_rating_of_zero(self):
        """Rating of 0 is below Likert scale (1-5) and should be rejected."""
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            HumanEvaluation(
                sample_id="bad-001",
                evaluator_id="evaluator-A",
                text_image_coherence=0,
                text_audio_coherence=3,
                image_audio_coherence=3,
                overall_coherence=3,
            )

    def test_rejects_rating_of_six(self):
        """Rating of 6 is above Likert scale (1-5) and should be rejected."""
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            HumanEvaluation(
                sample_id="bad-002",
                evaluator_id="evaluator-A",
                text_image_coherence=3,
                text_audio_coherence=6,
                image_audio_coherence=3,
                overall_coherence=3,
            )

    def test_rejects_negative_rating(self):
        """Negative rating should be rejected."""
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            HumanEvaluation(
                sample_id="bad-003",
                evaluator_id="evaluator-A",
                text_image_coherence=3,
                text_audio_coherence=3,
                image_audio_coherence=-1,
                overall_coherence=3,
            )

    def test_rejects_invalid_confidence(self):
        """Confidence field also validated to 1-5 range."""
        with pytest.raises(ValueError, match="must be between 1 and 5"):
            HumanEvaluation(
                sample_id="bad-004",
                evaluator_id="evaluator-A",
                text_image_coherence=3,
                text_audio_coherence=3,
                image_audio_coherence=3,
                overall_coherence=3,
                confidence=0,
            )

    def test_accepts_all_boundary_values(self):
        """Ratings of exactly 1 and 5 should be accepted."""
        ev = HumanEvaluation(
            sample_id="ok-001",
            evaluator_id="evaluator-A",
            text_image_coherence=1,
            text_audio_coherence=5,
            image_audio_coherence=1,
            overall_coherence=5,
            confidence=1,
        )
        assert ev.text_image_coherence == 1
        assert ev.text_audio_coherence == 5


class TestHumanEvaluationScoring:
    """Tests for scoring methods on HumanEvaluation."""

    def test_mean_pairwise_score(self, sample_evaluation: HumanEvaluation):
        """
        sample_evaluation has text_image=4, text_audio=3, image_audio=2.
        mean = (4 + 3 + 2) / 3.0 = 3.0
        """
        assert pytest.approx(sample_evaluation.mean_pairwise_score(), abs=1e-6) == 3.0

    def test_mean_pairwise_score_all_same(self):
        """If all pairwise scores are equal, mean equals that value."""
        ev = HumanEvaluation(
            sample_id="x", evaluator_id="e",
            text_image_coherence=4,
            text_audio_coherence=4,
            image_audio_coherence=4,
            overall_coherence=4,
        )
        assert pytest.approx(ev.mean_pairwise_score()) == 4.0

    def test_weighted_score(self, sample_evaluation: HumanEvaluation):
        """
        sample_evaluation: ti=4, ta=3, ia=2
        weighted = (0.45*4 + 0.45*3 + 0.10*2) / (1.0 * 5)
                 = (1.80 + 1.35 + 0.20) / 5.0
                 = 3.35 / 5.0
                 = 0.67
        """
        ws = sample_evaluation.weighted_score()
        assert pytest.approx(ws, abs=1e-6) == 0.67

    def test_weighted_score_perfect(self):
        """All 5s should give weighted_score = 1.0."""
        ev = HumanEvaluation(
            sample_id="x", evaluator_id="e",
            text_image_coherence=5,
            text_audio_coherence=5,
            image_audio_coherence=5,
            overall_coherence=5,
        )
        assert pytest.approx(ev.weighted_score(), abs=1e-6) == 1.0

    def test_weighted_score_minimum(self):
        """All 1s should give weighted_score = 1/5 = 0.2."""
        ev = HumanEvaluation(
            sample_id="x", evaluator_id="e",
            text_image_coherence=1,
            text_audio_coherence=1,
            image_audio_coherence=1,
            overall_coherence=1,
        )
        assert pytest.approx(ev.weighted_score(), abs=1e-6) == 0.2

    def test_weighted_score_custom_weights(self):
        """weighted_score with custom weights should use those weights."""
        ev = HumanEvaluation(
            sample_id="x", evaluator_id="e",
            text_image_coherence=5,
            text_audio_coherence=1,
            image_audio_coherence=1,
            overall_coherence=3,
        )
        # w_ti=1.0, w_ta=0.0, w_ia=0.0 -> only ti matters -> 5 / (1.0 * 5) = 1.0
        ws = ev.weighted_score(w_ti=1.0, w_ta=0.0, w_ia=0.0)
        assert pytest.approx(ws, abs=1e-6) == 1.0


class TestEvaluationSession:
    """Tests for EvaluationSession progress and state tracking."""

    def test_initial_progress_is_zero(self, evaluation_session: EvaluationSession):
        """Fresh session with no evaluations should be at 0%."""
        assert evaluation_session.progress == 0.0

    def test_progress_after_one_evaluation(self, evaluation_session: EvaluationSession):
        """One evaluation out of 5 samples = 20%."""
        ev = HumanEvaluation(
            sample_id="sample-000",
            evaluator_id="evaluator-A",
            text_image_coherence=3,
            text_audio_coherence=3,
            image_audio_coherence=3,
            overall_coherence=3,
        )
        evaluation_session.add_evaluation(ev)
        assert pytest.approx(evaluation_session.progress, abs=0.1) == 20.0

    def test_add_evaluation_advances_index(self, evaluation_session: EvaluationSession):
        """add_evaluation should increment current_index."""
        assert evaluation_session.current_index == 0
        ev = HumanEvaluation(
            sample_id="sample-000",
            evaluator_id="evaluator-A",
            text_image_coherence=3,
            text_audio_coherence=3,
            image_audio_coherence=3,
            overall_coherence=3,
        )
        evaluation_session.add_evaluation(ev)
        assert evaluation_session.current_index == 1

    def test_add_evaluation_stores_evaluation(self, evaluation_session: EvaluationSession):
        """add_evaluation should append to the evaluations list."""
        ev = HumanEvaluation(
            sample_id="sample-000",
            evaluator_id="evaluator-A",
            text_image_coherence=4,
            text_audio_coherence=4,
            image_audio_coherence=4,
            overall_coherence=4,
        )
        evaluation_session.add_evaluation(ev)
        assert len(evaluation_session.evaluations) == 1
        assert evaluation_session.evaluations[0].sample_id == "sample-000"

    def test_session_complete_after_all_evaluations(self, evaluation_session: EvaluationSession):
        """Session should be marked complete after evaluating all 5 samples."""
        for i in range(5):
            ev = HumanEvaluation(
                sample_id=f"sample-{i:03d}",
                evaluator_id="evaluator-A",
                text_image_coherence=3,
                text_audio_coherence=3,
                image_audio_coherence=3,
                overall_coherence=3,
            )
            evaluation_session.add_evaluation(ev)

        assert evaluation_session.is_complete is True
        assert evaluation_session.completed_at is not None
        assert pytest.approx(evaluation_session.progress, abs=0.1) == 100.0

    def test_progress_empty_session(self):
        """Session with no samples should return 0% progress."""
        session = EvaluationSession(
            session_id="empty",
            evaluator_id="e",
            samples=[],
        )
        assert session.progress == 0.0

    def test_add_evaluation_sets_session_id(self, evaluation_session: EvaluationSession):
        """add_evaluation should set the evaluation's session_id."""
        ev = HumanEvaluation(
            sample_id="sample-000",
            evaluator_id="evaluator-A",
            text_image_coherence=3,
            text_audio_coherence=3,
            image_audio_coherence=3,
            overall_coherence=3,
        )
        evaluation_session.add_evaluation(ev)
        assert ev.session_id == "session-test"


class TestEvaluationSampleSerialization:
    """Tests for EvaluationSample round-trip serialisation."""

    def test_to_dict_from_dict_round_trip(self, sample_evaluation_sample: EvaluationSample):
        """to_dict() -> from_dict() should produce an identical object."""
        d = sample_evaluation_sample.to_dict()
        restored = EvaluationSample.from_dict(d)

        assert restored.sample_id == sample_evaluation_sample.sample_id
        assert restored.text_content == sample_evaluation_sample.text_content
        assert restored.image_path == sample_evaluation_sample.image_path
        assert restored.audio_path == sample_evaluation_sample.audio_path
        assert restored.condition == sample_evaluation_sample.condition
        assert restored.mode == sample_evaluation_sample.mode
        assert restored.perturbation == sample_evaluation_sample.perturbation
        assert restored.msci_score == sample_evaluation_sample.msci_score
        assert restored.run_id == sample_evaluation_sample.run_id
        assert restored.original_prompt == sample_evaluation_sample.original_prompt

    def test_to_dict_contains_all_fields(self, sample_evaluation_sample: EvaluationSample):
        """to_dict should include every field."""
        d = sample_evaluation_sample.to_dict()
        expected_keys = {
            "sample_id", "text_content", "image_path", "audio_path",
            "condition", "mode", "perturbation", "msci_score",
            "run_id", "original_prompt",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_with_defaults(self):
        """from_dict with only required fields should use defaults for optional ones."""
        minimal = {
            "sample_id": "min-001",
            "text_content": "test",
            "image_path": "/img.jpg",
            "audio_path": "/aud.wav",
        }
        sample = EvaluationSample.from_dict(minimal)
        assert sample.condition == ""
        assert sample.msci_score is None

    def test_human_evaluation_round_trip(self, sample_evaluation: HumanEvaluation):
        """HumanEvaluation to_dict -> from_dict should preserve all fields."""
        d = sample_evaluation.to_dict()
        restored = HumanEvaluation.from_dict(d)
        assert restored.sample_id == sample_evaluation.sample_id
        assert restored.text_image_coherence == sample_evaluation.text_image_coherence
        assert restored.text_audio_coherence == sample_evaluation.text_audio_coherence
        assert restored.image_audio_coherence == sample_evaluation.image_audio_coherence
        assert restored.overall_coherence == sample_evaluation.overall_coherence
        assert restored.confidence == sample_evaluation.confidence
        assert restored.notes == sample_evaluation.notes
