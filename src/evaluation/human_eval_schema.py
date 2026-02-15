"""
Human Evaluation Schema for Multimodal Coherence Assessment

This module defines the data structures for collecting and storing
human judgments of multimodal coherence. Designed for single-rater
evaluation with bias mitigation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


class LikertScale(int, Enum):
    """5-point Likert scale for coherence ratings."""
    COMPLETELY_UNRELATED = 1
    VAGUE_CONNECTION = 2
    PARTIAL_MATCH = 3
    MOSTLY_ALIGNED = 4
    STRONG_ALIGNMENT = 5


@dataclass
class CoherenceRubric:
    """
    Structured rubric for consistent human evaluation.

    Each rating criterion has explicit descriptions for each Likert level
    to reduce subjectivity and improve intra-rater reliability.
    """

    text_image_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "Completely unrelated: Image has no semantic connection to text",
        2: "Vague thematic connection only: General theme matches but specifics differ",
        3: "Partial match: Some elements align, others clearly don't",
        4: "Mostly aligned: Most elements match, minor discrepancies",
        5: "Strong semantic alignment: Image accurately represents text content",
    })

    text_audio_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "Completely unrelated: Audio has no connection to described scene",
        2: "Vague connection: General mood might match but sounds don't fit",
        3: "Partial match: Some sounds fit the scene, others are mismatched",
        4: "Mostly aligned: Audio largely fits the scene with minor issues",
        5: "Strong alignment: Audio perfectly complements the described scene",
    })

    image_audio_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "Completely unrelated: Audio doesn't match what's shown in image",
        2: "Vague connection: Mood might match but sounds don't fit visuals",
        3: "Partial match: Some sounds plausible for image, others not",
        4: "Mostly aligned: Audio largely fits the visual scene",
        5: "Strong alignment: Audio sounds exactly right for the visual",
    })

    overall_rubric: Dict[int, str] = field(default_factory=lambda: {
        1: "No coherence: Modalities feel randomly combined",
        2: "Weak coherence: Some connection but feels disjointed",
        3: "Moderate coherence: Works together with noticeable gaps",
        4: "Good coherence: Modalities complement each other well",
        5: "Excellent coherence: Unified, immersive multimodal experience",
    })


@dataclass
class HumanEvaluation:
    """
    A single human evaluation of a multimodal sample.

    Attributes:
        sample_id: Unique identifier for the evaluated sample
        evaluator_id: Identifier for the human evaluator
        text_image_coherence: Rating of text-image alignment (1-5)
        text_audio_coherence: Rating of text-audio alignment (1-5)
        image_audio_coherence: Rating of image-audio alignment (1-5)
        overall_coherence: Holistic coherence rating (1-5)
        confidence: Self-reported confidence in ratings (1-5)
        notes: Optional free-text observations
        timestamp: When the evaluation was completed
        session_id: Evaluation session identifier (for tracking re-ratings)
        is_rerating: Whether this is a second pass for reliability check
    """
    sample_id: str
    evaluator_id: str
    text_image_coherence: int
    text_audio_coherence: int
    image_audio_coherence: int
    overall_coherence: int
    confidence: int = 3
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = ""
    is_rerating: bool = False

    def __post_init__(self):
        """Validate ratings are within Likert scale bounds."""
        for attr in ['text_image_coherence', 'text_audio_coherence',
                     'image_audio_coherence', 'overall_coherence', 'confidence']:
            value = getattr(self, attr)
            if not 1 <= value <= 5:
                raise ValueError(f"{attr} must be between 1 and 5, got {value}")

    def mean_pairwise_score(self) -> float:
        """Average of the three pairwise coherence ratings."""
        return (self.text_image_coherence + self.text_audio_coherence +
                self.image_audio_coherence) / 3.0

    def weighted_score(self, w_ti: float = 0.45, w_ta: float = 0.45,
                       w_ia: float = 0.10) -> float:
        """
        Weighted average matching MSCI weights for direct comparison.

        Default weights: text-image=0.45, text-audio=0.45, image-audio=0.10
        """
        total = w_ti + w_ta + w_ia
        return (w_ti * self.text_image_coherence +
                w_ta * self.text_audio_coherence +
                w_ia * self.image_audio_coherence) / (total * 5)  # Normalize to 0-1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "evaluator_id": self.evaluator_id,
            "text_image_coherence": self.text_image_coherence,
            "text_audio_coherence": self.text_audio_coherence,
            "image_audio_coherence": self.image_audio_coherence,
            "overall_coherence": self.overall_coherence,
            "confidence": self.confidence,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "is_rerating": self.is_rerating,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanEvaluation":
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class EvaluationSample:
    """
    A sample prepared for human evaluation.

    Contains all information needed to present a sample to the evaluator
    while keeping condition labels hidden for blind evaluation.
    """
    sample_id: str
    text_content: str
    image_path: str
    audio_path: str
    # Hidden metadata (not shown to evaluator during blind eval)
    condition: str = ""  # e.g., "planner_baseline", "direct_wrong_image"
    mode: str = ""  # "planner" or "direct"
    perturbation: str = ""  # "baseline", "wrong_image", "wrong_audio"
    msci_score: Optional[float] = None
    run_id: str = ""
    original_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "text_content": self.text_content,
            "image_path": self.image_path,
            "audio_path": self.audio_path,
            "condition": self.condition,
            "mode": self.mode,
            "perturbation": self.perturbation,
            "msci_score": self.msci_score,
            "run_id": self.run_id,
            "original_prompt": self.original_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSample":
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class EvaluationSession:
    """
    Tracks a complete human evaluation session.

    Manages the list of samples to evaluate, collects evaluations,
    and supports saving/loading for interrupted sessions.
    """
    session_id: str
    evaluator_id: str
    samples: List[EvaluationSample]
    evaluations: List[HumanEvaluation] = field(default_factory=list)
    current_index: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    # For intra-rater reliability
    rerating_sample_ids: List[str] = field(default_factory=list)

    @property
    def progress(self) -> float:
        """Completion percentage."""
        if not self.samples:
            return 0.0
        return len(self.evaluations) / len(self.samples) * 100

    @property
    def is_complete(self) -> bool:
        """Whether all samples have been evaluated."""
        return len(self.evaluations) >= len(self.samples)

    def get_current_sample(self) -> Optional[EvaluationSample]:
        """Get the next sample to evaluate."""
        if self.current_index < len(self.samples):
            return self.samples[self.current_index]
        return None

    def add_evaluation(self, evaluation: HumanEvaluation):
        """Add a completed evaluation and advance index."""
        evaluation.session_id = self.session_id
        self.evaluations.append(evaluation)
        self.current_index += 1

        if self.is_complete:
            self.completed_at = datetime.now().isoformat()

    def save(self, path: Path):
        """Save session state to JSON file."""
        data = {
            "session_id": self.session_id,
            "evaluator_id": self.evaluator_id,
            "samples": [s.to_dict() for s in self.samples],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "current_index": self.current_index,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "rerating_sample_ids": self.rerating_sample_ids,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "EvaluationSession":
        """Load session state from JSON file."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            session_id=data["session_id"],
            evaluator_id=data["evaluator_id"],
            samples=[EvaluationSample.from_dict(s) for s in data["samples"]],
            evaluations=[HumanEvaluation.from_dict(e) for e in data["evaluations"]],
            current_index=data["current_index"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            rerating_sample_ids=data.get("rerating_sample_ids", []),
        )


@dataclass
class ReliabilityMetrics:
    """
    Intra-rater reliability metrics for single-evaluator studies.

    Computes agreement between first and second ratings of the same
    samples to assess consistency.
    """
    kappa: float  # Cohen's kappa (self-agreement)
    percent_agreement: float  # Simple % exact matches
    weighted_kappa: float  # Quadratic weighted kappa for ordinal data
    mean_absolute_difference: float  # Average |rating1 - rating2|
    n_reratings: int  # Number of re-rated samples

    @property
    def is_acceptable(self) -> bool:
        """Threshold: κ ≥ 0.70 for acceptable self-consistency."""
        return self.kappa >= 0.70

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kappa": self.kappa,
            "percent_agreement": self.percent_agreement,
            "weighted_kappa": self.weighted_kappa,
            "mean_absolute_difference": self.mean_absolute_difference,
            "n_reratings": self.n_reratings,
            "is_acceptable": self.is_acceptable,
        }
