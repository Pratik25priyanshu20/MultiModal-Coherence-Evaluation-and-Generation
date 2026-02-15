"""
Experiment Configuration

Defines configuration for controlled experiments with statistical rigor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class PlannerMode(str, Enum):
    """Planning mode for generation."""
    DIRECT = "direct"  # Raw prompt → generators
    SINGLE_PLANNER = "single_planner"  # 1 LLM call → plan → generators
    COUNCIL = "council"  # 3 LLM calls → merge → generators
    EXTENDED_PROMPT = "extended_prompt"  # 1 LLM call with 3× token budget


class PerturbationType(str, Enum):
    """Perturbation conditions for sensitivity testing."""
    BASELINE = "baseline"  # Normal generation
    WRONG_IMAGE = "wrong_image"  # Image from different prompt
    WRONG_AUDIO = "wrong_audio"  # Audio from different prompt
    SEMANTIC_SHIFT_25 = "semantic_shift_25"  # 25% semantic mismatch
    SEMANTIC_SHIFT_50 = "semantic_shift_50"  # 50% semantic mismatch
    SEMANTIC_SHIFT_75 = "semantic_shift_75"  # 75% semantic mismatch
    RANDOM_IMAGE = "random_image"  # Completely random image
    RANDOM_AUDIO = "random_audio"  # Completely random audio


@dataclass
class ExperimentConfig:
    """
    Configuration for controlled experiments.

    Supports:
    - Multiple planning modes (RQ2: planning effect)
    - Multiple perturbation conditions (RQ1: MSCI sensitivity)
    - Statistical parameters for hypothesis testing
    """
    # Experiment identity
    name: str = "default_experiment"
    description: str = ""

    # Sample sizes
    n_prompts: int = 50  # Minimum for statistical power
    n_seeds: int = 3  # Replications per prompt

    # Experimental conditions
    modes: List[PlannerMode] = field(
        default_factory=lambda: [PlannerMode.DIRECT, PlannerMode.SINGLE_PLANNER]
    )
    perturbations: List[PerturbationType] = field(
        default_factory=lambda: [
            PerturbationType.BASELINE,
            PerturbationType.WRONG_IMAGE,
            PerturbationType.WRONG_AUDIO,
        ]
    )

    # Statistical parameters
    alpha: float = 0.05  # Significance level
    power: float = 0.80  # Target statistical power
    min_effect_size: float = 0.5  # Minimum Cohen's d to detect

    # Execution parameters
    deterministic: bool = True
    base_seed: int = 42
    use_ollama: bool = True
    output_dir: str = "runs/experiments"

    # Resource tracking
    track_tokens: bool = True
    track_time: bool = True

    @property
    def total_runs(self) -> int:
        """Total number of experimental runs."""
        return self.n_prompts * self.n_seeds * len(self.modes) * len(self.perturbations)

    @property
    def conditions(self) -> List[str]:
        """List of all condition keys (mode_perturbation)."""
        return [
            f"{mode.value}_{pert.value}"
            for mode in self.modes
            for pert in self.perturbations
        ]

    def required_sample_size(self, effect_size: Optional[float] = None) -> int:
        """
        Compute required sample size for given effect size using power analysis.

        For paired t-test with alpha=0.05, power=0.80:
        - d=0.5 (medium): N≈34
        - d=0.8 (large): N≈15
        - d=0.3 (small): N≈90

        Uses approximation: N ≈ 2 * ((z_alpha + z_beta) / d)^2
        """
        from scipy import stats

        d = effect_size or self.min_effect_size
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(self.power)

        n = 2 * ((z_alpha + z_beta) / d) ** 2
        return int(n) + 1  # Round up

    def validate(self) -> List[str]:
        """Validate configuration and return any warnings."""
        warnings = []

        required_n = self.required_sample_size()
        if self.n_prompts < required_n:
            warnings.append(
                f"Sample size ({self.n_prompts}) may be underpowered. "
                f"Recommended: {required_n} for effect size d={self.min_effect_size}"
            )

        if self.n_seeds < 2:
            warnings.append(
                "n_seeds < 2: No replication variance can be estimated"
            )

        if self.alpha > 0.10:
            warnings.append(
                f"High alpha ({self.alpha}): Increased false positive risk"
            )

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "n_prompts": self.n_prompts,
            "n_seeds": self.n_seeds,
            "modes": [m.value for m in self.modes],
            "perturbations": [p.value for p in self.perturbations],
            "alpha": self.alpha,
            "power": self.power,
            "min_effect_size": self.min_effect_size,
            "deterministic": self.deterministic,
            "base_seed": self.base_seed,
            "use_ollama": self.use_ollama,
            "output_dir": self.output_dir,
            "track_tokens": self.track_tokens,
            "track_time": self.track_time,
            "total_runs": self.total_runs,
            "conditions": self.conditions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        # Convert string enums back
        if "modes" in data:
            data["modes"] = [PlannerMode(m) for m in data["modes"]]
        if "perturbations" in data:
            data["perturbations"] = [PerturbationType(p) for p in data["perturbations"]]

        # Remove computed fields
        data.pop("total_runs", None)
        data.pop("conditions", None)

        return cls(**data)

    def save(self, path: Path):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# Preset configurations for common experiment types
PRESETS = {
    "rq1_sensitivity": ExperimentConfig(
        name="RQ1: MSCI Sensitivity",
        description="Test if MSCI is sensitive to controlled semantic perturbations",
        n_prompts=50,
        n_seeds=3,
        modes=[PlannerMode.SINGLE_PLANNER],
        perturbations=[
            PerturbationType.BASELINE,
            PerturbationType.WRONG_IMAGE,
            PerturbationType.WRONG_AUDIO,
            PerturbationType.SEMANTIC_SHIFT_25,
            PerturbationType.SEMANTIC_SHIFT_50,
            PerturbationType.SEMANTIC_SHIFT_75,
        ],
    ),
    "rq2_planning": ExperimentConfig(
        name="RQ2: Planning Effect",
        description="Test if structured planning improves cross-modal alignment",
        n_prompts=50,
        n_seeds=3,
        modes=[
            PlannerMode.DIRECT,
            PlannerMode.SINGLE_PLANNER,
            PlannerMode.COUNCIL,
            PlannerMode.EXTENDED_PROMPT,
        ],
        perturbations=[PerturbationType.BASELINE],
    ),
    "full_ablation": ExperimentConfig(
        name="Full Ablation Study",
        description="Complete ablation across all modes and perturbations",
        n_prompts=50,
        n_seeds=3,
        modes=[
            PlannerMode.DIRECT,
            PlannerMode.SINGLE_PLANNER,
            PlannerMode.COUNCIL,
            PlannerMode.EXTENDED_PROMPT,
        ],
        perturbations=[
            PerturbationType.BASELINE,
            PerturbationType.WRONG_IMAGE,
            PerturbationType.WRONG_AUDIO,
        ],
    ),
    "quick_test": ExperimentConfig(
        name="Quick Test",
        description="Small-scale test run",
        n_prompts=5,
        n_seeds=1,
        modes=[PlannerMode.DIRECT, PlannerMode.SINGLE_PLANNER],
        perturbations=[PerturbationType.BASELINE, PerturbationType.WRONG_IMAGE],
    ),
}
