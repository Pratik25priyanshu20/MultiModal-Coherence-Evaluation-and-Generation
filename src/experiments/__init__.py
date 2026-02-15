"""
Experiments module for rigorous empirical evaluation.

Provides:
- ExperimentConfig: Configuration for controlled experiments
- StatisticalAnalysis: Hypothesis testing and effect sizes
- PromptSampler: Stratified prompt sampling
- AblationRunner: Multi-condition ablation studies
"""

from src.experiments.config import ExperimentConfig
from src.experiments.statistical_analysis import (
    paired_ttest,
    compute_effect_size,
    compute_confidence_interval,
    bonferroni_correction,
    StatisticalResult,
)
from src.experiments.prompt_sampler import PromptSampler

__all__ = [
    "ExperimentConfig",
    "paired_ttest",
    "compute_effect_size",
    "compute_confidence_interval",
    "bonferroni_correction",
    "StatisticalResult",
    "PromptSampler",
]
