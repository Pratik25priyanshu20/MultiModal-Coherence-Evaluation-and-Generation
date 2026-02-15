"""
MSCI Validation Module

Provides tools for validating the Multimodal Semantic Coherence Index:
- Sensitivity analysis (perturbation tests)
- Human correlation analysis
- Threshold calibration (ROC/AUC)
- Failure mode analysis
"""

from src.validation.msci_sensitivity import (
    MSCISensitivityAnalyzer,
    PerturbationGradient,
)
from src.validation.human_correlation import (
    HumanCorrelationAnalyzer,
    CorrelationResult,
)
from src.validation.threshold_calibration import (
    ThresholdCalibrator,
    CalibrationResult,
)

__all__ = [
    "MSCISensitivityAnalyzer",
    "PerturbationGradient",
    "HumanCorrelationAnalyzer",
    "CorrelationResult",
    "ThresholdCalibrator",
    "CalibrationResult",
]
