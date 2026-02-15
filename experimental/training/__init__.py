"""
Training Module for Learned Embedding Alignment

Provides infrastructure for:
- Contrastive learning on multimodal triplets
- Projection head training
- MSCI weight optimization
"""

from src.training.learned_projection import (
    LearnedProjection,
    AlignedProjectionHead,
)
from src.training.contrastive_trainer import (
    ContrastiveTrainer,
    ContrastiveLoss,
)

__all__ = [
    "LearnedProjection",
    "AlignedProjectionHead",
    "ContrastiveTrainer",
    "ContrastiveLoss",
]
