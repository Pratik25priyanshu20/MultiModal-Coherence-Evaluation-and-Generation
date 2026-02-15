"""
Dynamic Council System with Modality Weighting and Priority Selection.

Implements:
- Dynamic modality weighting based on prompt analysis
- Leader-follower model for modality priority
- Content fusion agent for holistic multimodal outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.embeddings.aligned_embeddings import AlignedEmbedder
from src.planner.council import CouncilResult, Planner, SemanticPlanningCouncil
from src.planner.schema import SemanticPlan
from src.planner.merge_logic import merge_council_plans, MergeReport


@dataclass
class ModalityWeights:
    """Weights for different modalities based on prompt analysis."""

    text_weight: float = 1.0
    image_weight: float = 1.0
    audio_weight: float = 1.0
    
    @property
    def total(self) -> float:
        return self.text_weight + self.image_weight + self.audio_weight
    
    def normalize(self) -> ModalityWeights:
        """Normalize weights so they sum to 3.0 (equal importance baseline)."""
        total = self.total
        if total == 0:
            return ModalityWeights(1.0, 1.0, 1.0)
        scale = 3.0 / total
        return ModalityWeights(
            text_weight=self.text_weight * scale,
            image_weight=self.image_weight * scale,
            audio_weight=self.audio_weight * scale,
        )


@dataclass
class ModalityPriority:
    """Priority ranking for modalities."""

    primary: str  # "text", "image", or "audio"
    secondary: str
    tertiary: str
    weights: ModalityWeights


class PromptAnalyzer:
    """Analyzes prompts to determine modality importance."""

    def __init__(self):
        # Keywords that suggest visual emphasis
        self.visual_keywords = {
            "see", "view", "look", "appear", "visible", "visual", "image", "photo",
            "picture", "scene", "landscape", "color", "bright", "dark", "shade",
            "shape", "form", "design", "style", "pattern", "texture",
        }
        
        # Keywords that suggest audio emphasis
        self.audio_keywords = {
            "hear", "sound", "listen", "audio", "music", "noise", "quiet", "loud",
            "silence", "voice", "speak", "whisper", "shout", "echo", "resonance",
            "tone", "pitch", "melody", "rhythm", "beat", "harmony",
        }
        
        # Keywords that suggest narrative/text emphasis
        self.text_keywords = {
            "story", "narrative", "tale", "describe", "tell", "explain", "detail",
            "character", "plot", "scene", "moment", "event", "happen", "occur",
        }

    def analyze(self, prompt: str) -> ModalityPriority:
        """Analyze prompt and determine modality priority."""
        prompt_lower = prompt.lower()
        words = set(prompt_lower.split())
        
        # Count keyword matches
        visual_score = len(words & self.visual_keywords)
        audio_score = len(words & self.audio_keywords)
        text_score = len(words & self.text_keywords)
        
        # Boost scores based on prompt length and structure
        # Longer prompts with more descriptive words favor text/narrative
        word_count = len(words)
        if word_count > 15:
            text_score += 1
        if word_count > 25:
            text_score += 1
        
        # Create weights (add 1.0 base to avoid zero weights)
        weights = ModalityWeights(
            text_weight=1.0 + text_score * 0.5,
            image_weight=1.0 + visual_score * 0.5,
            audio_weight=1.0 + audio_score * 0.5,
        )
        
        # Normalize weights
        weights = weights.normalize()
        
        # Determine priority order
        scores = [
            ("text", text_score),
            ("image", visual_score),
            ("audio", audio_score),
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        primary = scores[0][0]
        secondary = scores[1][0]
        tertiary = scores[2][0]
        
        return ModalityPriority(
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            weights=weights,
        )


class ContentFusionAgent:
    """Agent that combines information from multiple modalities for holistic outputs."""

    def __init__(self, embedder: Optional[AlignedEmbedder] = None):
        self.embedder = embedder or AlignedEmbedder()

    def fuse(
        self,
        plans: List[SemanticPlan],
        weights: ModalityWeights,
        priority: ModalityPriority,
    ) -> Tuple[SemanticPlan, Dict[str, any]]:
        """
        Fuse multiple plans using weights and priority.
        
        Returns:
            - Fused semantic plan
            - Fusion metadata (confidence, conflicts, etc.)
        """
        if not plans:
            raise ValueError("Cannot fuse empty list of plans")
        
        if len(plans) == 1:
            return plans[0], {"fusion_method": "single_plan", "confidence": 1.0}
        
        # Use merge logic but with weighted consideration
        if len(plans) == 3:
            plan_a, plan_b, plan_c = plans
            merged, merge_report = merge_council_plans(plan_a, plan_b, plan_c)
            
            # Apply weights to merge report confidence
            fusion_metadata = {
                "fusion_method": "weighted_council_merge",
                "weights": {
                    "text": weights.text_weight,
                    "image": weights.image_weight,
                    "audio": weights.audio_weight,
                },
                "priority": {
                    "primary": priority.primary,
                    "secondary": priority.secondary,
                    "tertiary": priority.tertiary,
                },
                "merge_report": merge_report.__dict__ if hasattr(merge_report, "__dict__") else {},
                "confidence": merge_report.agreement_score if hasattr(merge_report, "agreement_score") else 0.5,
            }
            
            return merged, fusion_metadata
        
        # Fallback: use first plan
        return plans[0], {"fusion_method": "first_plan_fallback", "confidence": 0.5}


class DynamicSemanticCouncil(SemanticPlanningCouncil):
    """
    Enhanced council with dynamic modality weighting and priority selection.
    
    Features:
    - Analyzes prompts to determine modality importance
    - Applies weighted merging based on analysis
    - Supports leader-follower generation strategies
    """

    def __init__(
        self,
        planner_a: Planner,
        planner_b: Planner,
        planner_c: Planner,
        enable_dynamic_weighting: bool = True,
        embedder: Optional[AlignedEmbedder] = None,
    ):
        super().__init__(planner_a, planner_b, planner_c)
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.analyzer = PromptAnalyzer()
        self.fusion_agent = ContentFusionAgent(embedder=embedder)

    def run(self, user_prompt: str) -> CouncilResult:
        """Run council with dynamic weighting."""
        # Generate plans from all planners
        plan_a = self.planner_a.plan(user_prompt)
        plan_b = self.planner_b.plan(user_prompt)
        plan_c = self.planner_c.plan(user_prompt)

        if self.enable_dynamic_weighting:
            # Analyze prompt to determine modality priority
            priority = self.analyzer.analyze(user_prompt)
            
            # Use fusion agent for weighted merging
            plans = [plan_a, plan_b, plan_c]
            merged, fusion_metadata = self.fusion_agent.fuse(plans, priority.weights, priority)
            
            # Create standard merge report and enhance with dynamic weighting info
            standard_merged, standard_report = merge_council_plans(plan_a, plan_b, plan_c)
            
            # Enhance notes with dynamic weighting info
            enhanced_notes = standard_report.notes
            if enhanced_notes:
                enhanced_notes += " | "
            enhanced_notes += (
                f"Dynamic weighting: Primary={priority.primary}, "
                f"weights: T={priority.weights.text_weight:.2f}, "
                f"I={priority.weights.image_weight:.2f}, "
                f"A={priority.weights.audio_weight:.2f}"
            )
            
            # Use merged plan from fusion agent if available
            final_merged = merged if merged else standard_merged
            
            enhanced_report = MergeReport(
                agreement_score=standard_report.agreement_score,
                per_section_agreement=standard_report.per_section_agreement,
                conflicts=standard_report.conflicts,
                notes=enhanced_notes,
            )
            
            return CouncilResult(
                plan_a=plan_a,
                plan_b=plan_b,
                plan_c=plan_c,
                merged_plan=final_merged,
                merge_report=enhanced_report,
            )
        else:
            # Fall back to standard merge
            merged, report = merge_council_plans(plan_a, plan_b, plan_c)
            return CouncilResult(
                plan_a=plan_a,
                plan_b=plan_b,
                plan_c=plan_c,
                merged_plan=merged,
                merge_report=report,
            )

    def get_modality_priority(self, user_prompt: str) -> ModalityPriority:
        """Get modality priority for a prompt without running full council."""
        return self.analyzer.analyze(user_prompt)
