"""
Prompt Sampler Module

Provides stratified sampling of prompts for controlled experiments.
Ensures diversity across semantic categories while maintaining balance.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


@dataclass
class PromptCategory:
    """A semantic category for prompt stratification."""
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


# Default categories for multimodal scene prompts
DEFAULT_CATEGORIES = [
    PromptCategory(
        name="nature",
        description="Natural environments, landscapes, wildlife",
        keywords=["forest", "ocean", "mountain", "river", "wildlife", "sunset", "rain", "storm", "garden", "meadow"],
        examples=[
            "A calm foggy forest at dawn with distant birds and soft wind",
            "Ocean waves crashing against rocky cliffs under a stormy sky",
            "A peaceful meadow with wildflowers swaying in gentle breeze",
        ],
    ),
    PromptCategory(
        name="urban",
        description="City scenes, street life, architecture",
        keywords=["city", "street", "building", "traffic", "neon", "market", "crowd", "subway", "cafe", "downtown"],
        examples=[
            "A rainy neon-lit city street at night with reflections on wet pavement",
            "Busy morning market with vendors and the sounds of commerce",
            "Empty subway platform with echoing announcements",
        ],
    ),
    PromptCategory(
        name="abstract",
        description="Abstract concepts, emotions, moods",
        keywords=["peaceful", "chaotic", "melancholy", "joyful", "mysterious", "ethereal", "surreal", "dreamy"],
        examples=[
            "An ethereal dreamscape with floating lights and distant echoes",
            "The feeling of nostalgia on a quiet autumn afternoon",
            "A surreal landscape where gravity seems optional",
        ],
    ),
    PromptCategory(
        name="action",
        description="Dynamic scenes with movement and activity",
        keywords=["running", "flying", "dancing", "fighting", "racing", "jumping", "working", "playing"],
        examples=[
            "A horse galloping across an open prairie",
            "Children playing in a park with laughter and shouts",
            "Fireworks exploding over a celebration crowd",
        ],
    ),
    PromptCategory(
        name="domestic",
        description="Indoor scenes, everyday life, home settings",
        keywords=["kitchen", "bedroom", "office", "library", "home", "cozy", "fireplace", "window"],
        examples=[
            "A cozy reading nook with rain pattering on the window",
            "Morning kitchen scene with sizzling breakfast sounds",
            "A quiet home office with soft keyboard clicks",
        ],
    ),
]


@dataclass
class SampledPrompt:
    """A prompt with its category and metadata."""
    text: str
    category: str
    source: str = "generated"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "category": self.category,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampledPrompt":
        """Create from dictionary."""
        return cls(**data)


class PromptSampler:
    """
    Stratified prompt sampler for controlled experiments.

    Ensures balanced representation across semantic categories
    while allowing for custom prompts and external datasets.
    """

    def __init__(
        self,
        categories: Optional[List[PromptCategory]] = None,
        seed: int = 42,
    ):
        """
        Initialize sampler.

        Args:
            categories: List of semantic categories (uses defaults if None)
            seed: Random seed for reproducibility
        """
        self.categories = categories or DEFAULT_CATEGORIES
        self.category_names = [c.name for c in self.categories]
        self.rng = random.Random(seed)

        # Build keyword index for categorization
        self._keyword_to_category: Dict[str, str] = {}
        for cat in self.categories:
            for keyword in cat.keywords:
                self._keyword_to_category[keyword.lower()] = cat.name

    def categorize_prompt(self, prompt: str) -> str:
        """
        Assign a category to a prompt based on keyword matching.

        Args:
            prompt: The prompt text

        Returns:
            Category name (or "other" if no match)
        """
        prompt_lower = prompt.lower()

        # Count keyword matches per category
        category_scores: Dict[str, int] = defaultdict(int)

        for keyword, category in self._keyword_to_category.items():
            if keyword in prompt_lower:
                category_scores[category] += 1

        if category_scores:
            return max(category_scores, key=lambda k: category_scores[k])

        return "other"

    def sample_stratified(
        self,
        n_total: int,
        prompts_per_category: Optional[int] = None,
        custom_prompts: Optional[List[str]] = None,
        include_examples: bool = True,
    ) -> List[SampledPrompt]:
        """
        Sample prompts with stratified distribution across categories.

        Args:
            n_total: Total number of prompts to sample
            prompts_per_category: Override for prompts per category
            custom_prompts: Additional prompts to include (will be categorized)
            include_examples: Whether to include category example prompts

        Returns:
            List of SampledPrompt objects
        """
        n_categories = len(self.categories)
        per_category = prompts_per_category or (n_total // n_categories)

        sampled: List[SampledPrompt] = []

        # Organize custom prompts by category
        custom_by_category: Dict[str, List[str]] = defaultdict(list)
        if custom_prompts:
            for prompt in custom_prompts:
                category = self.categorize_prompt(prompt)
                custom_by_category[category].append(prompt)

        # Sample from each category
        for cat in self.categories:
            category_prompts: List[str] = []

            # Add examples if enabled
            if include_examples:
                category_prompts.extend(cat.examples)

            # Add custom prompts for this category
            category_prompts.extend(custom_by_category.get(cat.name, []))

            # Shuffle and select
            self.rng.shuffle(category_prompts)
            selected = category_prompts[:per_category]

            for text in selected:
                sampled.append(SampledPrompt(
                    text=text,
                    category=cat.name,
                    source="example" if text in cat.examples else "custom",
                ))

        # Handle "other" category from custom prompts
        other_prompts = custom_by_category.get("other", [])
        if other_prompts:
            self.rng.shuffle(other_prompts)
            for text in other_prompts[:per_category]:
                sampled.append(SampledPrompt(
                    text=text,
                    category="other",
                    source="custom",
                ))

        # Shuffle final list and trim to exact n_total
        self.rng.shuffle(sampled)
        return sampled[:n_total]

    def load_from_file(
        self,
        path: Path,
        n_samples: Optional[int] = None,
    ) -> List[SampledPrompt]:
        """
        Load and categorize prompts from a file.

        Supports JSON (list or object with "prompts" key) and plain text (one per line).

        Args:
            path: Path to prompts file
            n_samples: Optional limit on number of prompts

        Returns:
            List of SampledPrompt objects
        """
        path = Path(path)

        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and "prompts" in data:
                prompts = data["prompts"]
            else:
                raise ValueError(f"Unexpected JSON structure in {path}")

        else:
            # Plain text, one prompt per line
            with path.open("r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]

        # Categorize and create SampledPrompt objects
        sampled = []
        for prompt in prompts:
            if isinstance(prompt, dict):
                text = prompt.get("text") or prompt.get("prompt", "")
                category = prompt.get("category") or self.categorize_prompt(text)
            else:
                text = str(prompt)
                category = self.categorize_prompt(text)

            sampled.append(SampledPrompt(
                text=text,
                category=category,
                source=str(path.name),
            ))

        # Shuffle and limit
        self.rng.shuffle(sampled)
        if n_samples:
            sampled = sampled[:n_samples]

        return sampled

    def load_from_laion(
        self,
        laion_dir: Path = Path("data/laion"),
        n_samples: int = 50,
    ) -> List[SampledPrompt]:
        """
        Load prompts from LAION subset.

        Args:
            laion_dir: Path to LAION data directory
            n_samples: Number of prompts to sample

        Returns:
            List of SampledPrompt objects
        """
        prompts_file = laion_dir / "prompts_500.json"

        if not prompts_file.exists():
            # Try alternative locations
            for alt in [laion_dir / "prompts.json", laion_dir / "captions.json"]:
                if alt.exists():
                    prompts_file = alt
                    break

        if not prompts_file.exists():
            raise FileNotFoundError(f"No prompts file found in {laion_dir}")

        return self.load_from_file(prompts_file, n_samples)

    def load_from_audiocaps(
        self,
        audiocaps_dir: Path = Path("data/audiocaps"),
        n_samples: int = 50,
    ) -> List[SampledPrompt]:
        """
        Load prompts from AudioCaps dataset.

        Args:
            audiocaps_dir: Path to AudioCaps data directory
            n_samples: Number of prompts to sample

        Returns:
            List of SampledPrompt objects
        """
        # Look for processed captions
        for filename in ["captions.json", "audiocaps_subset.json", "prompts.json"]:
            captions_file = audiocaps_dir / filename
            if captions_file.exists():
                return self.load_from_file(captions_file, n_samples)

        raise FileNotFoundError(f"No captions file found in {audiocaps_dir}")

    def get_category_distribution(
        self,
        prompts: List[SampledPrompt],
    ) -> Dict[str, int]:
        """
        Get distribution of prompts across categories.

        Args:
            prompts: List of sampled prompts

        Returns:
            Dictionary mapping category names to counts
        """
        distribution: Dict[str, int] = defaultdict(int)
        for prompt in prompts:
            distribution[prompt.category] += 1
        return dict(distribution)

    def save_sample_set(
        self,
        prompts: List[SampledPrompt],
        path: Path,
    ):
        """
        Save a sample set to JSON file.

        Args:
            prompts: List of sampled prompts
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_prompts": len(prompts),
            "distribution": self.get_category_distribution(prompts),
            "prompts": [p.to_dict() for p in prompts],
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_sample_set(cls, path: Path) -> List[SampledPrompt]:
        """
        Load a previously saved sample set.

        Args:
            path: Path to JSON file

        Returns:
            List of SampledPrompt objects
        """
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        return [SampledPrompt.from_dict(p) for p in data["prompts"]]


def create_experiment_prompts(
    n_prompts: int = 50,
    output_path: Optional[Path] = None,
    seed: int = 42,
    custom_prompts: Optional[List[str]] = None,
) -> List[SampledPrompt]:
    """
    Convenience function to create a balanced prompt set for experiments.

    Args:
        n_prompts: Total number of prompts
        output_path: Optional path to save the prompt set
        seed: Random seed
        custom_prompts: Optional custom prompts to include

    Returns:
        List of SampledPrompt objects
    """
    sampler = PromptSampler(seed=seed)

    prompts = sampler.sample_stratified(
        n_total=n_prompts,
        custom_prompts=custom_prompts,
        include_examples=True,
    )

    if output_path:
        sampler.save_sample_set(prompts, output_path)

    return prompts
