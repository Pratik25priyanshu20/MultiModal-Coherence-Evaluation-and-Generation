"""
Phase 3: Validation to ensure generators use ONLY semantic plan, not raw prompt.

This module provides validation functions to detect prompt leakage.
"""

from typing import Dict, Any, List, Tuple


def extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text (simple heuristic)."""
    # Remove common stopwords and punctuation
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "they", "them", "their",
    }
    
    # Simple tokenization
    words = text.lower().replace(",", " ").replace(".", " ").split()
    keywords = {w.strip() for w in words if len(w) > 3 and w not in stopwords}
    return keywords


def detect_prompt_leakage(
    original_prompt: str,
    plan: Dict[str, Any],
    text_prompt: str,
    image_prompt: str,
    audio_prompt: str,
) -> Dict[str, Any]:
    """
    Detect if any generator prompt contains words from original prompt
    that are NOT in the semantic plan.
    
    Returns:
        {
            "has_leakage": bool,
            "leakage_details": {
                "text": [...],
                "image": [...],
                "audio": [...]
            },
            "plan_coverage": float  # How much of original prompt is covered by plan
        }
    """
    original_keywords = extract_keywords(original_prompt)
    
    # Extract keywords from plan (convert plan to text representation)
    plan_text = _plan_to_text(plan)
    plan_keywords = extract_keywords(plan_text)
    
    # Keywords in original but NOT in plan (potential leakage)
    leakage_candidates = original_keywords - plan_keywords
    
    # Check each modality prompt
    text_keywords = extract_keywords(text_prompt)
    image_keywords = extract_keywords(image_prompt)
    audio_keywords = extract_keywords(audio_prompt)
    
    text_leakage = text_keywords & leakage_candidates
    image_leakage = image_keywords & leakage_candidates
    audio_leakage = audio_keywords & leakage_candidates
    
    has_leakage = bool(text_leakage or image_leakage or audio_leakage)
    
    # Plan coverage: how much of original is explained by plan
    plan_coverage = len(plan_keywords & original_keywords) / max(len(original_keywords), 1)
    
    return {
        "has_leakage": has_leakage,
        "leakage_details": {
            "text": sorted(text_leakage),
            "image": sorted(image_leakage),
            "audio": sorted(audio_leakage),
        },
        "plan_coverage": round(plan_coverage, 4),
        "original_keywords_count": len(original_keywords),
        "plan_keywords_count": len(plan_keywords),
    }


def _plan_to_text(plan: Dict[str, Any]) -> str:
    """Convert semantic plan to text representation for keyword extraction."""
    parts = []
    
    if isinstance(plan, dict):
        # Extract all string fields from plan
        for key, value in plan.items():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(v) for v in value if isinstance(v, str))
            elif isinstance(value, dict):
                # Recursively extract from nested dicts
                parts.append(_plan_to_text(value))
    
    return " ".join(parts)


def validate_conditioning_strictness(
    original_prompt: str,
    plan: Dict[str, Any],
    modality_prompts: Dict[str, str],
) -> Tuple[bool, str]:
    """
    Validate that modality prompts are derived strictly from plan.
    
    Returns:
        (is_valid, reason)
    """
    validation = detect_prompt_leakage(
        original_prompt=original_prompt,
        plan=plan,
        text_prompt=modality_prompts.get("text_prompt", ""),
        image_prompt=modality_prompts.get("image_prompt", ""),
        audio_prompt=modality_prompts.get("audio_prompt", ""),
    )
    
    if validation["has_leakage"]:
        leakage = validation["leakage_details"]
        reasons = []
        if leakage["text"]:
            reasons.append(f"Text prompt leaks: {leakage['text']}")
        if leakage["image"]:
            reasons.append(f"Image prompt leaks: {leakage['image']}")
        if leakage["audio"]:
            reasons.append(f"Audio prompt leaks: {leakage['audio']}")
        return False, "; ".join(reasons)
    
    return True, "All prompts derived from plan"


def check_plan_completeness(plan: Dict[str, Any]) -> Dict[str, bool]:
    """Check if plan has all necessary fields for conditioning."""
    if isinstance(plan, dict):
        has_scene = bool(plan.get("scene_summary") or plan.get("scene"))
        has_visual = bool(plan.get("visual_attributes") or plan.get("visual_elements"))
        has_audio = bool(plan.get("audio_elements") or plan.get("audio_intent"))
        has_entities = bool(plan.get("primary_entities") or plan.get("main_subjects"))
        
        return {
            "has_scene": has_scene,
            "has_visual": has_visual,
            "has_audio": has_audio,
            "has_entities": has_entities,
            "is_complete": has_scene and (has_visual or has_audio),
        }
    
    return {
        "has_scene": False,
        "has_visual": False,
        "has_audio": False,
        "has_entities": False,
        "is_complete": False,
    }
