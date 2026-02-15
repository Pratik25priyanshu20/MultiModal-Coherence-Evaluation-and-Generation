"""
Tests for domain gating logic in src/generators/image/generator_improved.py

Covers _detect_prompt_domain and _is_domain_compatible without instantiating
the full ImprovedImageRetrievalGenerator (which requires an index file and
AlignedEmbedder).
"""

from __future__ import annotations

import pytest

from src.generators.image.generator_improved import (
    DOMAIN_KEYWORDS,
    INCOMPATIBLE_DOMAINS,
    _detect_prompt_domain,
    _is_domain_compatible,
)


class TestDetectPromptDomain:
    """Tests for _detect_prompt_domain."""

    def test_nature_domain_forest(self):
        """'forest with tall trees' should be detected as nature."""
        domain = _detect_prompt_domain("forest with tall trees")
        assert domain == "nature"

    def test_nature_domain_mountain(self):
        """'a foggy mountain at sunrise' should be nature."""
        domain = _detect_prompt_domain("a foggy mountain at sunrise")
        assert domain == "nature"

    def test_urban_domain_city_street(self):
        """'city street at night' should be detected as urban."""
        domain = _detect_prompt_domain("city street at night")
        assert domain == "urban"

    def test_urban_domain_neon(self):
        """'neon signs on a downtown building' should be urban."""
        domain = _detect_prompt_domain("neon signs on a downtown building")
        assert domain == "urban"

    def test_water_domain_ocean_waves(self):
        """'ocean waves' should be detected as water."""
        domain = _detect_prompt_domain("ocean waves")
        assert domain == "water"

    def test_water_domain_tropical_beach(self):
        """'tropical beach at sunset' should be water."""
        domain = _detect_prompt_domain("tropical beach at sunset")
        assert domain == "water"

    def test_ambiguous_returns_none(self):
        """Prompt with no domain keywords should return None."""
        domain = _detect_prompt_domain("abstract painting of emotion")
        assert domain is None

    def test_empty_prompt_returns_none(self):
        """Empty string should return None."""
        domain = _detect_prompt_domain("")
        assert domain is None

    def test_returns_strongest_domain(self):
        """When multiple domains match, the one with more keyword hits wins."""
        # "forest tree mountain garden park" has 5 nature keywords
        # vs no urban or water keywords
        domain = _detect_prompt_domain("forest tree mountain garden park")
        assert domain == "nature"


class TestIsDomainCompatible:
    """Tests for _is_domain_compatible."""

    def test_rejects_nature_image_for_urban_prompt(self):
        """Urban prompt + nature image should be incompatible."""
        assert _is_domain_compatible("urban", "nature") is False

    def test_rejects_urban_image_for_nature_prompt(self):
        """Nature prompt + urban image should be incompatible."""
        assert _is_domain_compatible("nature", "urban") is False

    def test_rejects_urban_image_for_water_prompt(self):
        """Water prompt + urban image should be incompatible."""
        assert _is_domain_compatible("water", "urban") is False

    def test_allows_other_domain_for_any_prompt(self):
        """'other' domain images should be allowed for any prompt domain."""
        for prompt_domain in ["nature", "urban", "water"]:
            assert _is_domain_compatible(prompt_domain, "other") is True

    def test_allows_same_domain(self):
        """Same domain should always be compatible."""
        for domain in ["nature", "urban", "water"]:
            assert _is_domain_compatible(domain, domain) is True

    def test_allows_everything_when_prompt_domain_is_none(self):
        """If no domain detected (None), all images pass."""
        for img_domain in ["nature", "urban", "water", "other"]:
            assert _is_domain_compatible(None, img_domain) is True

    def test_nature_allows_water(self):
        """Nature prompt should allow water images (not in INCOMPATIBLE_DOMAINS['nature'])."""
        assert _is_domain_compatible("nature", "water") is True

    def test_incompatible_domains_are_symmetric_for_nature_urban(self):
        """Nature-urban incompatibility should be symmetric."""
        assert _is_domain_compatible("nature", "urban") is False
        assert _is_domain_compatible("urban", "nature") is False
