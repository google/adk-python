# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Intelligent Skill Classification using LLM.

This module provides intelligent skill detection using a fast LLM model
to understand user intent and determine which skills are needed.

Key Features:
1. **LLM-based Classification**: Uses gemini-2.0-flash for fast, accurate intent detection
2. **Semantic Understanding**: Understands paraphrased requests, not just keywords
3. **Confidence Scores**: Returns confidence levels for skill activation decisions
4. **Caching**: Caches recent classifications to avoid redundant API calls
5. **Fallback**: Falls back to keyword matching if LLM is unavailable

Example use cases that keywords miss but LLM catches:
- "Help me understand patterns in customer behavior" → bqml (clustering)
- "I want to automatically categorize support tickets" → bq_ai_operator
- "Build something to predict next month's sales" → bqml (forecasting)
- "Rate these product descriptions by quality" → bq_ai_operator (AI.SCORE)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from google import genai

from .skill_registry import SkillRegistry

if TYPE_CHECKING:
    pass


@dataclass
class SkillClassification:
    """Result of skill classification."""
    skills: list[str]  # List of skill names to activate
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Brief explanation of why these skills were selected


# Classification prompt template - DYNAMIC version
# Skills are injected at runtime from the registry
CLASSIFICATION_PROMPT_TEMPLATE = """\
You are a skill classifier for a BigQuery data science agent. Your job is to determine which specialized skills (if any) should be loaded based on the user's request.

Available skills:
{skills_description}

Analyze the user's request and determine which skills are needed.

User request: "{user_input}"

Respond with a JSON object (no markdown, just raw JSON):
{{
  "skills": ["skill_name1", "skill_name2"],
  "confidence": 0.85,
  "reasoning": "Brief explanation"
}}

Rules:
- Return empty skills [] if the request is just a simple query (SELECT, list tables, etc.)
- Only return skills from the available list above
- Confidence should be 0.0-1.0 based on how certain you are
- Be conservative: only activate skills when clearly needed
"""


class SkillClassifier:
    """Intelligent skill classifier using LLM.

    This classifier uses a fast LLM model to understand user intent
    and determine which skills should be activated.

    SCALABILITY: Skills are dynamically loaded from the registry, so adding
    new skills only requires creating a SKILL.md file - no code changes needed.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        model: str = "gemini-2.0-flash",
        confidence_threshold: float = 0.6,
        cache_size: int = 100,
    ):
        """Initialize the skill classifier.

        Args:
            registry: The skill registry to validate skill names against.
            model: The LLM model to use for classification.
            confidence_threshold: Minimum confidence to activate a skill.
            cache_size: Number of recent classifications to cache.
        """
        self._registry = registry
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._cache_size = cache_size
        self._client: genai.Client | None = None

        # Simple in-memory cache for recent classifications
        self._cache: dict[str, SkillClassification] = {}

        # Build skills description from registry (dynamic, not hardcoded)
        self._skills_description = self._build_skills_description()

    def _build_skills_description(self) -> str:
        """Build skills description dynamically from the registry.

        This makes the classifier scalable - new skills are automatically
        included without code changes.
        """
        descriptions = []
        for i, metadata in enumerate(self._registry.get_all_metadata(), 1):
            descriptions.append(f"{i}. **{metadata.name}** - {metadata.description}")
        return "\n".join(descriptions) if descriptions else "No skills available."

    def _build_prompt(self, user_input: str) -> str:
        """Build the classification prompt with dynamic skills."""
        return CLASSIFICATION_PROMPT_TEMPLATE.format(
            skills_description=self._skills_description,
            user_input=user_input,
        )

    def _get_client(self) -> genai.Client:
        """Get or create the genai client."""
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def _normalize_input(self, text: str) -> str:
        """Normalize input text for caching."""
        # Lowercase and remove extra whitespace
        return " ".join(text.lower().split())

    def _parse_response(self, response_text: str) -> SkillClassification | None:
        """Parse the LLM response into a SkillClassification."""
        try:
            # Try to extract JSON from the response
            # Handle cases where LLM might wrap in markdown code blocks
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            # Validate and filter skills
            valid_skill_names = set(self._registry.get_skill_names())
            skills = [s for s in data.get("skills", []) if s in valid_skill_names]

            return SkillClassification(
                skills=skills,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def classify(self, user_input: str) -> SkillClassification:
        """Classify user input to determine which skills are needed.

        Args:
            user_input: The user's message/request.

        Returns:
            SkillClassification with recommended skills and confidence.
        """
        # Check cache first
        cache_key = self._normalize_input(user_input)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Call the LLM
            client = self._get_client()
            prompt = self._build_prompt(user_input)

            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "max_output_tokens": 256,
                },
            )

            # Parse the response
            result = self._parse_response(response.text)

            if result is None:
                # Fallback to empty classification
                result = SkillClassification(
                    skills=[],
                    confidence=0.0,
                    reasoning="Failed to parse LLM response",
                )

            # Apply confidence threshold
            if result.confidence < self._confidence_threshold:
                result = SkillClassification(
                    skills=[],
                    confidence=result.confidence,
                    reasoning=f"Below confidence threshold ({self._confidence_threshold}): {result.reasoning}",
                )

            # Cache the result
            if len(self._cache) >= self._cache_size:
                # Simple cache eviction: remove first item
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[cache_key] = result

            return result

        except Exception as e:
            # Return empty classification on error
            return SkillClassification(
                skills=[],
                confidence=0.0,
                reasoning=f"Classification error: {str(e)}",
            )

    def classify_with_fallback(
        self,
        user_input: str,
        keyword_detector: callable | None = None,
    ) -> SkillClassification:
        """Classify with keyword fallback if LLM fails.

        Args:
            user_input: The user's message/request.
            keyword_detector: Optional function that returns skill names from text.

        Returns:
            SkillClassification with recommended skills.
        """
        # Try LLM classification first
        result = self.classify(user_input)

        # If LLM returned skills with good confidence, use that
        if result.skills and result.confidence >= self._confidence_threshold:
            return result

        # Fall back to keyword detection if provided
        if keyword_detector is not None:
            keyword_skills = keyword_detector(user_input)
            if keyword_skills:
                return SkillClassification(
                    skills=keyword_skills,
                    confidence=0.7,  # Medium confidence for keyword match
                    reasoning="Detected via keyword matching (LLM uncertain)",
                )

        return result


# Convenience function for quick classification
def classify_skills(
    user_input: str,
    registry: SkillRegistry | None = None,
) -> list[str]:
    """Quick function to classify skills for a user input.

    Args:
        user_input: The user's message.
        registry: Optional skill registry (uses default if not provided).

    Returns:
        List of skill names to activate.
    """
    if registry is None:
        registry = SkillRegistry()

    classifier = SkillClassifier(registry)
    result = classifier.classify(user_input)
    return result.skills
