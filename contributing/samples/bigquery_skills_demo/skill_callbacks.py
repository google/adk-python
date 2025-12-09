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

"""Callback-based Skill Management for ADK Agents.

This module implements automatic skill activation/deactivation using ADK callbacks,
eliminating the need for explicit LLM tool calls to manage skills.

Key Features:
1. **Auto-activation via before_model_callback**: Analyzes user input and activates
   relevant skills before the LLM processes the request.
2. **Auto-deactivation via after_agent_callback**: Cleans up skills after agent
   completes a turn to free context for the next interaction.
3. **Intelligent Detection**: Uses LLM-based classification for semantic understanding
   with keyword fallback for reliability.

Detection Modes:
- "llm": Use LLM classification (most intelligent, understands paraphrases)
- "keyword": Use keyword matching only (fastest, no API calls)
- "hybrid": Try LLM first, fall back to keywords (recommended)

This approach saves LLM calls compared to having the agent explicitly call
activate_skill/deactivate_skill tools.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from .skill_registry import SkillRegistry, ACTIVE_SKILLS_KEY

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.flows.llm_flows.llm_request import LlmRequest
    from google.adk.flows.llm_flows.llm_response import LlmResponse
    from google.genai import types

# Detection mode type
DetectionMode = Literal["llm", "keyword", "hybrid"]


# Default keyword patterns for auto-activating skills (fallback if not in SKILL.md)
# New skills should define keywords in their SKILL.md frontmatter instead of here
DEFAULT_SKILL_ACTIVATION_PATTERNS: dict[str, list[str]] = {
    # These are fallback patterns - skills should define keywords in SKILL.md
}


class SkillCallbacks:
    """Callback handlers for automatic skill management.

    This class creates callbacks that can be registered with an LlmAgent to
    automatically activate and deactivate skills based on user input.

    Supports multiple detection modes:
    - "llm": Use LLM-based classification (most intelligent)
    - "keyword": Use keyword matching (fastest, no API calls)
    - "hybrid": Try LLM first, fall back to keywords (recommended)

    Usage:
        registry = SkillRegistry()
        skill_callbacks = SkillCallbacks(registry, detection_mode="hybrid")

        agent = LlmAgent(
            ...
            before_model_callback=skill_callbacks.before_model_callback,
            after_agent_callback=skill_callbacks.after_agent_callback,
        )
    """

    def __init__(
        self,
        registry: SkillRegistry,
        auto_deactivate: bool = True,
        activation_patterns: dict[str, list[str]] | None = None,
        detection_mode: DetectionMode = "hybrid",
    ):
        """Initialize skill callbacks.

        Args:
            registry: The skill registry to load skills from.
            auto_deactivate: If True, automatically deactivate all skills after
                agent completes a turn. Set to False to keep skills active across
                multiple turns.
            activation_patterns: Custom keyword patterns for skill activation.
                Maps skill name to list of regex patterns. If not provided,
                patterns are built dynamically from SKILL.md keywords.
            detection_mode: How to detect skills from user input:
                - "llm": Use LLM classification (semantic understanding)
                - "keyword": Use keyword matching only (fast, no API calls)
                - "hybrid": Try LLM first, fall back to keywords (recommended)
        """
        self._registry = registry
        self._auto_deactivate = auto_deactivate
        self._detection_mode = detection_mode
        self._classifier = None  # Lazy initialization

        # Build patterns from registry keywords (dynamic, scalable)
        # Falls back to custom patterns if provided, or default patterns
        if activation_patterns is not None:
            self._patterns = activation_patterns
        else:
            self._patterns = self._build_patterns_from_registry()

        # Compile regex patterns for efficiency
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for skill_name, patterns in self._patterns.items():
            self._compiled_patterns[skill_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def _build_patterns_from_registry(self) -> dict[str, list[str]]:
        """Build keyword patterns dynamically from the registry.

        This makes keyword detection scalable - new skills just need to add
        keywords to their SKILL.md frontmatter, no code changes needed.

        Returns:
            Dict mapping skill names to regex patterns built from keywords.
        """
        patterns: dict[str, list[str]] = {}

        # Get all keywords from registry
        all_keywords = self._registry.get_all_keywords()

        for skill_name, keywords in all_keywords.items():
            skill_patterns = []
            for keyword in keywords:
                # Convert keyword to regex pattern
                # Handle multi-word keywords and special characters
                escaped = re.escape(keyword)
                # Use word boundaries for better matching
                # For keywords with dots (like ai.classify), don't add word boundaries
                if "." in keyword:
                    pattern = escaped
                else:
                    pattern = rf"\b{escaped}\b"
                skill_patterns.append(pattern)
            if skill_patterns:
                patterns[skill_name] = skill_patterns

        return patterns

    def _build_skill_content(self, skill_names: list[str]) -> str:
        """Build skill content string from activated skills.

        This loads the full skill documentation from the registry and formats
        it for injection into the system instruction.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted string containing all skill documentation.
        """
        if not skill_names:
            return ""

        skill_sections = []
        for skill_name in skill_names:
            skill = self._registry.load_skill_content(skill_name)
            if skill:
                skill_sections.append(f"""
## Active Skill: {skill.name}

{skill.description}

---

{skill.content}
""")

        if not skill_sections:
            return ""

        return f"""
# Currently Active Skills

The following skills have been loaded and are available for this task:

{"".join(skill_sections)}

---
**Note**: Use `deactivate_skill(skill_name)` when you're done with a skill to free up context.
"""

    def _inject_skills_into_request(
        self,
        llm_request: "LlmRequest",
        skill_names: list[str],
    ) -> None:
        """Inject skill content directly into the LLM request's system instruction.

        This is the key fix for the timing issue: by modifying llm_request.config.system_instruction
        directly in the before_model_callback, the skills are available in the FIRST LLM call,
        not just subsequent calls.

        The LlmRequest has a config.system_instruction field that can be a string.
        We use the append_instructions() method which handles string concatenation properly.

        Args:
            llm_request: The LLM request to modify.
            skill_names: List of skill names to inject.
        """
        if not skill_names:
            return

        skill_content = self._build_skill_content(skill_names)
        if not skill_content:
            return

        # Use append_instructions which properly handles string system_instruction
        # This concatenates to config.system_instruction using "\n\n"
        llm_request.append_instructions([skill_content])

        print(f"[SkillCallbacks] Injected skill content into system instruction: {skill_names}")

    def _get_classifier(self):
        """Lazy initialization of the skill classifier."""
        if self._classifier is None and self._detection_mode in ("llm", "hybrid"):
            try:
                from .skill_classifier import SkillClassifier
                self._classifier = SkillClassifier(self._registry)
            except ImportError:
                # Fall back to keyword mode if classifier not available
                print("[SkillCallbacks] Warning: SkillClassifier not available, using keyword mode")
                self._detection_mode = "keyword"
        return self._classifier

    def _detect_skills_from_keywords(self, text: str) -> list[str]:
        """Detect skills using keyword matching.

        Args:
            text: The text to analyze.

        Returns:
            List of skill names that matched keywords.
        """
        detected_skills = []

        for skill_name, patterns in self._compiled_patterns.items():
            # Only detect skills that exist in the registry
            if skill_name not in self._registry.get_skill_names():
                continue

            for pattern in patterns:
                if pattern.search(text):
                    detected_skills.append(skill_name)
                    break  # One match is enough to activate the skill

        return detected_skills

    def _detect_skills_from_text(self, text: str) -> list[str]:
        """Detect which skills should be activated based on text content.

        Uses the configured detection mode (llm, keyword, or hybrid).

        Args:
            text: The text to analyze (typically user input).

        Returns:
            List of skill names that should be activated.
        """
        if self._detection_mode == "keyword":
            return self._detect_skills_from_keywords(text)

        elif self._detection_mode == "llm":
            classifier = self._get_classifier()
            if classifier:
                result = classifier.classify(text)
                if result.skills:
                    print(f"[SkillCallbacks] LLM detected: {result.skills} (confidence: {result.confidence:.2f})")
                return result.skills
            # Fall back to keyword if classifier unavailable
            return self._detect_skills_from_keywords(text)

        else:  # hybrid mode
            classifier = self._get_classifier()
            if classifier:
                result = classifier.classify_with_fallback(
                    text,
                    keyword_detector=self._detect_skills_from_keywords,
                )
                if result.skills:
                    print(f"[SkillCallbacks] Detected: {result.skills} ({result.reasoning})")
                return result.skills
            # Fall back to keyword if classifier unavailable
            return self._detect_skills_from_keywords(text)

    def _get_original_user_message_text(self, llm_request: "LlmRequest") -> str:
        """Extract the ORIGINAL user message text from an LLM request.

        This looks for the FIRST user message (not the last), which is the
        original user request. In a multi-turn tool-use flow, the conversation
        grows but the original user intent is always in the first user message.

        Args:
            llm_request: The LLM request containing conversation contents.

        Returns:
            The text of the first user message, or empty string if not found.
        """
        if not llm_request.contents:
            return ""

        # Look for the FIRST user message (original user request)
        for content in llm_request.contents:
            if content.role == "user" and content.parts:
                # Concatenate all text parts
                texts = []
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        texts.append(part.text)
                if texts:
                    return " ".join(texts)

        return ""

    def _get_user_message_text(self, llm_request: "LlmRequest") -> str:
        """Extract the latest user message text from an LLM request.

        Args:
            llm_request: The LLM request containing conversation contents.

        Returns:
            The text of the latest user message, or empty string if not found.
        """
        if not llm_request.contents:
            return ""

        # Look for the last user message
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                # Concatenate all text parts
                texts = []
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        texts.append(part.text)
                return " ".join(texts)

        return ""

    def before_model_callback(
        self,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
    ) -> "LlmResponse | None":
        """Auto-activate skills based on user input before LLM processes it.

        This callback analyzes the user message and automatically activates
        relevant skills, then DIRECTLY injects their documentation into the
        llm_request.system_instruction. This ensures skills are available in
        the FIRST LLM call, not just subsequent calls.

        Strategy:
        - If NO skills are currently active: Detect from the LATEST user message
          (this handles new user requests after skills were cleared)
        - If skills ARE already active: Use the ORIGINAL user message to avoid
          re-detecting on subsequent LLM calls in the same tool-use flow
        - ALWAYS inject skills directly into llm_request.system_instruction

        Args:
            callback_context: Context for accessing/modifying state.
            llm_request: The LLM request (can be modified).

        Returns:
            None to proceed with LLM call (we only modify state, not short-circuit).
        """
        # Get current active skills
        active_skills: list[str] = list(
            callback_context.state.get(ACTIVE_SKILLS_KEY, [])
        )

        # Choose which message to analyze based on current skill state
        if not active_skills:
            # No skills active: This is likely a NEW user request
            # Use the LATEST user message to detect skills
            user_text = self._get_user_message_text(llm_request)
            if not user_text:
                return None

            # Detect skills from the latest message
            skills_to_activate = self._detect_skills_from_text(user_text)

            if not skills_to_activate:
                print(f"[SkillCallbacks] No skills detected from: {user_text[:100]}...")
                return None

            # Activate detected skills
            callback_context.state[ACTIVE_SKILLS_KEY] = skills_to_activate
            print(f"[SkillCallbacks] Detecting skills from: {user_text[:100]}...")
            print(f"[SkillCallbacks] Auto-activated skills: {skills_to_activate}")

            # KEY FIX: Inject skills directly into the llm_request
            # This ensures skills are available in the FIRST LLM call
            self._inject_skills_into_request(llm_request, skills_to_activate)
        else:
            # Skills already active: This is a subsequent LLM call in the same flow
            # Use the ORIGINAL user message to ensure consistency
            user_text = self._get_original_user_message_text(llm_request)
            if not user_text:
                # Even if no user text, still inject active skills
                self._inject_skills_into_request(llm_request, active_skills)
                return None

            # Detect which skills should be activated from the original user message
            skills_to_activate = self._detect_skills_from_text(user_text)

            # Check if we need to activate any additional skills
            newly_activated = []
            for skill_name in skills_to_activate:
                if skill_name not in active_skills:
                    active_skills.append(skill_name)
                    newly_activated.append(skill_name)

            # Update state if we activated any new skills
            if newly_activated:
                callback_context.state[ACTIVE_SKILLS_KEY] = active_skills
                print(f"[SkillCallbacks] Additional skills detected: {newly_activated}")

            # KEY FIX: Always inject skills into the llm_request for subsequent calls too
            self._inject_skills_into_request(llm_request, active_skills)

        # Return None to proceed with LLM call
        return None

    def after_agent_callback(
        self,
        callback_context: "CallbackContext",
    ) -> "types.Content | None":
        """Auto-deactivate skills after agent completes a turn.

        This callback cleans up active skills to free context for the next
        interaction. Skills are ephemeral - they're loaded for a task and
        removed when done.

        Args:
            callback_context: Context for accessing/modifying state.

        Returns:
            None (we don't add any content, just modify state).
        """
        if not self._auto_deactivate:
            return None

        # Get current active skills
        active_skills: list[str] = callback_context.state.get(ACTIVE_SKILLS_KEY, [])

        if active_skills:
            # Clear all active skills
            callback_context.state[ACTIVE_SKILLS_KEY] = []
            print(f"[SkillCallbacks] Auto-deactivated skills: {active_skills}")

        return None


def create_skill_callbacks(
    registry: SkillRegistry,
    auto_deactivate: bool = True,
) -> SkillCallbacks:
    """Factory function to create skill callbacks.

    Args:
        registry: The skill registry to use.
        auto_deactivate: Whether to auto-deactivate skills after each turn.

    Returns:
        SkillCallbacks instance with configured callbacks.
    """
    return SkillCallbacks(registry, auto_deactivate=auto_deactivate)


# =============================================================================
# Alternative: Keep skills active until explicitly different task
# =============================================================================

class PersistentSkillCallbacks(SkillCallbacks):
    """Skill callbacks that keep skills active until task changes.

    This variant doesn't auto-deactivate after each turn. Instead, it only
    deactivates skills when the user's request suggests a different task type.

    This is useful for multi-turn conversations about the same topic.
    """

    def __init__(self, registry: SkillRegistry):
        super().__init__(registry, auto_deactivate=False)

    def before_model_callback(
        self,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
    ) -> "LlmResponse | None":
        """Auto-activate skills and potentially switch skills if task changes.

        Uses the ORIGINAL user message for skill detection to ensure skills are
        activated on the first LLM call, not after tool results come back.
        """
        # Extract the ORIGINAL user message (first user message, not last)
        user_text = self._get_original_user_message_text(llm_request)
        if not user_text:
            return None

        # Detect which skills are relevant for this message
        relevant_skills = self._detect_skills_from_text(user_text)

        # Get current active skills
        active_skills: list[str] = list(
            callback_context.state.get(ACTIVE_SKILLS_KEY, [])
        )

        # If we detected new skills, add them
        # If we detected different skills (and have some active), switch to them
        if relevant_skills:
            # Check if this is a task switch (all new skills)
            if active_skills and not any(s in active_skills for s in relevant_skills):
                # User seems to be switching tasks - deactivate old skills
                print(f"[SkillCallbacks] Task switch detected, deactivating: {active_skills}")
                active_skills = []

            # Activate relevant skills
            newly_activated = []
            for skill_name in relevant_skills:
                if skill_name not in active_skills:
                    active_skills.append(skill_name)
                    newly_activated.append(skill_name)

            if newly_activated:
                callback_context.state[ACTIVE_SKILLS_KEY] = active_skills
                print(f"[SkillCallbacks] Auto-activated skills: {newly_activated}")

        return None
