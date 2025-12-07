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

"""Dynamic Skill Registry following Anthropic's Skills Pattern.

This module implements the dynamic skill discovery pattern from:
https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills

Key concepts:
1. Skills are stored as SKILL.md files with YAML frontmatter
2. At startup, only skill names and descriptions are loaded (progressive disclosure)
3. Agent activates/deactivates skills which inject content into system prompt
4. Skill content is in system prompt (not conversation history) - ephemeral!

This approach mirrors Claude Code's filesystem-based skill loading where:
- Skills are loaded on-demand into the current context
- Skills can be unloaded when no longer needed
- Context doesn't accumulate skill content permanently
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import ToolContext


# State key for tracking active skills (using temp: prefix so it's session-scoped)
ACTIVE_SKILLS_KEY = "active_skills"


@dataclass
class SkillMetadata:
    """Metadata for a skill (name + description only)."""
    name: str
    description: str
    path: Path


@dataclass
class SkillContent:
    """Full content of a skill including documentation."""
    name: str
    description: str
    content: str
    path: Path


class SkillRegistry:
    """Registry for dynamically discovering and loading skills.

    Following Anthropic's progressive disclosure pattern:
    - Level 1: Skill names and descriptions (loaded at startup)
    - Level 2: Full skill content (injected into system prompt when activated)

    Skills are discovered from SKILL.md files in the skills directory.
    Each SKILL.md has YAML frontmatter with name and description.

    Example SKILL.md structure:
    ```markdown
    ---
    name: my_skill
    description: Short description of what this skill does
    ---

    # Full Skill Documentation

    Detailed instructions, examples, and usage patterns...
    ```
    """

    def __init__(self, skills_dir: str | Path | None = None):
        """Initialize the skill registry.

        Args:
            skills_dir: Directory containing skill subdirectories.
                       Each subdirectory should have a SKILL.md file.
                       Defaults to ./skills relative to this module.
        """
        if skills_dir is None:
            # Default to skills/ directory next to this file
            skills_dir = Path(__file__).parent / "skills"
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillMetadata] = {}
        self._discover_skills()

    def _discover_skills(self) -> None:
        """Discover all skills in the skills directory.

        Scans for SKILL.md files and parses YAML frontmatter
        to extract name and description (Level 1 disclosure).
        """
        if not self._skills_dir.exists():
            return

        for skill_dir in self._skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            metadata = self._parse_skill_metadata(skill_file)
            if metadata:
                self._skills[metadata.name] = metadata

    def _parse_skill_metadata(self, skill_path: Path) -> SkillMetadata | None:
        """Parse YAML frontmatter from a SKILL.md file.

        Args:
            skill_path: Path to the SKILL.md file

        Returns:
            SkillMetadata with name and description, or None if parsing fails
        """
        try:
            content = skill_path.read_text()

            # Parse YAML frontmatter (between --- delimiters)
            frontmatter_match = re.match(
                r'^---\s*\n(.*?)\n---\s*\n',
                content,
                re.DOTALL
            )

            if not frontmatter_match:
                return None

            frontmatter = yaml.safe_load(frontmatter_match.group(1))

            if not frontmatter or 'name' not in frontmatter:
                return None

            return SkillMetadata(
                name=frontmatter['name'],
                description=frontmatter.get('description', ''),
                path=skill_path,
            )
        except Exception:
            return None

    def get_skill_names(self) -> list[str]:
        """Get list of all discovered skill names."""
        return list(self._skills.keys())

    def get_skill_metadata(self, name: str) -> SkillMetadata | None:
        """Get metadata for a specific skill."""
        return self._skills.get(name)

    def get_all_metadata(self) -> list[SkillMetadata]:
        """Get metadata for all discovered skills."""
        return list(self._skills.values())

    def load_skill_content(self, name: str) -> SkillContent | None:
        """Load the full content of a skill.

        Args:
            name: The skill name to load

        Returns:
            SkillContent with full documentation, or None if not found
        """
        metadata = self._skills.get(name)
        if not metadata:
            return None

        try:
            full_content = metadata.path.read_text()

            # Remove YAML frontmatter for the content
            content_match = re.match(
                r'^---\s*\n.*?\n---\s*\n(.*)$',
                full_content,
                re.DOTALL
            )

            if content_match:
                content = content_match.group(1).strip()
            else:
                content = full_content

            return SkillContent(
                name=metadata.name,
                description=metadata.description,
                content=content,
                path=metadata.path,
            )
        except Exception:
            return None

    def get_skills_summary(self) -> str:
        """Get a formatted summary of all available skills.

        This is used in the agent's system prompt to inform it
        of available skills without loading full content.

        Returns:
            Formatted string listing all skills with descriptions
        """
        if not self._skills:
            return "No skills available."

        lines = ["Available Skills:"]
        for name, metadata in sorted(self._skills.items()):
            lines.append(f"- **{name}**: {metadata.description}")

        lines.append("")
        lines.append("Use `activate_skill(skill_name)` to load a skill's full documentation.")
        lines.append("Use `deactivate_skill(skill_name)` to unload a skill when done.")

        return "\n".join(lines)


# Module-level registry instance for convenience
_default_registry: SkillRegistry | None = None


def get_default_registry() -> SkillRegistry:
    """Get the default skill registry (singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry


def get_skills_summary() -> str:
    """Get summary of all available skills from the default registry."""
    return get_default_registry().get_skills_summary()


# =============================================================================
# Dynamic Instruction Provider (Claude Code-style approach)
# =============================================================================

def create_skill_instruction_provider(registry: SkillRegistry):
    """Create an instruction provider that injects active skills into system prompt.

    This is the key to ephemeral skill loading! The instruction provider is called
    on EVERY LLM request and returns content that goes into the system prompt.
    Since it reads from session state, skills can be activated/deactivated and
    the system prompt updates automatically.

    Unlike tool responses which persist in conversation history, the system prompt
    is rebuilt fresh each time - so deactivated skills truly disappear from context.

    Args:
        registry: The skill registry to load skills from

    Returns:
        An instruction provider function compatible with LlmAgent.instruction
    """
    def instruction_provider(ctx: ReadonlyContext) -> str:
        """Generate dynamic instructions based on active skills."""
        # Get active skills from session state
        active_skills: list[str] = ctx.state.get(ACTIVE_SKILLS_KEY, [])

        if not active_skills:
            return ""  # No active skills, no additional instructions

        # Build skill content section
        skill_sections = []
        for skill_name in active_skills:
            skill = registry.load_skill_content(skill_name)
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

    return instruction_provider


# =============================================================================
# Skill Activation/Deactivation Tools
# =============================================================================

def activate_skill(skill_name: str, tool_context: ToolContext) -> str:
    """Activate a skill to load its full documentation into context.

    When activated, the skill's content will be injected into the system prompt
    for all subsequent LLM calls. This is ephemeral - the content is NOT stored
    in conversation history and can be removed by calling deactivate_skill.

    Args:
        skill_name: Name of the skill to activate (e.g., 'bqml', 'bq_ai_operator')

    Returns:
        Confirmation message or error if skill not found
    """
    registry = get_default_registry()

    # Verify skill exists
    if skill_name not in registry.get_skill_names():
        available = ", ".join(registry.get_skill_names())
        return f"Skill '{skill_name}' not found. Available skills: {available}"

    # Get current active skills from state
    active_skills: list[str] = list(tool_context.state.get(ACTIVE_SKILLS_KEY, []))

    # Add skill if not already active
    if skill_name not in active_skills:
        active_skills.append(skill_name)
        tool_context.state[ACTIVE_SKILLS_KEY] = active_skills

        # Get skill metadata for confirmation
        metadata = registry.get_skill_metadata(skill_name)
        return f"Activated skill '{skill_name}': {metadata.description}\n\nThe skill documentation is now available in your context. Use deactivate_skill('{skill_name}') when done."
    else:
        return f"Skill '{skill_name}' is already active."


def deactivate_skill(skill_name: str, tool_context: ToolContext) -> str:
    """Deactivate a skill to remove its documentation from context.

    This removes the skill content from the system prompt, freeing up context
    space for other information. The skill can be reactivated later if needed.

    Args:
        skill_name: Name of the skill to deactivate

    Returns:
        Confirmation message
    """
    # Get current active skills from state
    active_skills: list[str] = list(tool_context.state.get(ACTIVE_SKILLS_KEY, []))

    if skill_name in active_skills:
        active_skills.remove(skill_name)
        tool_context.state[ACTIVE_SKILLS_KEY] = active_skills
        return f"Deactivated skill '{skill_name}'. Its documentation has been removed from context."
    else:
        return f"Skill '{skill_name}' is not currently active."


def list_active_skills(tool_context: ToolContext) -> str:
    """List all currently active skills.

    Returns:
        List of active skill names or message if none active
    """
    active_skills: list[str] = tool_context.state.get(ACTIVE_SKILLS_KEY, [])

    if not active_skills:
        return "No skills are currently active. Use activate_skill(skill_name) to load a skill."

    registry = get_default_registry()
    lines = ["Currently active skills:"]
    for name in active_skills:
        metadata = registry.get_skill_metadata(name)
        if metadata:
            lines.append(f"- **{name}**: {metadata.description}")
        else:
            lines.append(f"- **{name}**: (metadata not found)")

    return "\n".join(lines)


# =============================================================================
# Legacy load_skill function (for backward compatibility)
# =============================================================================

def load_skill(skill_name: str) -> str:
    """Load a skill's documentation (legacy function).

    Note: This returns the skill content as a string which will persist in
    conversation history. For ephemeral skill loading, use activate_skill()
    instead with the dynamic instruction provider.

    Args:
        skill_name: Name of the skill to load

    Returns:
        Full skill documentation
    """
    registry = get_default_registry()
    skill = registry.load_skill_content(skill_name)

    if skill is None:
        available = ", ".join(registry.get_skill_names())
        return f"Skill '{skill_name}' not found. Available skills: {available}"

    return f"""# Skill: {skill.name}

{skill.description}

---

{skill.content}
"""
