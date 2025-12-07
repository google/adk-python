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
3. Agent loads full skill content on-demand via load_skill tool
4. This reduces context usage while providing rich capabilities when needed
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    - Level 2: Full skill content (loaded on-demand via load_skill)

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

    def load_skill(self, name: str) -> SkillContent | None:
        """Load the full content of a skill (Level 2 disclosure).

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
        lines.append("Use `load_skill(skill_name)` to load full documentation for a skill.")

        return "\n".join(lines)


def create_load_skill_tool(registry: SkillRegistry) -> dict[str, Any]:
    """Create a load_skill tool function for the agent.

    This creates a callable tool that the agent can use to
    load full skill content when it determines a skill is relevant.

    Args:
        registry: The skill registry to load from

    Returns:
        A tool function that can be added to the agent
    """
    def load_skill(skill_name: str) -> str:
        """Load the full documentation for a skill.

        Use this tool when you need detailed instructions, examples,
        or patterns for a specific capability. Only load skills that
        are relevant to the current task.

        Args:
            skill_name: Name of the skill to load (e.g., 'bqml', 'bq_ai_operator')

        Returns:
            Full skill documentation including examples and patterns
        """
        skill = registry.load_skill(skill_name)
        if skill is None:
            available = ", ".join(registry.get_skill_names())
            return f"Skill '{skill_name}' not found. Available skills: {available}"

        return f"""# Skill: {skill.name}

{skill.description}

---

{skill.content}
"""

    return load_skill


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


def load_skill(skill_name: str) -> str:
    """Load a skill from the default registry.

    This is the function that should be added as a tool to the agent.
    """
    return create_load_skill_tool(get_default_registry())(skill_name)
