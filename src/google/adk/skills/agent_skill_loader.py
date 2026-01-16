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

"""AgentSkillLoader for discovering and loading Agent Skills standard skills.

This module provides utilities for discovering skills from directories
following the Agent Skills standard (https://agentskills.io).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from ..utils.feature_decorator import experimental
from .markdown_skill import MarkdownSkill

if TYPE_CHECKING:
  from .skill_manager import SkillsManager

logger = logging.getLogger("google_adk." + __name__)


@experimental
class AgentSkillLoader:
  """Discovers and loads Agent Skills standard skills.

  Implements progressive disclosure by loading only metadata initially,
  with full content loaded on-demand when skills are activated.

  This loader scans directories for folders containing SKILL.md files
  and creates MarkdownSkill instances that can be registered with
  a SkillsManager.

  Example:
    ```python
    loader = AgentSkillLoader()

    # Discover skills from multiple directories
    loader.add_skill_directory("/path/to/skills")
    loader.add_skill_directory("/path/to/custom-skills")

    # Get all discovered skills (Stage 1 - metadata only)
    skills = loader.get_all_skills()

    # Register with SkillsManager
    from google.adk.skills import SkillsManager
    manager = SkillsManager()
    loader.register_all(manager)

    # Generate discovery prompt for LLM
    prompt = loader.generate_discovery_prompt()
    ```
  """

  def __init__(self, validate_names: bool = True):
    """Initialize the AgentSkillLoader.

    Args:
      validate_names: Whether to validate that skill names match
        their directory names (recommended for consistency).
    """
    self._skill_directories: List[Path] = []
    self._discovered_skills: Dict[str, MarkdownSkill] = {}
    self._load_errors: Dict[str, str] = {}
    self._validate_names = validate_names

  def add_skill_directory(self, path: Union[str, Path]) -> int:
    """Add a directory to scan for skills.

    Scans the directory for subdirectories containing SKILL.md files
    and loads them as MarkdownSkill instances (Stage 1 only).

    Args:
      path: Directory containing skill folders.

    Returns:
      Number of skills discovered in this directory.

    Raises:
      FileNotFoundError: If directory doesn't exist.
      ValueError: If path is not a directory.
    """
    dir_path = Path(path).resolve()

    if not dir_path.exists():
      raise FileNotFoundError(f"Skill directory not found: {path}")

    if not dir_path.is_dir():
      raise ValueError(f"Path is not a directory: {path}")

    self._skill_directories.append(dir_path)
    return self._discover_skills_in_directory(dir_path)

  def add_skill(self, skill_dir: Union[str, Path]) -> bool:
    """Add a single skill from a directory.

    Args:
      skill_dir: Path to a single skill directory containing SKILL.md.

    Returns:
      True if skill was loaded successfully, False otherwise.
    """
    skill_path = Path(skill_dir).resolve()

    if not skill_path.is_dir():
      self._load_errors[str(skill_path)] = "Not a directory"
      return False

    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
      self._load_errors[str(skill_path)] = "SKILL.md not found"
      return False

    try:
      skill = MarkdownSkill.from_directory(
          skill_path, validate_name=self._validate_names
      )
      self._discovered_skills[skill.name] = skill
      logger.info("Loaded skill: %s from %s", skill.name, skill_path)
      return True
    except Exception as e:
      self._load_errors[str(skill_path)] = str(e)
      logger.warning("Failed to load skill from %s: %s", skill_path, e)
      return False

  def _discover_skills_in_directory(self, dir_path: Path) -> int:
    """Discover all skills in a directory.

    Args:
      dir_path: Directory to scan.

    Returns:
      Number of skills discovered.
    """
    count = 0

    for item in sorted(dir_path.iterdir()):
      if not item.is_dir():
        continue

      # Skip hidden directories
      if item.name.startswith("."):
        continue

      skill_md = item / "SKILL.md"
      if not skill_md.exists():
        continue

      try:
        skill = MarkdownSkill.from_directory(
            item, validate_name=self._validate_names
        )

        # Check for duplicate names
        if skill.name in self._discovered_skills:
          existing_path = self._discovered_skills[skill.name].skill_path
          logger.warning(
              "Duplicate skill name '%s': %s shadows %s",
              skill.name,
              item,
              existing_path,
          )

        self._discovered_skills[skill.name] = skill
        count += 1
        logger.info("Discovered skill: %s", skill.name)

      except Exception as e:
        self._load_errors[str(item)] = str(e)
        logger.warning("Failed to load skill from %s: %s", item, e)

    return count

  def get_skill(self, name: str) -> Optional[MarkdownSkill]:
    """Get a discovered skill by name.

    Args:
      name: The skill name.

    Returns:
      MarkdownSkill instance or None if not found.
    """
    return self._discovered_skills.get(name)

  def get_all_skills(self) -> List[MarkdownSkill]:
    """Get all discovered skills.

    Returns:
      List of all MarkdownSkill instances.
    """
    return list(self._discovered_skills.values())

  def get_skill_names(self) -> List[str]:
    """Get names of all discovered skills.

    Returns:
      Sorted list of skill names.
    """
    return sorted(self._discovered_skills.keys())

  def get_load_errors(self) -> Dict[str, str]:
    """Get any errors encountered during discovery.

    Returns:
      Dictionary mapping paths to error messages.
    """
    return self._load_errors.copy()

  def has_skill(self, name: str) -> bool:
    """Check if a skill is loaded.

    Args:
      name: The skill name.

    Returns:
      True if skill is loaded.
    """
    return name in self._discovered_skills

  def clear(self) -> None:
    """Clear all discovered skills and errors."""
    self._discovered_skills.clear()
    self._load_errors.clear()
    self._skill_directories.clear()

  def register_all(self, manager: "SkillsManager") -> int:
    """Register all discovered skills with a SkillsManager.

    Args:
      manager: The SkillsManager to register skills with.

    Returns:
      Number of skills successfully registered.
    """
    count = 0
    for skill in self._discovered_skills.values():
      try:
        manager.register_skill(skill)
        count += 1
      except ValueError as e:
        logger.warning("Failed to register skill %s: %s", skill.name, e)

    return count

  def generate_discovery_prompt(self, include_resources: bool = True) -> str:
    """Generate XML prompt with skill metadata for LLM discovery.

    This implements Stage 1 of progressive disclosure - only
    name and description are included, keeping context minimal
    (~100 tokens per skill).

    Args:
      include_resources: Whether to include has_scripts/has_references hints.

    Returns:
      XML-formatted string with available skills.
    """
    if not self._discovered_skills:
      return "<available_skills></available_skills>"

    skills_xml = []
    for skill in sorted(self._discovered_skills.values(), key=lambda s: s.name):
      lines = [
          "  <skill>",
          f"    <name>{self._escape_xml(skill.name)}</name>",
          (
              f"    <description>{self._escape_xml(skill.description)}</description>"
          ),
      ]

      if include_resources:
        lines.append(f"    <has_scripts>{skill.has_scripts()}</has_scripts>")
        lines.append(
            f"    <has_references>{skill.has_references()}</has_references>"
        )

      lines.append("  </skill>")
      skills_xml.append("\n".join(lines))

    return (
        "<available_skills>\n" + "\n".join(skills_xml) + "\n</available_skills>"
    )

  def generate_activation_prompt(self, skill_name: str) -> Optional[str]:
    """Generate full skill prompt for activation (Stage 2).

    Loads the full SKILL.md instructions and includes information
    about available resources.

    Args:
      skill_name: Name of the skill to activate.

    Returns:
      Full skill instructions or None if skill not found.
    """
    skill = self._discovered_skills.get(skill_name)
    if not skill:
      return None

    instructions = skill.get_instructions()
    sections = [f"# Skill: {skill.name}", "", instructions]

    # Add resources section
    resources = []
    scripts = skill.list_scripts()
    if scripts:
      resources.append(f"**Available scripts:** {', '.join(scripts)}")

    refs = skill.list_references()
    if refs:
      resources.append(f"**Available references:** {', '.join(refs)}")

    assets = skill.list_assets()
    if assets:
      shown = assets[:10]
      if len(assets) > 10:
        shown.append(f"... and {len(assets) - 10} more")
      resources.append(f"**Available assets:** {', '.join(shown)}")

    if resources:
      sections.append("")
      sections.append("## Available Resources")
      sections.extend(f"- {r}" for r in resources)

    # Add compatibility info
    if skill.skill_metadata.compatibility:
      sections.append("")
      sections.append("## Requirements")
      sections.append(skill.skill_metadata.compatibility)

    return "\n".join(sections)

  def generate_summary(self) -> str:
    """Generate a human-readable summary of loaded skills.

    Returns:
      Formatted summary string.
    """
    lines = [
        f"Agent Skills Loader Summary",
        f"=" * 40,
        f"Directories scanned: {len(self._skill_directories)}",
        f"Skills discovered: {len(self._discovered_skills)}",
        f"Load errors: {len(self._load_errors)}",
        "",
    ]

    if self._discovered_skills:
      lines.append("Discovered Skills:")
      for name in sorted(self._discovered_skills.keys()):
        skill = self._discovered_skills[name]
        desc = (
            skill.description[:50] + "..."
            if len(skill.description) > 50
            else skill.description
        )
        lines.append(f"  - {name}: {desc}")

    if self._load_errors:
      lines.append("")
      lines.append("Load Errors:")
      for path, error in self._load_errors.items():
        lines.append(f"  - {path}: {error}")

    return "\n".join(lines)

  @staticmethod
  def _escape_xml(text: str) -> str:
    """Escape special XML characters.

    Args:
      text: Text to escape.

    Returns:
      XML-safe string.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

  def __len__(self) -> int:
    """Return number of discovered skills."""
    return len(self._discovered_skills)

  def __contains__(self, name: str) -> bool:
    """Check if a skill name is loaded."""
    return name in self._discovered_skills

  def __iter__(self):
    """Iterate over discovered skills."""
    return iter(self._discovered_skills.values())
