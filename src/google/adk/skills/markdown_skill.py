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

"""MarkdownSkill implementation for Agent Skills standard SKILL.md format.

This module provides support for loading skills defined using the Agent Skills
standard (https://agentskills.io), enabling skills built with SKILL.md files
to be used directly as ADK Skills.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr

from ..utils.feature_decorator import experimental
from .base_skill import BaseSkill
from .base_skill import SkillConfig


class MarkdownSkillMetadata(BaseModel):
  """Metadata extracted from SKILL.md frontmatter.

  Follows the Agent Skills standard specification for frontmatter fields.
  """

  model_config = ConfigDict(extra="allow")

  name: str = Field(
      description="Unique identifier for the skill (max 64 chars).",
  )
  description: str = Field(
      description="LLM-readable description of what the skill does.",
  )
  license: Optional[str] = Field(
      default=None,
      description="License name or reference to bundled license file.",
  )
  compatibility: Optional[str] = Field(
      default=None,
      description="Environment requirements (max 500 chars).",
  )
  metadata: Dict[str, Any] = Field(
      default_factory=dict,
      description="Additional metadata (author, version, etc.).",
  )
  allowed_tools: Optional[str] = Field(
      default=None,
      description="Space-delimited list of pre-approved tools.",
      alias="allowed-tools",
  )

  # ADK-specific extensions
  adk: Optional[Dict[str, Any]] = Field(
      default=None,
      description="ADK-specific configuration extensions.",
  )


@experimental
class MarkdownSkill(BaseSkill):
  """Skill loaded from Agent Skills standard SKILL.md format.

  Supports progressive disclosure with three loading stages:
  - Stage 1 (Discovery): Only name and description loaded (~100 tokens)
  - Stage 2 (Activation): Full SKILL.md content loaded (~2000-5000 tokens)
  - Stage 3 (Execution): Scripts and references loaded on-demand

  Example:
    ```python
    # Load a skill from directory
    skill = MarkdownSkill.from_directory("/path/to/pdf-processing")

    # Stage 1: Metadata only (loaded at construction)
    print(skill.name)         # "pdf-processing"
    print(skill.description)  # "Extract text from PDFs..."

    # Stage 2: Full instructions (loaded on demand)
    instructions = skill.get_instructions()

    # Stage 3: Access scripts/references (loaded on demand)
    script = skill.get_script("extract_text.py")
    reference = skill.get_reference("FORMS.md")
    ```
  """

  model_config = ConfigDict(
      extra="forbid",
      arbitrary_types_allowed=True,
  )

  # Path to the skill directory
  skill_path: Path = Field(
      description="Absolute path to the skill directory.",
  )

  # Parsed frontmatter metadata
  skill_metadata: MarkdownSkillMetadata = Field(
      description="Parsed metadata from SKILL.md frontmatter.",
  )

  # Private attributes for caching (not part of model)
  _instructions_cache: Optional[str] = PrivateAttr(default=None)
  _scripts_cache: Dict[str, str] = PrivateAttr(default_factory=dict)
  _references_cache: Dict[str, str] = PrivateAttr(default_factory=dict)
  _current_stage: int = PrivateAttr(default=1)

  @classmethod
  def from_directory(
      cls,
      skill_dir: Union[str, Path],
      validate_name: bool = True,
  ) -> "MarkdownSkill":
    """Load a skill from a directory containing SKILL.md.

    Args:
      skill_dir: Path to the skill directory.
      validate_name: Whether to validate that name matches directory name.

    Returns:
      MarkdownSkill instance with Stage 1 (metadata) loaded.

    Raises:
      FileNotFoundError: If SKILL.md doesn't exist.
      ValueError: If frontmatter is invalid or name doesn't match directory.
    """
    skill_path = Path(skill_dir).resolve()
    skill_md_path = skill_path / "SKILL.md"

    if not skill_md_path.exists():
      raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

    # Parse only frontmatter for Stage 1
    content = skill_md_path.read_text(encoding="utf-8")
    metadata = cls._parse_frontmatter(content)

    # Validate name matches directory (optional but recommended)
    if validate_name and metadata.name != skill_path.name:
      raise ValueError(
          f"Skill name '{metadata.name}' must match "
          f"directory name '{skill_path.name}'"
      )

    # Build configuration from metadata
    config = cls._build_config(metadata)

    # Determine allowed_callers
    allowed_callers = ["code_execution_20250825"]
    if metadata.adk and "allowed_callers" in metadata.adk:
      allowed_callers = metadata.adk["allowed_callers"]

    return cls(
        name=metadata.name,
        description=metadata.description,
        skill_path=skill_path,
        skill_metadata=metadata,
        config=config,
        allowed_callers=allowed_callers,
    )

  @staticmethod
  def _parse_frontmatter(content: str) -> MarkdownSkillMetadata:
    """Parse YAML frontmatter from SKILL.md content.

    Args:
      content: Full content of the SKILL.md file.

    Returns:
      MarkdownSkillMetadata with parsed values.

    Raises:
      ValueError: If frontmatter is missing or invalid.
    """
    import yaml

    if not content.startswith("---"):
      raise ValueError("SKILL.md must start with YAML frontmatter (---)")

    # Split frontmatter from body
    parts = content.split("---", 2)
    if len(parts) < 3:
      raise ValueError("Invalid frontmatter format. Expected '---' delimiters.")

    frontmatter_yaml = parts[1].strip()
    if not frontmatter_yaml:
      raise ValueError("Empty frontmatter")

    try:
      frontmatter = yaml.safe_load(frontmatter_yaml)
    except yaml.YAMLError as e:
      raise ValueError(f"Invalid YAML in frontmatter: {e}") from e

    if not isinstance(frontmatter, dict):
      raise ValueError("Frontmatter must be a YAML mapping")

    # Validate required fields
    if "name" not in frontmatter:
      raise ValueError("Missing required field 'name' in frontmatter")
    if "description" not in frontmatter:
      raise ValueError("Missing required field 'description' in frontmatter")

    return MarkdownSkillMetadata(**frontmatter)

  @staticmethod
  def _build_config(metadata: MarkdownSkillMetadata) -> SkillConfig:
    """Build SkillConfig from metadata.

    Args:
      metadata: Parsed frontmatter metadata.

    Returns:
      SkillConfig with appropriate settings.
    """
    config_kwargs = {}

    # Check for ADK-specific config
    if metadata.adk and "config" in metadata.adk:
      adk_config = metadata.adk["config"]
      if "max_parallel_calls" in adk_config:
        config_kwargs["max_parallel_calls"] = adk_config["max_parallel_calls"]
      if "timeout_seconds" in adk_config:
        config_kwargs["timeout_seconds"] = adk_config["timeout_seconds"]
      if "allow_network" in adk_config:
        config_kwargs["allow_network"] = adk_config["allow_network"]
      if "memory_limit_mb" in adk_config:
        config_kwargs["memory_limit_mb"] = adk_config["memory_limit_mb"]

    # Parse compatibility for network requirements
    if metadata.compatibility:
      compat_lower = metadata.compatibility.lower()
      if "network" in compat_lower or "internet" in compat_lower:
        config_kwargs.setdefault("allow_network", True)

    return SkillConfig(**config_kwargs)

  # ===========================================================================
  # Progressive Disclosure Implementation
  # ===========================================================================

  def get_instructions(self) -> str:
    """Get full SKILL.md instructions (Stage 2).

    Loads and caches the markdown body on first access.

    Returns:
      The markdown content after the frontmatter.
    """
    if self._instructions_cache is None:
      skill_md_path = self.skill_path / "SKILL.md"
      content = skill_md_path.read_text(encoding="utf-8")

      # Extract body after frontmatter
      parts = content.split("---", 2)
      self._instructions_cache = parts[2].strip() if len(parts) > 2 else ""
      self._current_stage = max(self._current_stage, 2)

    return self._instructions_cache

  def get_script(self, script_name: str) -> Optional[str]:
    """Get script content from scripts/ directory (Stage 3).

    Args:
      script_name: Name of the script file.

    Returns:
      Script content or None if not found.
    """
    if script_name not in self._scripts_cache:
      script_path = self.skill_path / "scripts" / script_name
      if script_path.exists() and script_path.is_file():
        self._scripts_cache[script_name] = script_path.read_text(
            encoding="utf-8"
        )
        self._current_stage = 3
      else:
        return None

    return self._scripts_cache.get(script_name)

  def get_script_path(self, script_name: str) -> Optional[Path]:
    """Get absolute path to a script file.

    Args:
      script_name: Name of the script file.

    Returns:
      Absolute Path or None if not found.
    """
    script_path = self.skill_path / "scripts" / script_name
    if script_path.exists() and script_path.is_file():
      return script_path
    return None

  def get_reference(self, ref_name: str) -> Optional[str]:
    """Get reference content from references/ directory (Stage 3).

    Args:
      ref_name: Name of the reference file.

    Returns:
      Reference content or None if not found.
    """
    if ref_name not in self._references_cache:
      ref_path = self.skill_path / "references" / ref_name
      if ref_path.exists() and ref_path.is_file():
        self._references_cache[ref_name] = ref_path.read_text(encoding="utf-8")
        self._current_stage = 3
      else:
        return None

    return self._references_cache.get(ref_name)

  def get_asset_path(self, asset_name: str) -> Optional[Path]:
    """Get absolute path to an asset file (Stage 3).

    Args:
      asset_name: Relative path within assets/ directory.

    Returns:
      Absolute Path or None if not found.
    """
    asset_path = self.skill_path / "assets" / asset_name
    if asset_path.exists():
      self._current_stage = 3
      return asset_path
    return None

  def list_scripts(self) -> List[str]:
    """List available scripts in the skill.

    Returns:
      List of script file names.
    """
    scripts_dir = self.skill_path / "scripts"
    if scripts_dir.exists() and scripts_dir.is_dir():
      return sorted([f.name for f in scripts_dir.iterdir() if f.is_file()])
    return []

  def list_references(self) -> List[str]:
    """List available references in the skill.

    Returns:
      List of reference file names.
    """
    refs_dir = self.skill_path / "references"
    if refs_dir.exists() and refs_dir.is_dir():
      return sorted([f.name for f in refs_dir.iterdir() if f.is_file()])
    return []

  def list_assets(self) -> List[str]:
    """List available assets in the skill.

    Returns:
      List of relative paths to asset files.
    """
    assets_dir = self.skill_path / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
      return sorted([
          str(f.relative_to(assets_dir))
          for f in assets_dir.rglob("*")
          if f.is_file()
      ])
    return []

  @property
  def current_stage(self) -> int:
    """Get the current progressive disclosure stage.

    Returns:
      1 for Discovery, 2 for Activation, 3 for Execution.
    """
    return self._current_stage

  def has_scripts(self) -> bool:
    """Check if the skill has any scripts."""
    scripts_dir = self.skill_path / "scripts"
    return scripts_dir.exists() and any(scripts_dir.iterdir())

  def has_references(self) -> bool:
    """Check if the skill has any references."""
    refs_dir = self.skill_path / "references"
    return refs_dir.exists() and any(refs_dir.iterdir())

  def has_assets(self) -> bool:
    """Check if the skill has any assets."""
    assets_dir = self.skill_path / "assets"
    return assets_dir.exists() and any(assets_dir.rglob("*"))

  # ===========================================================================
  # BaseSkill Abstract Method Implementations
  # ===========================================================================

  def get_tool_declarations(self) -> List[dict[str, Any]]:
    """Return tool declarations extracted from SKILL.md.

    Generates declarations for:
    - Script-based tools (from scripts/ directory)
    - Reference loading tools (from references/ directory)
    - ADK-specific tool declarations (from frontmatter)

    Returns:
      List of tool declaration dictionaries.
    """
    declarations = []

    # Check for ADK-specific tool declarations in frontmatter
    if self.skill_metadata.adk and "tools" in self.skill_metadata.adk:
      for tool_def in self.skill_metadata.adk["tools"]:
        declarations.append({
            "name": tool_def.get("name", "unknown"),
            "description": tool_def.get("description", ""),
            "parameters": tool_def.get("parameters", {}),
        })
      return declarations

    # Auto-generate declarations from scripts
    for script_name in self.list_scripts():
      script_path = self.skill_path / "scripts" / script_name
      docstring = self._extract_script_docstring(script_path)

      # Create safe tool name
      safe_name = script_name.replace(".", "_").replace("-", "_")

      declarations.append({
          "name": f"run_{safe_name}",
          "description": docstring or f"Execute {script_name}",
          "parameters": {"args": "Command-line arguments for the script"},
      })

    # Add reference loading declarations
    for ref_name in self.list_references():
      safe_name = ref_name.replace(".", "_").replace("-", "_")
      declarations.append({
          "name": f"load_{safe_name}",
          "description": f"Load reference document: {ref_name}",
      })

    return declarations

  def get_orchestration_template(self) -> str:
    """Return example orchestration code for this skill.

    Generates a template based on available scripts and tools.

    Returns:
      A Python async function as a string showing example usage.
    """
    safe_name = self.name.replace("-", "_").replace(".", "_")
    scripts = self.list_scripts()

    if not scripts:
      return f'''async def use_{safe_name}(tools):
    """Example orchestration for {self.name} skill.

    This skill provides instructions but no bundled scripts.
    Follow the instructions in the SKILL.md file.
    """
    return {{"status": "ready", "skill": "{self.name}"}}
'''

    # Generate script calls for first few scripts
    script_lines = []
    for i, script in enumerate(scripts[:3]):
      safe_script = script.replace(".", "_").replace("-", "_")
      script_lines.append(
          f'    result_{i} = await tools.run_{safe_script}(args="")'
      )

    script_calls = "\n".join(script_lines)
    result_list = ", ".join(f"result_{i}" for i in range(min(3, len(scripts))))

    return f'''async def use_{safe_name}(tools):
    """Example orchestration for {self.name} skill.

    Available scripts: {", ".join(scripts)}
    """
{script_calls}
    return {{"results": [{result_list}]}}
'''

  def get_skill_prompt(self) -> str:
    """Generate LLM-friendly skill description with progressive detail.

    Returns:
      Formatted string describing the skill and its capabilities.
    """
    base_prompt = super().get_skill_prompt()

    # Add available resources
    resources = []

    scripts = self.list_scripts()
    if scripts:
      resources.append(f"Scripts: {', '.join(scripts)}")

    refs = self.list_references()
    if refs:
      resources.append(f"References: {', '.join(refs)}")

    assets = self.list_assets()
    if assets:
      # Only show first few assets to avoid overwhelming
      shown_assets = assets[:5]
      if len(assets) > 5:
        shown_assets.append(f"... and {len(assets) - 5} more")
      resources.append(f"Assets: {', '.join(shown_assets)}")

    if resources:
      base_prompt += "\n\nAvailable resources:\n" + "\n".join(
          f"  - {r}" for r in resources
      )

    # Add compatibility info if present
    if self.skill_metadata.compatibility:
      base_prompt += f"\n\nRequirements: {self.skill_metadata.compatibility}"

    return base_prompt

  @staticmethod
  def _extract_script_docstring(script_path: Path) -> Optional[str]:
    """Extract docstring from a Python script.

    Args:
      script_path: Path to the script file.

    Returns:
      First line of docstring or None if not found/not Python.
    """
    if script_path.suffix != ".py":
      # For non-Python scripts, try to extract from comments
      if script_path.suffix in (".sh", ".bash"):
        return MarkdownSkill._extract_shell_description(script_path)
      return None

    try:
      content = script_path.read_text(encoding="utf-8")
      # Try to extract module docstring
      # Handle both """ and ''' styles
      for quote in ['"""', "'''"]:
        pattern = rf"^{quote}(.+?){quote}"
        match = re.match(pattern, content, re.DOTALL)
        if match:
          docstring = match.group(1).strip()
          # Return first line only
          return docstring.split("\n")[0].strip()
    except Exception:
      pass

    return None

  @staticmethod
  def _extract_shell_description(script_path: Path) -> Optional[str]:
    """Extract description from shell script comments.

    Args:
      script_path: Path to the shell script.

    Returns:
      Description from comment or None if not found.
    """
    try:
      content = script_path.read_text(encoding="utf-8")
      lines = content.split("\n")

      for line in lines:
        line = line.strip()
        # Skip shebang and empty lines
        if line.startswith("#!") or not line:
          continue
        # Found a comment line
        if line.startswith("#"):
          return line[1:].strip()
        # Found non-comment, stop looking
        break
    except Exception:
      pass

    return None
