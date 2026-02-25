# Copyright 2026 Google LLC
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

"""Data models for Agent Skills."""

from __future__ import annotations

import re
from typing import Optional
import unicodedata

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class Frontmatter(BaseModel):
  """L1 skill content: metadata parsed from SKILL.md frontmatter for skill discovery.

  Attributes:
      name: Skill name in kebab-case (required).
      description: What the skill does and when the model should use it
        (required).
      license: License for the skill (optional).
      compatibility: Compatibility information for the skill (optional).
      allowed_tools: Tool patterns the skill requires (optional, experimental).
        Accepts both ``allowed_tools`` and the YAML-friendly ``allowed-tools``
        key.
      metadata: Key-value pairs for client-specific properties (defaults to
        empty dict).
  """

  model_config = ConfigDict(
      extra="allow",
      populate_by_name=True,
  )

  name: str
  description: str
  license: Optional[str] = None
  compatibility: Optional[str] = None
  allowed_tools: Optional[str] = Field(
      default=None,
      alias="allowed-tools",
      serialization_alias="allowed-tools",
  )
  metadata: dict[str, str] = {}

  @field_validator("name")
  @classmethod
  def _validate_name(cls, v: str) -> str:
    v = unicodedata.normalize("NFKC", v)
    if len(v) > 64:
      raise ValueError("name must be at most 64 characters")
    if not _NAME_PATTERN.match(v):
      raise ValueError(
          "name must be lowercase kebab-case (a-z, 0-9, hyphens),"
          " with no leading, trailing, or consecutive hyphens"
      )
    return v

  @field_validator("description")
  @classmethod
  def _validate_description(cls, v: str) -> str:
    if not v:
      raise ValueError("description must not be empty")
    if len(v) > 1024:
      raise ValueError("description must be at most 1024 characters")
    return v

  @field_validator("compatibility")
  @classmethod
  def _validate_compatibility(cls, v: Optional[str]) -> Optional[str]:
    if v is not None and len(v) > 500:
      raise ValueError("compatibility must be at most 500 characters")
    return v


class Script(BaseModel):
  """Wrapper for script content."""

  src: str

  def __str__(self) -> str:
    """Returns the string representation of the script content.

    This ensures that any script type can be converted to a string, which is
    useful for including the script in prompts or saving it to the file system.
    """
    return self.src


class Resources(BaseModel):
  """L3 skill content: additional instructions, assets, and scripts, loaded as needed.

  Attributes:
      references: Additional markdown files with instructions, workflows, or
        guidance.
      assets: Resource materials like database schemas, API documentation,
        templates, or examples.
      scripts: Executable scripts that can be run via bash.
  """

  references: dict[str, str] = {}
  assets: dict[str, str] = {}
  scripts: dict[str, Script] = {}

  def get_reference(self, reference_id: str) -> Optional[str]:
    """Get content of a reference file.

    Args:
        reference_id: Unique path or name of the reference file.

    Returns:
        Reference content as string, or None if not found
    """
    return self.references.get(reference_id)

  def get_asset(self, asset_id: str) -> Optional[str]:
    """Get content of an asset file.

    Args:
        asset_id: Unique path or name of the asset file.

    Returns:
        Asset content as string, or None if not found
    """
    return self.assets.get(asset_id)

  def get_script(self, script_id: str) -> Optional[Script]:
    """Get content of a script file.

    Args:
        script_id: Unique path or name of the script file.

    Returns:
        Script object, or None if not found
    """
    return self.scripts.get(script_id)

  def list_references(self) -> list[str]:
    """List all available reference paths."""
    return list(self.references.keys())

  def list_assets(self) -> list[str]:
    """List all available asset paths."""
    return list(self.assets.keys())

  def list_scripts(self) -> list[str]:
    """List all available script paths."""
    return list(self.scripts.keys())


class Skill(BaseModel):
  """Complete skill representation including frontmatter, instructions, and resources.

  A skill combines:
  - L1: Frontmatter for discovery (name, description).
  - L2: Instructions from SKILL.md body, loaded when skill is triggered.
  - L3: Resources including additional instructions, assets, and scripts,
  loaded as needed.

  Attributes:
      frontmatter: Parsed skill frontmatter from SKILL.md.
      instructions: L2 skill content: markdown instruction from SKILL.md body.
      resources: L3 skill content: additional instructions, assets, and scripts.
  """

  frontmatter: Frontmatter
  instructions: str
  resources: Resources = Resources()

  @property
  def name(self) -> str:
    """Convenience property to access skill name."""
    return self.frontmatter.name

  @property
  def description(self) -> str:
    """Convenience property to access skill description."""
    return self.frontmatter.description
