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

import logging
import pathlib
import re
from typing import Any
from typing import Callable
from typing import Optional
import unicodedata

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from ..features import FeatureName
from ..features import is_feature_enabled

_KEBAB_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SNAKE_OR_KEBAB_NAME_PATTERN = re.compile(
    r"^([a-z0-9]+(-[a-z0-9]+)*|[a-z0-9]+(_[a-z0-9]+)*)$"
)


class Frontmatter(BaseModel):
  """L1 skill content: metadata parsed from SKILL.md for skill discovery.

  Attributes:
      name: Skill name in kebab-case or snake_case (required).
      description: What the skill does and when the model should use it
        (required).
      license: License for the skill (optional).
      compatibility: Compatibility information for the skill (optional).
      allowed_tools: A space-delimited list of tools that are pre-approved to
        run (optional, experimental). Accepts both ``allowed_tools`` and the
        YAML-friendly ``allowed-tools`` key. For more details, see
        https://agentskills.io/specification#allowed-tools-field.
      metadata: Key-value pairs for client-specific properties (defaults to
        empty dict). For example, to include additional tools, use the
        ``adk_additional_tools`` key with a list of tools. Set
        ``adk_inject_state: true`` to enable ``{var}`` interpolation in the
        SKILL.md body when the skill is loaded via ``load_skill`` (same syntax
        as ``LlmAgent.instruction``). Each ``{var}`` is replaced with the
        matching value read from the invocation's session state; use ``{var?}``
        to substitute an empty string when the key is absent (instead of
        raising), and ``{artifact.name}`` to inject artifact contents.
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
  metadata: dict[str, Any] = {}

  @field_validator("metadata")
  @classmethod
  def _validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
    if "adk_additional_tools" in v:
      tools = v["adk_additional_tools"]
      if not isinstance(tools, list):
        raise ValueError("adk_additional_tools must be a list of strings")
    if "adk_inject_state" in v and not isinstance(v["adk_inject_state"], bool):
      raise ValueError("adk_inject_state must be a bool")
    return v

  @field_validator("name")
  @classmethod
  def _validate_name(cls, v: str) -> str:
    v = unicodedata.normalize("NFKC", v)
    if len(v) > 64:
      raise ValueError("name must be at most 64 characters")
    if is_feature_enabled(FeatureName.SNAKE_CASE_SKILL_NAME):
      pattern = _SNAKE_OR_KEBAB_NAME_PATTERN
      msg = (
          "name must be lowercase kebab-case (a-z, 0-9, hyphens) or"
          " snake_case (a-z, 0-9, underscores), with no leading, trailing,"
          " or consecutive delimiters. Mixing hyphens and underscores is"
          " not allowed."
      )
    else:
      pattern = _KEBAB_NAME_PATTERN
      msg = (
          "name must be lowercase kebab-case (a-z, 0-9,"
          " hyphens), with no leading, trailing, or"
          " consecutive delimiters"
      )
    if not pattern.match(v):
      raise ValueError(msg)
    return v

  @field_validator("description")
  @classmethod
  def _validate_description(cls, v: str) -> str:
    if not v:
      raise ValueError("description must not be empty")
    description_len = len(v)
    if description_len > 1024:
      raise ValueError(
          "description must be at most 1024 characters. Description length:"
          f" {description_len}"
      )
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


def _read_resource_file(path: pathlib.Path) -> str | bytes:
  """Reads a resource file as UTF-8 text or falls back to raw bytes."""
  try:
    return path.read_text(encoding="utf-8")
  except UnicodeDecodeError:
    return path.read_bytes()


class Resources(BaseModel):
  """L3 skill content: additional instructions, assets, and scripts.

  Attributes:
      references: Additional markdown files with instructions, workflows, or
        guidance.
      assets: Resource materials like database schemas, API documentation,
        templates, or examples.
      scripts: Executable scripts that can be run via bash.
  """

  references: dict[str, str | bytes] = {}
  assets: dict[str, str | bytes] = {}
  scripts: dict[str, Script] = {}

  _reference_paths: dict[str, pathlib.Path] = {}
  _asset_paths: dict[str, pathlib.Path] = {}
  _script_paths: dict[str, pathlib.Path] = {}

  def __getattribute__(self, name: str) -> Any:
    if name in ("references", "assets", "scripts"):
      private = object.__getattribute__(self, "__pydantic_private__") or {}
      if name == "references" and private.get("_reference_paths"):
        for key in list(private["_reference_paths"]):
          self.get_reference(key)
      elif name == "assets" and private.get("_asset_paths"):
        for key in list(private["_asset_paths"]):
          self.get_asset(key)
      elif name == "scripts" and private.get("_script_paths"):
        for key in list(private["_script_paths"]):
          self.get_script(key)
    return super().__getattribute__(name)

  def get_reference(self, reference_id: str) -> Optional[str | bytes]:
    """Get content of a reference file.

    Args:
        reference_id: Unique path or name of the reference file.

    Returns:
        Reference content as string or bytes, or None if not found
    """
    refs = object.__getattribute__(self, "__dict__")["references"]
    if reference_id in refs:
      return refs[reference_id]
    path = self._reference_paths.pop(reference_id, None)
    if path is not None:
      refs[reference_id] = _read_resource_file(path)
    return refs.get(reference_id)

  def get_asset(self, asset_id: str) -> Optional[str | bytes]:
    """Get content of an asset file.

    Args:
        asset_id: Unique path or name of the asset file.

    Returns:
        Asset content as string or bytes, or None if not found
    """
    assets = object.__getattribute__(self, "__dict__")["assets"]
    if asset_id in assets:
      return assets[asset_id]
    path = self._asset_paths.pop(asset_id, None)
    if path is not None:
      assets[asset_id] = _read_resource_file(path)
    return assets.get(asset_id)

  def get_script(self, script_id: str) -> Optional[Script]:
    """Get content of a script file.

    Args:
        script_id: Unique path or name of the script file.

    Returns:
        Script object, or None if not found
    """
    scripts = object.__getattribute__(self, "__dict__")["scripts"]
    if script_id in scripts:
      return scripts[script_id]
    path = self._script_paths.pop(script_id, None)
    if path is not None:
      try:
        scripts[script_id] = Script(src=path.read_text(encoding="utf-8"))
      except UnicodeDecodeError:
        logging.warning("Skipping non-UTF-8 skill script '%s'.", script_id)
        return None
    return scripts.get(script_id)

  def list_references(self) -> list[str]:
    """List all available reference paths."""
    refs = object.__getattribute__(self, "__dict__")["references"]
    return list(dict.fromkeys([*refs, *self._reference_paths]))

  def list_assets(self) -> list[str]:
    """List all available asset paths."""
    assets = object.__getattribute__(self, "__dict__")["assets"]
    return list(dict.fromkeys([*assets, *self._asset_paths]))

  def list_scripts(self) -> list[str]:
    """List all available script paths."""
    for key in list(self._script_paths):
      self.get_script(key)
    return list(object.__getattribute__(self, "__dict__")["scripts"].keys())


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

  _uri: Optional[str] = None
  """Location the skill was loaded from, used for telemetry.
  Should be compliant with RFC 3986.
  """
  _instructions_loader: Optional[Callable[[], str]] = None

  def __getattribute__(self, name: str) -> Any:
    if name == "instructions":
      private = object.__getattribute__(self, "__pydantic_private__") or {}
      loader = private.get("_instructions_loader")
      if loader is not None:
        d = object.__getattribute__(self, "__dict__")
        if not d.get("instructions"):
          d["instructions"] = loader()
        private["_instructions_loader"] = None
    return super().__getattribute__(name)

  @property
  def name(self) -> str:
    """Convenience property to access skill name."""
    return self.frontmatter.name

  @property
  def description(self) -> str:
    """Convenience property to access skill description."""
    return self.frontmatter.description
