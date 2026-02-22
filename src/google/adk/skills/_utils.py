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

"""Utility functions for Agent Skills."""

from __future__ import annotations

import pathlib
from typing import Union

import yaml

from . import models

_ALLOWED_FRONTMATTER_KEYS = frozenset({
    "name",
    "description",
    "license",
    "allowed-tools",
    "allowed_tools",
    "metadata",
    "compatibility",
})


def _load_dir(directory: pathlib.Path) -> dict[str, str]:
  """Recursively load files from a directory into a dictionary.

  Args:
    directory: Path to the directory to load.

  Returns:
    Dictionary mapping relative file paths to their string content.
  """
  files = {}
  if directory.exists() and directory.is_dir():
    for file_path in directory.rglob("*"):
      if "__pycache__" in file_path.parts:
        continue
      if file_path.is_file():
        relative_path = file_path.relative_to(directory)
        try:
          files[str(relative_path)] = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
          # Binary files or non-UTF-8 files are skipped for text content.
          continue
  return files


def _parse_skill_md(
    skill_dir: pathlib.Path,
) -> tuple[dict, str, pathlib.Path]:
  """Parse SKILL.md from a skill directory.

  Args:
    skill_dir: Resolved path to the skill directory.

  Returns:
    Tuple of (parsed_frontmatter_dict, body_string, skill_md_path).

  Raises:
    FileNotFoundError: If the directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid.
  """
  if not skill_dir.is_dir():
    raise FileNotFoundError(f"Skill directory '{skill_dir}' not found.")

  skill_md = None
  for name in ("SKILL.md", "skill.md"):
    path = skill_dir / name
    if path.exists():
      skill_md = path
      break

  if skill_md is None:
    raise FileNotFoundError(f"SKILL.md not found in '{skill_dir}'.")

  content = skill_md.read_text(encoding="utf-8")
  if not content.startswith("---"):
    raise ValueError("SKILL.md must start with YAML frontmatter (---)")

  parts = content.split("---", 2)
  if len(parts) < 3:
    raise ValueError("SKILL.md frontmatter not properly closed with ---")

  frontmatter_str = parts[1]
  body = parts[2].strip()

  try:
    parsed = yaml.safe_load(frontmatter_str)
  except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML in frontmatter: {e}") from e

  if not isinstance(parsed, dict):
    raise ValueError("SKILL.md frontmatter must be a YAML mapping")

  return parsed, body, skill_md


def _load_skill_from_dir(skill_dir: Union[str, pathlib.Path]) -> models.Skill:
  """Load a complete skill from a directory.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    Skill object with all components loaded.

  Raises:
    FileNotFoundError: If the skill directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid or the skill name does not match
      the directory name.
  """
  skill_dir = pathlib.Path(skill_dir).resolve()

  parsed, body, skill_md = _parse_skill_md(skill_dir)

  # Use model_validate to handle aliases like allowed-tools
  frontmatter = models.Frontmatter.model_validate(parsed)

  # Validate that skill name matches the directory name
  if skill_dir.name != frontmatter.name:
    raise ValueError(
        f"Skill name '{frontmatter.name}' does not match directory"
        f" name '{skill_dir.name}'."
    )

  references = _load_dir(skill_dir / "references")
  assets = _load_dir(skill_dir / "assets")
  raw_scripts = _load_dir(skill_dir / "scripts")
  scripts = {
      name: models.Script(src=content) for name, content in raw_scripts.items()
  }

  resources = models.Resources(
      references=references,
      assets=assets,
      scripts=scripts,
  )

  return models.Skill(
      frontmatter=frontmatter,
      instructions=body,
      resources=resources,
  )


def _validate_skill_dir(
    skill_dir: Union[str, pathlib.Path],
) -> list[str]:
  """Validate a skill directory without fully loading it.

  Checks that the directory exists, contains a valid SKILL.md with correct
  frontmatter, and that the skill name matches the directory name.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    List of problem strings. Empty list means the skill is valid.
  """
  problems: list[str] = []
  skill_dir = pathlib.Path(skill_dir).resolve()

  if not skill_dir.exists():
    return [f"Directory '{skill_dir}' does not exist."]
  if not skill_dir.is_dir():
    return [f"'{skill_dir}' is not a directory."]

  skill_md = None
  for name in ("SKILL.md", "skill.md"):
    path = skill_dir / name
    if path.exists():
      skill_md = path
      break
  if skill_md is None:
    return [f"SKILL.md not found in '{skill_dir}'."]

  try:
    parsed, _, _ = _parse_skill_md(skill_dir)
  except (FileNotFoundError, ValueError) as e:
    return [str(e)]

  unknown = set(parsed.keys()) - _ALLOWED_FRONTMATTER_KEYS
  if unknown:
    problems.append(f"Unknown frontmatter fields: {sorted(unknown)}")

  try:
    frontmatter = models.Frontmatter.model_validate(parsed)
  except Exception as e:
    problems.append(f"Frontmatter validation error: {e}")
    return problems

  if skill_dir.name != frontmatter.name:
    problems.append(
        f"Skill name '{frontmatter.name}' does not match directory"
        f" name '{skill_dir.name}'."
    )

  return problems


def _read_skill_properties(
    skill_dir: Union[str, pathlib.Path],
) -> models.Frontmatter:
  """Read only the frontmatter properties from a skill directory.

  This is a lightweight alternative to ``load_skill_from_dir`` when you
  only need the skill metadata without loading instructions or resources.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    Frontmatter object with the skill's metadata.

  Raises:
    FileNotFoundError: If the directory or SKILL.md is not found.
    ValueError: If the frontmatter is invalid.
  """
  skill_dir = pathlib.Path(skill_dir).resolve()
  parsed, _, _ = _parse_skill_md(skill_dir)
  return models.Frontmatter.model_validate(parsed)
