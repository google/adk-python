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
      if file_path.is_file():
        relative_path = file_path.relative_to(directory)
        files[str(relative_path)] = file_path.read_text(encoding="utf-8")
  return files


def load_skill_from_dir(skill_dir: Union[str, pathlib.Path]) -> models.Skill:
  """Load a complete skill from a directory.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    Skill object with all components loaded.

  Raises:
    FileNotFoundError: If the skill directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid.
  """
  skill_dir = pathlib.Path(skill_dir).resolve()

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

  # Frontmatter class handles required field validation
  frontmatter = models.Frontmatter(**parsed)

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
