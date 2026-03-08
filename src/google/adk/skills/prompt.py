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

"""Module for skill prompt generation."""

from __future__ import annotations

import html
from typing import Any
from typing import List
from typing import Union
import warnings

from . import models


def format_skills_as_xml(
    skills: List[Union[models.Frontmatter, models.Skill]],
) -> str:
  """Formats available skills into a standard XML string.

  Args:
    skills: A list of skill frontmatter or full skill objects.

  Returns:
      XML string with <available_skills> block containing each skill's
      name and description.
  """

  if not skills:
    return "<available_skills>\n</available_skills>"

  lines = ["<available_skills>"]

  for item in skills:
    lines.append("<skill>")
    lines.append("<name>")
    lines.append(html.escape(item.name))
    lines.append("</name>")
    lines.append("<description>")
    lines.append(html.escape(item.description))
    lines.append("</description>")
    lines.append("</skill>")

  lines.append("</available_skills>")

  return "\n".join(lines)


def __getattr__(name: str) -> Any:
  if name == "DEFAULT_SKILL_SYSTEM_INSTRUCTION":

    from ..tools import skill_toolset

    warnings.warn(
        (
            "Importing DEFAULT_SKILL_SYSTEM_INSTRUCTION from"
            " google.adk.skills.prompt is deprecated."
            " Please import it from google.adk.tools.skill_toolset instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return skill_toolset.DEFAULT_SKILL_SYSTEM_INSTRUCTION
  raise AttributeError(f"module {__name__} has no attribute {name}")
