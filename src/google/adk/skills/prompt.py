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
from typing import List
from typing import Union

from . import models


def format_skills_as_xml(
    skills: List[Union[models.Frontmatter, models.Skill]],
) -> str:
  """Formats available skills into a standard XML string.

  Args:
    skills: A list of skill frontmatter or full skill objects.

  Returns:
      XML string with <available_skills> block containing each skill's
      name, description, and optionally source location.
  """

  if not skills:
    return "<available_skills>\n</available_skills>"

  lines = ["<available_skills>"]

  for item in skills:
    source_path = None
    if isinstance(item, models.Skill):
      source_path = item.source_path

    lines.append("<skill>")
    lines.append("<name>")
    lines.append(html.escape(item.name))
    lines.append("</name>")
    lines.append("<description>")
    lines.append(html.escape(item.description))
    lines.append("</description>")
    if source_path is not None:
      lines.append("<location>")
      lines.append(html.escape(source_path))
      lines.append("</location>")
    lines.append("</skill>")

  lines.append("</available_skills>")

  return "\n".join(lines)
