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

from . import models


def format_skills_as_xml(skills: List[models.Frontmatter]) -> str:
  """Formats available skills into a standard XML string.

  Args:
    skills: A list of skill frontmatter objects.

  Returns:
      XML string with <available_skills> block containing each skill's
      name and description.
  """

  if not skills:
    return "<available_skills>\n</available_skills>"

  lines = ["<available_skills>"]

  for skill in skills:
    lines.append("<skill>")
    lines.append("<name>")
    lines.append(html.escape(skill.name))
    lines.append("</name>")
    lines.append("<description>")
    lines.append(html.escape(skill.description))
    lines.append("</description>")
    lines.append("</skill>")

  lines.append("</available_skills>")

  return "\n".join(lines)
