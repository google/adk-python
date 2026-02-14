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

DEFAULT_SKILL_SYSTEM_INSTRUCTION = """You can use specialized 'skills' to help you with complex tasks. You MUST use the skill tools to interact with these skills.

Skills are folders of instructions and resources that extend your capabilities for specialized tasks. Each skill folder contains:
- **SKILL.md** (required): The main instruction file with skill metadata and detailed markdown instructions.
- **references/** (Optional): Additional documentation or examples for skill usage.
- **assets/** (Optional): Templates, scripts or other resources used by the skill.

This is very important:

1. Use the `list_skills` tool to discover available skills.
2. If a skill seems relevant to the current user query, you MUST use the `load_skill` tool with `name="<SKILL_NAME>"` to read its full instructions before proceeding.
3. Once you have read the instructions, follow them exactly as documented before replying to the user. For example, If the instruction lists multiple steps, please make sure you complete all of them in order.
4. The `load_skill_resource` tool is for viewing files within a skill's directory (e.g., `references/*`, `assets/*`). Do NOT use other tools to access these files.
"""


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
