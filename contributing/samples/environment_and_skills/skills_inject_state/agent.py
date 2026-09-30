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

"""Sample showing how a skill personalizes its instructions from session state.

A skill whose ``SKILL.md`` frontmatter sets ``metadata.adk_inject_state: true``
has its instructions rendered through ``inject_session_state`` at load time.
Any ``{placeholder}`` in the instructions is replaced with the matching value
from session state, so the same skill yields different instructions per session.

Run from the parent directory with ``adk web``.
"""

from __future__ import annotations

import pathlib

from google.adk.agents import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.tool_context import ToolContext


def remember_developer_profile(
    name: str,
    primary_language: str,
    experience_level: str,
    tool_context: ToolContext,
) -> dict:
  """Saves the developer's profile into session state for later personalization.

  Args:
    name: The developer's name.
    primary_language: The language they primarily work in, e.g. "Python".
    experience_level: Their experience level, e.g. "junior" or "senior".
  """
  tool_context.state["dev_name"] = name
  tool_context.state["dev_language"] = primary_language
  tool_context.state["dev_level"] = experience_level
  return {
      "status": "ok",
      "stored": {
          "dev_name": name,
          "dev_language": primary_language,
          "dev_level": experience_level,
      },
  }


# Load a directory-based skill. Its SKILL.md opts into state injection via
# `metadata.adk_inject_state: true`, so `{dev_name}`, `{dev_language}`, and
# `{dev_level}` in its instructions are substituted from session state when the
# skill is loaded.
code_review_skill = load_skill_from_dir(
    pathlib.Path(__file__).parent / "skills" / "code-review-skill"
)

my_skill_toolset = SkillToolset(skills=[code_review_skill])

root_agent = Agent(
    name="skills_inject_state_agent",
    description=(
        "An agent that personalizes a code-review skill using session state."
    ),
    instruction=(
        "You help developers review their code.\n"
        "- When a user introduces themselves, call"
        " `remember_developer_profile` to save who they are.\n"
        "- When a user asks for a code review, load the `code-review-skill`"
        " and follow its (personalized) instructions exactly."
    ),
    tools=[
        remember_developer_profile,
        my_skill_toolset,
    ],
)
