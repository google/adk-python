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

"""The skill scenario: an agent that loads a skill before it answers.

``build_skill_test_runner`` runs a model that calls ``load_skill`` (and the
resource / script tools) before answering.
"""

from __future__ import annotations

from typing import Literal
from typing import Sequence

from google.adk.agents.llm_agent import Agent
from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.models.base_llm import BaseLlm
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Resources
from google.adk.skills.models import Script
from google.adk.skills.models import Skill
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.tools.skill_toolset import SkillToolset
from google.genai.types import Part
from typing_extensions import override

from ....testing_utils import TestInMemoryRunner
from .conversation import AGENT_DESCRIPTION
from .conversation import AGENT_NAME
from .conversation import BASE_INSTRUCTION
from .conversation import FINAL_TEXT
from .conversation import FIRST_TURN_USAGE
from .conversation import SECOND_TURN_USAGE
from .conversation import Turn

# The type of skill being used in a test case.
SkillType = Literal["local", "registry", "nonexistent"]
SkillResourceType = Literal[
    "references", "assets", "scripts", "wrong_type", "wrong_name"
]

REGISTRY_SKILL_NAME = "registry-skill"
LOCAL_SKILL_NAME = "local-skill"
NONEXISTENT_SKILL_NAME = "nonexistent-skill"
SKILL_DESCRIPTION = "A sample skill."


def _make_skill(
    *,
    name: str = LOCAL_SKILL_NAME,
    source: str = "static",
    additional_tools: Sequence[str] | None = None,
) -> Skill:
  additional_tools = additional_tools or []

  skill = Skill(
      frontmatter=Frontmatter(
          name=name,
          description=SKILL_DESCRIPTION,
          metadata={"adk_additional_tools": additional_tools},
      ),
      instructions="skill instructions",
      resources=Resources(
          references={"ref1": "ref1_content"},
          assets={"deeply/hidden/asset1": "asset1_content"},
          scripts={
              "script1": Script(src="script1_content"),
              "ec_0.py": Script(src="print(':D')"),
              "ec_1.py": Script(src="foo = 1/0"),
              "ec_10.py": Script(src="import sys; sys.exit(10)"),
          },
      ),
  )
  if source == "registry":
    skill._uri = f"https://fake-registry.com/skill/{name}"
  else:
    skill._uri = f"file://{name}"
  return skill


class _FakeSkillRegistry(SkillRegistry):
  """Registry serving one in-memory skill, with no network of its own."""

  def __init__(self, skill: Skill) -> None:
    self._skill = skill

  @override
  async def get_skill(self, *, name: str) -> Skill:
    # A fresh copy per fetch: the toolset stamps `source` on what it gets back.
    if name == self._skill.frontmatter.name:
      return self._skill.model_copy(deep=True)
    else:
      raise KeyError(f"Skill {name} not found")

  @override
  async def search_skills(self, *, query: str) -> list[Frontmatter]:
    return []


_SKILL_CALL_PARTS: dict[SkillType, Part] = {
    "local": Part.from_function_call(
        name="load_skill", args={"skill_name": LOCAL_SKILL_NAME}
    ),
    "registry": Part.from_function_call(
        name="load_skill", args={"skill_name": REGISTRY_SKILL_NAME}
    ),
    "nonexistent": Part.from_function_call(
        name="load_skill", args={"skill_name": NONEXISTENT_SKILL_NAME}
    ),
}


def _load_resource(file_path: str) -> Part:
  return Part.from_function_call(
      name="load_skill_resource",
      args={"skill_name": REGISTRY_SKILL_NAME, "file_path": file_path},
  )


_SKILL_RESOURCE_PARTS: dict[SkillResourceType, Part] = {
    "references": _load_resource("references/ref1"),
    "assets": _load_resource("assets/deeply/hidden/asset1"),
    "scripts": _load_resource("scripts/script1"),
    "wrong_type": _load_resource("fake/file/not/existing"),
    "wrong_name": _load_resource("references/nope/never"),
}


def _run_script(exit_code: int) -> Part:
  return Part.from_function_call(
      name="run_skill_script",
      args={
          "skill_name": REGISTRY_SKILL_NAME,
          "file_path": f"scripts/ec_{exit_code}.py",
      },
  )


def skill_turns(
    skills: Sequence[SkillType],
    resources: Sequence[SkillResourceType] = (),
    scripts_return_exit_codes: Sequence[int] = (),
) -> tuple[Turn, ...]:
  """The canned conversation for the skill scenario.

  One ``load_skill`` call per skill the case loads, one
  ``load_skill_resource`` call per resource, then the answer: the skill
  scenario's counterpart to ``TOOL_CALLING_TURNS``, which every other
  scenario shares. Billed like that one, so what the skill cases record
  differs from the rest only in which tool the model calls.
  """
  return (
      *((_SKILL_CALL_PARTS[skill], FIRST_TURN_USAGE) for skill in skills),
      *(
          (_SKILL_RESOURCE_PARTS[resource], FIRST_TURN_USAGE)
          for resource in resources
      ),
      *(
          (_run_script(exit_code), FIRST_TURN_USAGE)
          for exit_code in scripts_return_exit_codes
      ),
      (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
  )


def build_skill_test_agent(model: BaseLlm) -> Agent:
  """Builds the agent whose model calls ``load_skill`` then answers."""
  registry = _FakeSkillRegistry(
      _make_skill(name=REGISTRY_SKILL_NAME, source="registry"),
  )
  toolset = SkillToolset(
      [_make_skill(additional_tools=["foo", "bar"])],
      registry=registry,
      code_executor=UnsafeLocalCodeExecutor(),
  )
  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      tools=[toolset],
  )


def build_skill_test_runner(model: BaseLlm) -> TestInMemoryRunner:
  """Builds a runner whose model calls ``load_skill`` then answers."""
  return TestInMemoryRunner(node=build_skill_test_agent(model))
