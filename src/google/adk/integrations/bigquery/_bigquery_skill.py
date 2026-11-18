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

"""Pre-packaged BigQuery skill for use with SkillToolset."""

from __future__ import annotations

import pathlib
from typing import Literal

from ...skills import load_skill_from_dir
from ...skills import Skill

_SKILLS_BASE_DIR = pathlib.Path(__file__).parent / "skills"
_AVAILABLE_SKILLS: tuple[Literal["bigquery-ai-ml", "bigquery-graph"], ...] = (
    "bigquery-ai-ml",
    "bigquery-graph",
)


def get_bigquery_skill(
    skill_name: Literal["bigquery-ai-ml", "bigquery-graph"] = "bigquery-ai-ml",
) -> Skill:
  """Returns a pre-packaged BigQuery skill by name.

  These skills follow the agentskills.io specification and provide curated
  instructions for BigQuery AI/ML functions and Graph Query Language (GQL) or
  SQL/PGQ queries. Use them with SkillToolset alongside BigQueryToolset:

    from google.adk.integrations.bigquery import BigQueryToolset
    from google.adk.integrations.bigquery import get_bigquery_skill
    from google.adk.tools.skill_toolset import SkillToolset

    bq_skill = get_bigquery_skill("bigquery-ai-ml")
    graph_skill = get_bigquery_skill("bigquery-graph")
    toolset = SkillToolset(skills=[bq_skill, graph_skill])
    bigquery_toolset = BigQueryToolset(...)
    agent = LlmAgent(tools=[bigquery_toolset, toolset])

  Args:
    skill_name: Name of the built-in BigQuery skill to load. Supported values
      are ``"bigquery-ai-ml"`` (default) and ``"bigquery-graph"``.

  Returns:
    The loaded BigQuery ``Skill`` instance.

  Raises:
    ValueError: If ``skill_name`` is not a recognized built-in BigQuery skill.
  """
  if skill_name not in _AVAILABLE_SKILLS:
    available = ", ".join(repr(s) for s in _AVAILABLE_SKILLS)
    raise ValueError(
        f"Unknown BigQuery skill {skill_name!r}. Available skills: {available}."
    )
  return load_skill_from_dir(_SKILLS_BASE_DIR / skill_name)
