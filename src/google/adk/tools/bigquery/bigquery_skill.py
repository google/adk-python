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

from ...skills import load_skill_from_dir
from ...skills import Skill

_SKILL_DIR = pathlib.Path(__file__).parent / "skills" / "bigquery-ai-ml"


def get_bigquery_skill() -> Skill:
  """Returns the pre-packaged BigQuery data analysis skill.

  This skill follows the agentskills.io specification and
  provides curated instructions for BigQuery data analysis.
  Use it with SkillToolset alongside BigQueryToolset:

    from google.adk.tools.bigquery import BigQueryToolset
    from google.adk.tools.bigquery.bigquery_skill import get_bigquery_skill
    from google.adk.tools.skill_toolset import SkillToolset

    bq_skill = get_bigquery_skill()
    toolset = SkillToolset(skills=[bq_skill])
    bigquery_toolset = BigQueryToolset(...)
    agent = LlmAgent(tools=[bigquery_toolset, toolset])
  """
  return load_skill_from_dir(_SKILL_DIR)
