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

"""BigQueryBench evaluation agent.

Uses BigQueryToolset with read-only defaults against BigQuery public
datasets, and SkillToolset for skill-based workflows.  Override
write_mode via BQ_EVAL_WRITE_MODE env var when evaluating AI/ML
tools (forecast, detect_anomalies, etc.).
"""

from functools import cached_property
import logging
import os
import pathlib

from google.adk.agents.llm_agent import LlmAgent
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
from google.adk.models.google_llm import Gemini
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# ── BigQuery toolset (optional — requires ADC) ───────────────────
bigquery_toolset = None
try:
  from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsConfig
  from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset
  from google.adk.tools.bigquery.config import BigQueryToolConfig
  from google.adk.tools.bigquery.config import WriteMode
  import google.auth

  _WRITE_MODE_MAP = {
      "blocked": WriteMode.BLOCKED,
      "protected": WriteMode.PROTECTED,
      "allowed": WriteMode.ALLOWED,
  }

  _write_mode_str = os.environ.get("BQ_EVAL_WRITE_MODE", "blocked").lower()
  _write_mode = _WRITE_MODE_MAP.get(_write_mode_str, WriteMode.BLOCKED)

  application_default_credentials, project = google.auth.default()
  if not project and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    raise EnvironmentError("No GCP project found. Set GOOGLE_CLOUD_PROJECT.")
  credentials_config = BigQueryCredentialsConfig(
      credentials=application_default_credentials,
  )

  tool_config = BigQueryToolConfig(
      write_mode=_write_mode,
      max_query_result_rows=50,
  )

  bigquery_toolset = BigQueryToolset(
      credentials_config=credentials_config,
      bigquery_tool_config=tool_config,
  )
except Exception as e:
  logger.warning(
      "BigQuery toolset unavailable (%s). "
      "Skill-only evaluation will still work.",
      e,
  )

# ── Skill toolset ──────────────────────────────────────────────────
_SKILLS_DIR = pathlib.Path(__file__).parent / "skills"

_SKILL_NAMES = [
    "bq-sql-analyst",
]

_skills = [load_skill_from_dir(_SKILLS_DIR / name) for name in _SKILL_NAMES]

skill_toolset = SkillToolset(
    skills=_skills,
    code_executor=UnsafeLocalCodeExecutor(),
)


# ── Model (Vertex AI + API key) ──────────────────────────────────
class _VertexGemini(Gemini):
  """Gemini subclass that uses vertexai=True with an API key."""

  @cached_property
  def api_client(self):
    from google.genai import Client

    return Client(
        vertexai=True,
        api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
        http_options=genai_types.HttpOptions(
            headers=self._tracking_headers(),
            retry_options=self.retry_options,
            base_url=self.base_url,
        ),
    )


_SKILL_INSTRUCTION = """\
You are a data analyst with access to skills.

Workflow for skill-based tasks:
1. Use list_skills to discover available skills.
2. Use load_skill to read the skill's instructions.
3. Use load_skill_resource to examine references, sample data,
   or templates from the skill.
4. Follow the skill's instructions — this may involve running
   the skill's scripts via run_skill_script.
5. Present results clearly.

IMPORTANT: Only use the tools available to you (list_skills,
load_skill, load_skill_resource, run_skill_script). Do NOT
attempt to call tools that are not listed.
"""

_BQ_INSTRUCTION = """\
You are a data analyst with access to BigQuery tools and skills.

Workflow for direct BigQuery queries:
1. Always explore the schema first: use list_dataset_ids,
   list_table_ids, and get_table_info to understand the data
   before writing any SQL.
2. Use execute_sql to run queries. Prefer explicit column names
   over SELECT *.
3. For forecasting, anomaly detection, or contribution analysis,
   use the dedicated tools (forecast, detect_anomalies,
   analyze_contribution) instead of raw SQL.
4. Present results clearly with column headers and values.

Workflow for skill-based tasks:
1. Use list_skills to discover available skills.
2. Use load_skill to read the skill's instructions.
3. Use load_skill_resource to examine references, sample data,
   or templates from the skill.
4. Follow the skill's instructions — this may involve calling
   BigQuery tools (get_table_info, execute_sql) or running
   the skill's scripts via run_skill_script.
5. Present results clearly.

All public datasets are in project "bigquery-public-data".
"""

_INSTRUCTION = _BQ_INSTRUCTION if bigquery_toolset else _SKILL_INSTRUCTION

_api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
_model = (
    _VertexGemini(
        model="gemini-3-flash-preview",
        retry_options=genai_types.HttpRetryOptions(
            initialDelay=2,
            expBase=2,
            attempts=5,
        ),
    )
    if _api_key
    else "gemini-3-flash-preview"
)

root_agent = LlmAgent(
    model=_model,
    name="bigquerybench_agent",
    description=(
        "Agent for BigQuery data exploration, SQL execution, and"
        " AI/ML operations against public datasets.  Also supports"
        " skill-based workflows via SkillToolset."
    ),
    instruction=_INSTRUCTION,
    tools=[t for t in [bigquery_toolset, skill_toolset] if t],
)
