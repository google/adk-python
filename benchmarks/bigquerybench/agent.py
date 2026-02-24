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
datasets.  Override write_mode via BQ_EVAL_WRITE_MODE env var when
evaluating AI/ML tools (forecast, detect_anomalies, etc.).
"""

import os

from google.adk.agents.llm_agent import LlmAgent
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

application_default_credentials, _ = google.auth.default()
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

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="bigquerybench_agent",
    description=(
        "Agent for BigQuery data exploration, SQL execution, and"
        " AI/ML operations against public datasets."
    ),
    instruction="""\
You are a data analyst with access to BigQuery tools.

Workflow:
1. Always explore the schema first: use list_dataset_ids,
   list_table_ids, and get_table_info to understand the data
   before writing any SQL.
2. Use execute_sql to run queries. Prefer explicit column names
   over SELECT *.
3. For forecasting, anomaly detection, or contribution analysis,
   use the dedicated tools (forecast, detect_anomalies,
   analyze_contribution) instead of raw SQL.
4. Present results clearly with column headers and values.

All public datasets are in project "bigquery-public-data".
""",
    tools=[bigquery_toolset],
)
