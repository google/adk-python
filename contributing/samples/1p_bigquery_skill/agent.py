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

"""Example agent using BigQuery toolset with a 1P skill for guided workflows.

This sample demonstrates the pattern of combining a raw toolset (BigQueryToolset)
with a curated skill (SkillToolset) to provide both tools and guided workflows
to the agent.

Setup:
  1. Install ADK with BigQuery extras: pip install google-adk[bigquery]
  2. Configure OAuth credentials (see README.md)
  3. Run: adk web contributing/samples
"""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.integration.bigquery import BigQueryToolset
from google.adk.integration.bigquery import get_bigquery_skill
from google.adk.integration.bigquery.bigquery_credentials import BigQueryCredentialsConfig
from google.adk.tools.skill_toolset import SkillToolset

# Configure BigQuery credentials.
# Replace with your OAuth client credentials.
credentials_config = BigQueryCredentialsConfig(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
)

# BigQueryToolset provides the raw tools: execute_sql, list_dataset_ids, etc.
bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config,
)

# SkillToolset provides guided workflows via the 1P BigQuery skill.
# The agent can discover and load the skill's instructions and references.
bq_skill_toolset = SkillToolset(
    skills=[get_bigquery_skill()],
)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="bigquery_skill_agent",
    instruction=(
        "You are a data analyst. Use your tools and skills to help"
        " users explore and analyze BigQuery data. When starting a new"
        " analysis task, use `list_skills` to discover available skills"
        " and `load_skill` to get step-by-step guidance."
    ),
    tools=[bigquery_toolset, bq_skill_toolset],
)
