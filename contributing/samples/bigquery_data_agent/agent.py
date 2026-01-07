# Copyright 2025 Google LLC
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

import os

from google.adk.agents import Agent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsConfig
from google.adk.tools.bigquery.bigquery_data_agent_toolset import BigQueryDataAgentToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
import google.auth

# Define the desired credential type.
# By default use Application Default Credentials (ADC) from the local
# environment, which can be set up by following
# https://cloud.google.com/docs/authentication/provide-credentials-adc.
CREDENTIALS_TYPE = None

# Define an appropriate application name
BIGQUERY_AGENT_NAME = "adk_sample_bigquery_agent"


# Define BigQuery tool config with write mode set to allowed. Note that this is
# only to demonstrate the full capability of the BigQuery tools. In production
# you may want to change to BLOCKED (default write mode, effectively makes the
# tool read-only) or PROTECTED (only allows writes in the anonymous dataset of a
# BigQuery session) write mode.
tool_config = BigQueryToolConfig(
    write_mode=WriteMode.ALLOWED, application_name=BIGQUERY_AGENT_NAME
)

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initiaze the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = BigQueryCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
  )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = BigQueryCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = BigQueryCredentialsConfig(
      credentials=application_default_credentials
  )

bq_da_toolset = BigQueryDataAgentToolset(
    credentials_config=credentials_config,
    bigquery_tool_config=tool_config,
    tool_filter=[
        "list_accessible_data_agents",
        "get_data_agent_info",
        "ask_data_agent",
    ],
)

root_agent = Agent(
    name="bigquery_data_agent",
    model="gemini-2.0-flash",
    description="Agent to answer user questions using BigQuery Data Agents.",
    instruction=(
        "## Persona\nYou are a helpful assistant that uses BigQuery Data Agents"
        " to answer user questions about their data.\n\n## Tools\n- You can"
        " list available data agents using `list_accessible_data_agents`.\n-"
        " You can get information about a specific data agent using"
        " `get_data_agent_info`.\n- You can chat with a specific data"
        " agent using `ask_data_agent`.\n"
    ),
    tools=[bq_da_toolset],
)
