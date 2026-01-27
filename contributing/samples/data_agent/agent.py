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

import os

from google.adk.agents import Agent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.data_agent.config import DataAgentToolConfig
from google.adk.tools.data_agent.credentials import DataAgentCredentialsConfig
from google.adk.tools.data_agent.data_agent_toolset import DataAgentToolset
import google.auth
import google.auth.transport.requests

# Define the desired credential type.
# By default use Application Default Credentials (ADC) from the local
# environment, which can be set up by following
# https://cloud.google.com/docs/authentication/provide-credentials-adc.
CREDENTIALS_TYPE = None

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initiaze the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = DataAgentCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
  )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file(
      "service_account_key.json",
      scopes=["https://www.googleapis.com/auth/cloud-platform"],
  )
  creds.refresh(google.auth.transport.requests.Request())
  credentials_config = DataAgentCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = DataAgentCredentialsConfig(
      credentials=application_default_credentials
  )

tool_config = DataAgentToolConfig(
    max_query_result_rows=100,
)
da_toolset = DataAgentToolset(
    credentials_config=credentials_config,
    data_agent_tool_config=tool_config,
    tool_filter=[
        "list_accessible_data_agents",
        "get_data_agent_info",
        "ask_data_agent",
    ],
)

root_agent = Agent(
    name="data_agent",
    model="gemini-2.0-flash",
    description="Agent to answer user questions using Data Agents.",
    instruction=(
        "## Persona\nYou are a helpful assistant that uses Data Agents"
        " to answer user questions about their data.\n\n## Tools\n- You can"
        " list available data agents using `list_accessible_data_agents`.\n-"
        " You can get information about a specific data agent using"
        " `get_data_agent_info`.\n- You can chat with a specific data"
        " agent using `ask_data_agent`.\n"
    ),
    tools=[da_toolset],
)
