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

from __future__ import annotations

import asyncio
import os
from typing import Any

from google.adk.agents import Agent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools import load_artifacts
from google.adk.tools.data_agent.config import DataAgentToolConfig
from google.adk.tools.data_agent.credentials import DataAgentCredentialsConfig
from google.adk.tools.data_agent.data_agent_toolset import DataAgentToolset
from google.adk.tools.tool_context import ToolContext
import google.auth
import google.auth.transport.requests
from google.genai import types

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
  if not application_default_credentials.valid:
    application_default_credentials.refresh(
        google.auth.transport.requests.Request()
    )
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


# NOTE: The generate_chart tool requires 'altair' and 'vl-convert-python' to be
# installed in your environment. You can install them using:
# pip install altair vl-convert-python
async def generate_chart(
    chart_spec: dict[str, Any], tool_context: ToolContext
) -> dict[str, str]:
  """Generates a professional chart using Altair based on a Vega-Lite spec.

  Args:
    chart_spec: A dictionary defining a Vega-Lite chart.
    tool_context: The tool context.

  Returns:
    A dictionary containing the status of the chart generation ("success" or
    "error"), a detail message, and the filename if successful.
  """
  import altair as alt
  import vl_convert as vlc

  try:
    # Altair can take a Vega-Lite dict directly and render it.
    # We use vl-convert to transform the spec into a high-quality PNG.
    png_data = await asyncio.to_thread(vlc.vegalite_to_png, chart_spec, scale=2)

    # Save as artifact
    await tool_context.save_artifact(
        "chart.png",
        types.Part.from_bytes(data=png_data, mime_type="image/png"),
    )
    title = chart_spec.get("title", "Chart")
    return {
        "status": "success",
        "detail": (
            f"Professional chart '{title}' rendered using Altair/Vega-Lite."
        ),
        "filename": "chart.png",
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    return {"status": "error", "detail": f"Failed to render chart: {str(e)}"}


root_agent = Agent(
    name="data_agent",
    description=(
        "Agent to answer user questions using Data Agents and generate charts."
    ),
    instruction=(
        "## Persona\nYou are a helpful assistant that uses Data Agents"
        " to answer user questions about their data.\n\n## Tools\n- You can"
        " list available data agents using `list_accessible_data_agents`.\n-"
        " You can get information about a specific data agent using"
        " `get_data_agent_info`.\n- You can chat with a specific data"
        " agent using `ask_data_agent`.\n- `generate_chart` renders"
        " professional charts from a `chart_spec` (Vega-Lite JSON). Use this"
        " whenever you need to visualize data; do not show raw JSON to the"
        " user.\n- You can load artifacts using `load_artifacts`.\n"
    ),
    tools=[da_toolset, generate_chart, load_artifacts],
)
