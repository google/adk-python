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
import textwrap
import uuid

from google.adk.agents import llm_agent
from google.adk.auth import auth_credential
from google.adk.integrations.eventarc import AgentProvided
from google.adk.integrations.eventarc import CloudEventAttributesBinding
from google.adk.integrations.eventarc import EventarcCredentialsConfig
from google.adk.integrations.eventarc import EventarcToolConfig
from google.adk.integrations.eventarc import EventarcToolset
from google.adk.integrations.eventarc import OMIT
import google.auth
import pydantic

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your_gcp_project_id")
BUS_NAME = os.getenv("EVENTARC_BUS_NAME", "outreach-bus")

# Define the desired credential type.
# By default use Application Default Credentials (ADC) from the local
# environment, which can be set up by following
# https://cloud.google.com/docs/authentication/provide-credentials-adc.
CREDENTIALS_TYPE = None

# Define an appropriate application name
EVENTARC_DOMAIN_AGENT_NAME = "adk_sample_domain_eventarc_agent"


# Define Eventarc tool config.
tool_config = EventarcToolConfig(project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))

if CREDENTIALS_TYPE == auth_credential.AuthCredentialTypes.OAUTH2:
  credentials_config = EventarcCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
  )
elif CREDENTIALS_TYPE == auth_credential.AuthCredentialTypes.SERVICE_ACCOUNT:
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = EventarcCredentialsConfig(credentials=creds)
else:
  application_default_credentials, _ = google.auth.default()
  credentials_config = EventarcCredentialsConfig(
      credentials=application_default_credentials
  )

toolset = EventarcToolset(
    credentials_config=credentials_config, tool_config=tool_config
)

# ---------------------------------------------------------------------------
# Create Domain-Specific Publish Tools
# ---------------------------------------------------------------------------


class OutreachContext(pydantic.BaseModel):
  customer_id: str
  resolution_notes: str
  successful: bool


# Example A: Fully Statically Bound (Safest)
# The developer locks down all routing. The agent only provides the business data.
complete_outreach_static_tool = toolset.create_publish_tool(
    name="complete_outreach_static",
    description="Logs a completed outreach attempt (statically bound routing).",
    payload_schema=OutreachContext,
    bus=f"projects/{PROJECT_ID}/locations/us-central1/messageBuses/{BUS_NAME}",
    ce_attributes_binding=CloudEventAttributesBinding(
        type="vendor_outreach.completed",
        source="//my-agent/outreach",
    ),
)

# Example B: Agent-Provided Attributes (Dynamic)
# The developer forces the agent to decide the routing bus and the event subject.
complete_outreach_dynamic_tool = toolset.create_publish_tool(
    name="complete_outreach_dynamic",
    description="Logs a completed outreach attempt (dynamic routing).",
    payload_schema=OutreachContext,
    bus=AgentProvided(
        "The full regional bus name: e.g.,"
        f" 'projects/{PROJECT_ID}/locations/us-central1/messageBuses/{BUS_NAME}'"
    ),
    ce_attributes_binding=CloudEventAttributesBinding(
        type="vendor_outreach.completed",
        source="//my-agent/outreach",
        subject=AgentProvided("The unique Customer ID being reached out to."),
    ),
)


# Example C: Lambda Execution & Mixed Custom Attributes
# The developer uses Python callables to generate IDs dynamically at runtime.
def get_custom_trace_id(payload: OutreachContext) -> str:
  return f"trace-{payload.customer_id}-{uuid.uuid4().hex[:8]}"


complete_outreach_lambda_tool = toolset.create_publish_tool(
    name="complete_outreach_lambda",
    description=(
        "Logs a completed outreach attempt (with lambda executions and custom"
        " attributes)."
    ),
    payload_schema=OutreachContext,
    bus=f"projects/{PROJECT_ID}/locations/us-central1/messageBuses/{BUS_NAME}",
    ce_attributes_binding=CloudEventAttributesBinding(
        type="vendor_outreach.completed",
        source="//my-agent/outreach",
        id=get_custom_trace_id,
        custom_attributes={
            "environment": "production",
            "priority": AgentProvided(
                "The priority of the outreach: 'high' or 'low'"
            ),
        },
    ),
)


# Example D: Empty Payloads & Dynamic Defaults
# Emit a simple signal (no business payload). The agent optionally decides the priority.
def default_priority(_: None) -> str:
  return "low"


ping_system_tool = toolset.create_publish_tool(
    name="ping_system",
    description="Pings the system. No data required.",
    payload_schema=None,
    bus=f"projects/{PROJECT_ID}/locations/us-central1/messageBuses/{BUS_NAME}",
    ce_attributes_binding=CloudEventAttributesBinding(
        type="system.ping",
        source="//my-agent/ping",
        custom_attributes={
            "retry": AgentProvided(
                "Whether to retry on failure", default="false"
            ),
            "priority": AgentProvided(
                "The priority of the ping", default=default_priority
            ),
        },
    ),
)

root_agent = llm_agent.LlmAgent(
    name=EVENTARC_DOMAIN_AGENT_NAME,
    description=(
        "Agent configured with domain-specific Eventarc publishing tools."
    ),
    instruction=textwrap.dedent(
        """        You are an e-commerce outreach agent. You can publish specific business events.
        You have four tools showing different configurations:
        1. `complete_outreach_static`: Fully static routing. Just pass `event_data`.
        2. `complete_outreach_dynamic`: Provide the `bus`, `subject` and `event_data`.
        3. `complete_outreach_lambda`: Provide the `priority` and `event_data`.
        4. `ping_system`: No payload! Optionally provide `retry` and `priority`.

        When a user gives you an instruction, determine which tool to use and execute it.
    """
    ),
    tools=[
        complete_outreach_static_tool,
        complete_outreach_dynamic_tool,
        complete_outreach_lambda_tool,
        ping_system_tool,
    ],
)
