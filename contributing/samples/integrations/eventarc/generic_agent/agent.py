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

from google.adk.agents import llm_agent
from google.adk.auth import auth_credential
from google.adk.integrations.eventarc import EventarcCredentialsConfig
from google.adk.integrations.eventarc import EventarcToolConfig
from google.adk.integrations.eventarc import EventarcToolset
import google.auth

# Define the desired credential type.
# By default use Application Default Credentials (ADC) from the local
# environment, which can be set up by following
# https://cloud.google.com/docs/authentication/provide-credentials-adc.
CREDENTIALS_TYPE = None

# Define an appropriate application name
EVENTARC_AGENT_NAME = "adk_sample_eventarc_agent"


# Define Eventarc tool config.
# You can optionally set the project_id here, or let the agent infer it from context/user input.
tool_config = EventarcToolConfig(project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))

if CREDENTIALS_TYPE == auth_credential.AuthCredentialTypes.OAUTH2:
  # Initialize the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = EventarcCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
  )
elif CREDENTIALS_TYPE == auth_credential.AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = EventarcCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = EventarcCredentialsConfig(
      credentials=application_default_credentials
  )

toolset = EventarcToolset(
    credentials_config=credentials_config, tool_config=tool_config
)

# The variable name `root_agent` determines what your root agent is for the
# debug CLI
root_agent = llm_agent.LlmAgent(
    name=EVENTARC_AGENT_NAME,
    description=(
        "Agent to publish structured CloudEvents to Google Cloud Eventarc."
    ),
    instruction=textwrap.dedent("""\
        You are a cloud engineer agent with access to Google Cloud Eventarc tools.
        You can publish CloudEvents structured messages to Eventarc message buses.
    """),
    tools=[toolset],
)
