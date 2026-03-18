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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.spanner.admin_toolset import SpannerAdminToolset
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig
import google.auth

# Define an appropriate credential type
# Set to None to use the application default credentials (ADC) for a quick
# development.
CREDENTIALS_TYPE = None

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initialize the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = SpannerCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
      scopes=[
          "https://www.googleapis.com/auth/spanner.admin",
          "https://www.googleapis.com/auth/spanner.data",
      ],
  )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = SpannerCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = SpannerCredentialsConfig(
      credentials=application_default_credentials
  )

spanner_admin_toolset = SpannerAdminToolset(
    credentials_config=credentials_config,
)

# The variable name `root_agent` determines what your root agent is for the
# debug CLI
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="spanner_admin_agent",
    description=(
        "Agent to perform Spanner admin tasks and answer questions about"
        " Spanner databases."
    ),
    instruction="""\
        You are a Spanner admin agent with access to several Spanner admin tools.
        Make use of those tools to answer user's questions and perform admin
        tasks like listing instances or databases.
    """,
    tools=[
        # Use tools from Spanner admin toolset.
        spanner_admin_toolset,
    ],
)
