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
from google.adk.integrations.gcs import GCSToolset
from google.adk.integrations.gcs.gcs_credentials import GCSCredentialsConfig
from google.adk.integrations.gcs.settings import Capabilities
from google.adk.integrations.gcs.settings import GCSToolSettings
import google.auth

# Define an appropriate credential type.
# Set to None to use Application Default Credentials (ADC).
# This is the recommended way to use your `gcloud` credentials locally:
# Run `gcloud auth application-default login` in your terminal first.
CREDENTIALS_TYPE = None

# Define GCS tool config (default is READ_ONLY; add Capabilities.READ_WRITE for modification access)
tool_settings = GCSToolSettings(capabilities=[Capabilities.READ_WRITE])

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initialize the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = GCSCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
      scopes=[
          "https://www.googleapis.com/auth/cloud-platform",
          "https://www.googleapis.com/auth/devstorage.full_control",
      ],
  )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = GCSCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = GCSCredentialsConfig(
      credentials=application_default_credentials
  )

gcs_toolset = GCSToolset(
    credentials_config=credentials_config, gcs_tool_settings=tool_settings
)

# The variable name `root_agent` determines what your root agent is for the
# debug CLI
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="gcs_agent",
    description=(
        "Agent to answer questions about Google Cloud Storage (GCS) buckets"
        " and objects."
    ),
    instruction="""\
        You are a storage agent with access to several GCS tools.
        Make use of those tools to answer the user's questions about buckets and objects.
    """,
    tools=[
        gcs_toolset,
    ],
)
