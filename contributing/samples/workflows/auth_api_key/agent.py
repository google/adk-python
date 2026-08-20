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

"""Auth API Key sample: FunctionNode with API key authentication.

Demonstrates how to use `auth_config` on a FunctionNode to pause
the workflow and request user credentials before running the node.

Flow:
  1. User sends any message to start the workflow.
  2. The `fetch_weather` node pauses and requests an API key.
  3. The user provides the API key through the auth UI.
  4. The node runs with the credential available in session state.
  5. The `summarize` node displays the result.
"""

from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from google.adk import Event
from google.adk import Workflow
from google.adk.agents.context import Context
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.workflow import node

# --- Auth configuration ---
# Uses API key auth: the simplest credential type.
# The user will be prompted to provide an API key via the auth UI.
auth_config = AuthConfig(
    auth_scheme=APIKey(**{'in': APIKeyIn.header, 'name': 'X-Api-Key'}),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key='placeholder',
    ),
    credential_key='weather_api_key',
)


@node(auth_config=auth_config, rerun_on_resume=True)
def fetch_weather(ctx: Context):
  """Fetches weather data using the authenticated API key."""
  # After auth completes, the credential is available via ctx.
  cred = ctx.get_auth_response(auth_config)
  api_key = cred.api_key if cred else 'unknown'

  # In a real agent, you would use the api_key to call an external API.
  # For this sample, we just echo it back (masked).
  masked = api_key[:4] + '****' if len(api_key) > 4 else '****'
  return {
      'city': 'San Francisco',
      'temperature': '18C',
      'condition': 'Sunny',
      'api_key_used': masked,
  }


def summarize(node_input: dict):
  """Displays the weather result."""
  yield Event(
      message=(
          f"Weather for {node_input['city']}:"
          f" {node_input['temperature']}, {node_input['condition']}."
          f" (Authenticated with key: {node_input['api_key_used']})"
      )
  )


root_agent = Workflow(
    name='auth_api_key',
    edges=[('START', fetch_weather, summarize)],
)
