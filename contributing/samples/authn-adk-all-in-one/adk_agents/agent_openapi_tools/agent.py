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

import logging
import os

from google.adk.tools.openapi_tool.auth.auth_helpers import openid_url_to_scheme_credential
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

credential_dict = {
    "client_id": os.environ.get("OAUTH_CLIENT_ID"),
    "client_secret": os.environ.get("OAUTH_CLIENT_SECRET"),
}
auth_scheme, auth_credential = openid_url_to_scheme_credential(
    openid_url="http://localhost:5000/.well-known/openid-configuration",
    credential_dict=credential_dict,
    scopes=[],
)


# Open API spec
file_path = "./agent_openapi_tools/openapi.yaml"
file_content = None

try:
  with open(file_path, "r") as file:
    file_content = file.read()
except FileNotFoundError:
  # so that the execution does not continue when the file is not found.
  raise FileNotFoundError(f"Error: The API Spec '{file_path}' was not found.")


# Example with a JSON string
openapi_spec_yaml = file_content  # Your OpenAPI YAML string
openapi_toolset = OpenAPIToolset(
    spec_str=openapi_spec_yaml,
    spec_str_type="yaml",
    auth_scheme=auth_scheme,
    auth_credential=auth_credential,
)

from google.adk.agents import LlmAgent

root_agent = LlmAgent(
    name="hotel_agent",
    instruction=(
        "Help user find and book hotels, fetch their bookings using the tools"
        " provided."
    ),
    description="Hotel Booking Agent",
    model=os.environ.get("GOOGLE_MODEL"),
    tools=[openapi_toolset],  # Pass the toolset
    # ... other agent config ...
)
