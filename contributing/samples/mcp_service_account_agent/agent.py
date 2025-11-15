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


from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowClientCredentials
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

# TODO: Update this to the production MCP server url and scopes.
MCP_SERVER_URL = "https://test.sandbox.googleapis.com/mcp"
SCOPES = {"https://www.googleapis.com/auth/cloud-platform": ""}

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="enterprise_assistant",
    instruction="""
Help the user with the tools available to you.
    """,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPServerParams(
                url=MCP_SERVER_URL,
            ),
            auth_scheme=OAuth2(
                flows=OAuthFlows(
                    clientCredentials=OAuthFlowClientCredentials(
                        tokenUrl="https://oauth2.googleapis.com/token",
                        scopes=SCOPES,
                    )
                )
            ),
            # TODO: Update this to the user's service account credentials.
            auth_credential=AuthCredential(
                auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
                service_account=ServiceAccount(
                    service_account_credential=ServiceAccountCredential(
                        type_="service_account",
                        project_id="example",
                        private_key_id="123",
                        private_key="123",
                        client_email="test@example.iam.gserviceaccount.com",
                        client_id="123",
                        auth_uri="https://accounts.google.com/o/oauth2/auth",
                        token_uri="https://oauth2.googleapis.com/token",
                        auth_provider_x509_cert_url=(
                            "https://www.googleapis.com/oauth2/v1/certs"
                        ),
                        client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/example.iam.gserviceaccount.com",
                        universe_domain="googleapis.com",
                    ),
                    scopes=SCOPES.keys(),
                ),
            ),
        )
    ],
)
