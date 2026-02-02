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

"""Agent that uses MCP toolset requiring OAuth authentication.

This agent demonstrates the toolset authentication feature where OAuth
credentials are required for both tool listing and tool calling.
"""

from __future__ import annotations

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.agents import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

# OAuth2 auth scheme with authorization code flow
# This specifies the OAuth metadata needed for the full OAuth flow
auth_scheme = OAuth2(
    flows=OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl='https://example.com/oauth/authorize',
            tokenUrl='https://example.com/oauth/token',
            scopes={'read': 'Read access', 'write': 'Write access'},
        )
    )
)

# OAuth credential with client credentials (used for token exchange)
# In a real scenario, this would be used to obtain the access token
auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id='test_client_id',
        client_secret='test_client_secret',
    ),
)

# Create the MCP toolset with OAuth authentication
mcp_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url='http://localhost:3001/mcp',
    ),
    auth_scheme=auth_scheme,
    auth_credential=auth_credential,
)

# Define the agent that uses the OAuth-protected MCP toolset
root_agent = LlmAgent(
    model='gemini-2.0-flash',
    name='oauth_mcp_agent',
    instruction="""You are a helpful assistant that can access user information.

You have access to tools that require authentication:
- get_user_profile: Get profile information for a specific user
- list_users: List all available users

When the user asks about users, use these tools to help them.""",
    tools=[mcp_toolset],
)
