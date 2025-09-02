from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams

import os
from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set. Please create a .env file with your token.")

def get_github_tools() -> MCPToolset:
    """Initializes and returns the GitHub toolset."""
    github_auth_scheme, github_auth_credential = token_to_scheme_credential(
    token_type='apikey',
    location='header',
    name="GitHub-Token",
    credential_value=GITHUB_TOKEN
    )
    return MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="https://api.githubcopilot.com/mcp/",
            headers={
                "Authorization": "Bearer " + GITHUB_TOKEN,
            },
        ),
        auth_credential=github_auth_credential,
        auth_scheme=github_auth_scheme,
        # tool_filter=["get_me", "list_pull_requests", "request_copilot_review", "merge_pull_request"]
    )
