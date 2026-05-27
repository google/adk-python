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

"""OAuth Authentication sample: FunctionNode with GitHub OAuth2 token request.

Demonstrates how to use `auth_config` with GitHub OAuth2 on a FunctionNode to pause
the workflow, request an OAuth token from the user, and use it to list the user's
GitHub repositories.

Flow:
  1. User sends any message to start the workflow.
  2. The `list_github_repos` node pauses and requests GitHub OAuth credentials.
  3. The user provides the credentials (after logging in to GitHub).
  4. The node runs, calls the GitHub API to list repos, and returns the list.
  5. The `display_result` node displays the repository names.

Sample queries:
  - "start"
  - "list my repos"
"""

import os

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk import Event
from google.adk import Workflow
from google.adk.agents.context import Context
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.workflow import node
import requests

# --- Auth configuration ---
# Uses GitHub OAuth2 authorization code flow.
# To use this sample, you need to register an OAuth application on GitHub
# and get a Client ID and Client Secret.
# Set the GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET environment variables.
auth_config = AuthConfig(
    auth_scheme=OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://github.com/login/oauth/authorize",
                tokenUrl="https://github.com/login/oauth/access_token",
                scopes={
                    "user": "Read user profile",
                    "repo": "Access public repositories",
                },
            )
        )
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id=os.environ.get(
                "GITHUB_CLIENT_ID", "YOUR_GITHUB_CLIENT_ID"
            ),
            client_secret=os.environ.get(
                "GITHUB_CLIENT_SECRET", "YOUR_GITHUB_CLIENT_SECRET"
            ),
        ),
    ),
    credential_key="github_oauth_token",
)


@node(auth_config=auth_config, rerun_on_resume=True)
def list_github_repos(ctx: Context):
  """Fetches GitHub repositories for the authenticated user."""
  # After auth completes, the credential is available via ctx.
  cred = ctx.get_auth_response(auth_config)

  access_token = cred.oauth2.access_token if cred and cred.oauth2 else None

  if not access_token:
    return {"status": "Error", "message": "No access token found"}

  # GitHub API requires a User-Agent header
  headers = {
      "Authorization": f"Bearer {access_token}",
      "User-Agent": "ADK-Sample-Agent",
      "Accept": "application/json",
  }

  try:
    response = requests.get(
        "https://api.github.com/user/repos", headers=headers
    )
    response.raise_for_status()
    repos_data = response.json()
    # Extract repo names
    repo_names = [repo["name"] for repo in repos_data]
    return {
        "status": "Success",
        "repos": repo_names,
    }
  except Exception as e:
    return {
        "status": "Error",
        "message": f"Failed to fetch repos: {e}",
    }


def display_result(node_input: dict):
  """Displays the result of accessing the resource."""
  if node_input["status"] == "Success":
    repos_str = ", ".join(node_input["repos"])
    yield Event(message=f"Successfully fetched repositories: {repos_str}")
  else:
    yield Event(
        message=(
            "Failed to fetch repositories. Error:"
            f" {node_input.get('message', 'Unknown error')}"
        )
    )


root_agent = Workflow(
    name="auth_oauth",
    edges=[("START", list_github_repos, display_result)],
)
