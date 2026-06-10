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

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
from google.adk.integrations.agent_identity import GcpAuthProvider
from google.adk.integrations.agent_identity import GcpAuthProviderScheme
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import httpx

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
MAPS_API_AUTH_PROVIDER_ID = os.environ.get("MAPS_API_AUTH_PROVIDER_ID")
SPOTIFY_2LO_AUTH_PROVIDER_ID = os.environ.get("SPOTIFY_2LO_AUTH_PROVIDER_ID")
SPOTIFY_3LO_AUTH_PROVIDER_ID = os.environ.get("SPOTIFY_3LO_AUTH_PROVIDER_ID")

MAPS_API_AUTH_PROVIDER = f"projects/{PROJECT_ID}/locations/{LOCATION}/connectors/{MAPS_API_AUTH_PROVIDER_ID}"
SPOTIFY_2LO_AUTH_PROVIDER = f"projects/{PROJECT_ID}/locations/{LOCATION}/connectors/{SPOTIFY_2LO_AUTH_PROVIDER_ID}"
SPOTIFY_3LO_AUTH_PROVIDER = f"projects/{PROJECT_ID}/locations/{LOCATION}/connectors/{SPOTIFY_3LO_AUTH_PROVIDER_ID}"

MAPS_MCP_ENDPOINT = "https://mapstools.googleapis.com/mcp"
CONTINUE_URI = "http://localhost:8080/commit"
MODEL = "gemini-2.5-flash"


async def spotify_search_track(
    credential: AuthCredential, query: str
) -> str | list:
  """Searches for a track on Spotify and returns its details."""
  headers = {}
  if http := credential.http:
    if http.scheme and http.credentials and (token := http.credentials.token):
      headers["Authorization"] = f"{http.scheme.title()} {token}"
    if http.additional_headers:
      headers.update(http.additional_headers)

  if not headers:
    return "Error: No authentication token available."

  async with httpx.AsyncClient() as client:
    response = await client.get(
        "https://api.spotify.com/v1/search",
        headers=headers,
        params={"q": query, "type": "track", "limit": 1},
    )

    if response.status_code != 200:
      return f"Error from Spotify API: {response.status_code} - {response.text}"

    data = response.json()
    items = data.get("tracks", {}).get("items", [])

    if not items:
      return f"No track found for query '{query}'."

    return items


async def spotify_get_playlists(credential: AuthCredential) -> str | list:
  """Fetches the current user's private playlists on Spotify."""
  headers = {}
  if http := credential.http:
    if http.scheme and http.credentials and (token := http.credentials.token):
      headers["Authorization"] = f"{http.scheme.title()} {token}"
    if http.additional_headers:
      headers.update(http.additional_headers)

  if not headers:
    return "Error: No authentication token available."

  async with httpx.AsyncClient() as client:
    response = await client.get(
        "https://api.spotify.com/v1/me/playlists",
        headers=headers,
        params={"limit": 10},
    )

    if response.status_code != 200:
      return f"Error from Spotify API: {response.status_code} - {response.text}"

    data = response.json()
    items = data.get("items", [])

    if not items:
      return "No playlists found for the current user."

    # Extract useful information
    return [
        {
            "name": item.get("name"),
            "public": item.get("public"),
            "total_tracks": item.get("tracks", {}).get("total"),
        }
        for item in items
        if item
    ]


spotify_auth_config_2lo = AuthConfig(
    auth_scheme=GcpAuthProviderScheme(name=SPOTIFY_2LO_AUTH_PROVIDER)
)
spotify_search_track_tool = AuthenticatedFunctionTool(
    func=spotify_search_track,
    auth_config=spotify_auth_config_2lo,
)

spotify_auth_config_3lo = AuthConfig(
    auth_scheme=GcpAuthProviderScheme(
        name=SPOTIFY_3LO_AUTH_PROVIDER,
        scopes=["playlist-read-private"],
        continue_uri=CONTINUE_URI,
    )
)
spotify_get_playlist_tool = AuthenticatedFunctionTool(
    func=spotify_get_playlists,
    auth_config=spotify_auth_config_3lo,
)

maps_tools = McpToolset(
    connection_params=StreamableHTTPConnectionParams(url=MAPS_MCP_ENDPOINT),
    auth_scheme=GcpAuthProviderScheme(name=MAPS_API_AUTH_PROVIDER),
    errlog=None,  # Required for agent freezing (pickling)
)

CredentialManager.register_auth_provider(GcpAuthProvider())

root_agent = Agent(
    name="gcp_auth_agent",
    model=MODEL,
    instruction=(
        "You are a Spotify and Google Maps assistant. Use your tools to "
        "search for track details, fetch the user's private playlists, "
        "and look up locations. Keep responses concise, friendly, and "
        "emoji-filled!"
    ),
    tools=[
        spotify_search_track_tool,
        spotify_get_playlist_tool,
        maps_tools,
    ],
)

app = App(
    name="gcp_auth",
    root_agent=root_agent,
)
