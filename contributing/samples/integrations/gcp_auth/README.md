# GCP Auth Sample

## Overview

Demonstrates the use of Agent Identity auth manager with an agent that queries
Spotify and Google Maps using auth providers.

Use `adk web` to run API key and 2-legged oauth flows, while use the included
custom agent web client to run 3-legged oauth flows.

## Setup

### 1. Activate environment

```bash
cd adk-python
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install "google-adk[agent-identity]"
```

### 3. Authenticate your environment

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="YOUR_GOOGLE_CLOUD_PROJECT"
gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
```

### 4. Create auth providers

Refer to the [public documentation](https://cloud.google.com/iam/docs/manage-auth-providers) to create the following Agent Identity auth providers.

> **Note:**
> The identity running the agent (via Application Default Credentials) must have
> the necessary [permissions](https://docs.cloud.google.com/iam/docs/roles-permissions/iamconnectors#iamconnectors.user)
> to retrieve credentials from these connectors. Ensure your account has the
> necessary role to access these resources.

```bash
export GOOGLE_CLOUD_LOCATION="YOUR_GOOGLE_CLOUD_LOCATION"
export MAPS_API_AUTH_PROVIDER_ID="YOUR_MAPS_API_AUTH_PROVIDER_ID"
export SPOTIFY_2LO_AUTH_PROVIDER_ID="YOUR_SPOTIFY_2LO_AUTH_PROVIDER_ID"
export SPOTIFY_3LO_AUTH_PROVIDER_ID="YOUR_SPOTIFY_3LO_AUTH_PROVIDER_ID"

gcloud alpha agent-identity connectors create $MAPS_API_AUTH_PROVIDER_ID \
    --project=$GOOGLE_CLOUD_PROJECT \
    --location=$GOOGLE_CLOUD_LOCATION \
    --api-key=YOUR_API_KEY

gcloud alpha agent-identity connectors create $SPOTIFY_2LO_AUTH_PROVIDER_ID \
    --project=$GOOGLE_CLOUD_PROJECT \
    --location=$GOOGLE_CLOUD_LOCATION \
    --two-legged-oauth-client-id=OAUTH_CLIENT_ID \
    --two-legged-oauth-client-secret=OAUTH_CLIENT_SECRET \
    --two-legged-oauth-token-endpoint=OAUTH_TOKEN_ENDPOINT

gcloud alpha agent-identity connectors create $SPOTIFY_3LO_AUTH_PROVIDER_ID \
    --project=$GOOGLE_CLOUD_PROJECT \
    --location=$GOOGLE_CLOUD_LOCATION \
    --three-legged-oauth-client-id=OAUTH_CLIENT_ID \
    --three-legged-oauth-client-secret=OAUTH_CLIENT_SECRET \
    --three-legged-oauth-authorization-url=AUTHORIZATION_URL \
    --three-legged-oauth-token-url=TOKEN_URL \
    --allowed-scopes=ALLOWED_SCOPES
```

## Sample Inputs

- `What is the current weather in New York?`

  *Tests the API key auth provider using the Google Maps tool.*

- `Tell me about the song: Waving Flag`

  *Tests the 2-legged OAuth (2LO) auth provider using the Spotify search track
  tool.*

- `Get my private playlists`

  *Tests the 3-legged OAuth (3LO) auth provider using the custom web client and
  the Spotify get playlists tool.*

## How To

### 1. Register the GCP Auth Provider

Register the Agent Identity authentication provider with the credential manager
so it can resolve GCP auth provider connector schemes.

```python
CredentialManager.register_auth_provider(GcpAuthProvider())
```

### 2. Configure 2-Legged OAuth (2LO)

Define an `AuthConfig` utilizing `GcpAuthProviderScheme` pointing to the 2LO
connector resource name. Attach it to an `AuthenticatedFunctionTool`.

```python
spotify_auth_config_2lo = AuthConfig(
    auth_scheme=GcpAuthProviderScheme(name=SPOTIFY_2LO_AUTH_PROVIDER)
)
spotify_search_track_tool = AuthenticatedFunctionTool(
    func=spotify_search_track,
    auth_config=spotify_auth_config_2lo,
)
```

See https://docs.cloud.google.com/iam/docs/auth-with-2lo for more details.

### 3. Configure 3-Legged OAuth (3LO)

For interactive user authorization flows, configure `GcpAuthProviderScheme` with
required `scopes` and a `continue_uri` where the OAuth callback will redirect
upon completion.

```python
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
```

See https://docs.cloud.google.com/iam/docs/auth-with-3lo for more details.

### 4. Configure Auth for MCP Toolsets

When utilizing an `McpToolset`, supply the `auth_scheme` directly to enable
automatic authentication (such as API key injection) during MCP server
communication.

```python
maps_tools = McpToolset(
    connection_params=StreamableHTTPConnectionParams(url=MAPS_MCP_ENDPOINT),
    auth_scheme=GcpAuthProviderScheme(name=MAPS_API_AUTH_PROVIDER),
    errlog=None,  # Required for agent-freezing (pickling)
)
```

## Testing the Sample

### 1. Test API key and 2LO auth provider using ADK web client

```bash
adk web contributing/samples
```

- On the ADK web UI, select the agent named `gcp_auth` from the dropdown.
- Try the sample queries from the **Sample Inputs** section for API key
  (Google Maps) and 2LO (Spotify).

### 2. Test 3LO auth provider using custom web client

- Navigate to the client directory and install dependencies:

```bash
cd contributing/samples/integrations/gcp_auth/client
pip install -r requirements.txt
```

- Start the client application:

```bash
uvicorn main:app --port 8080 --reload
```

- Open `http://localhost:8080`. (**Note:** You must use `localhost` and not
  `127.0.0.1`, as the OAuth redirect URL specifically requires it.)
- In the sidebar, configure your GCP Project ID and Location, click "Load Remote
  Agents", choose an engine to query, and click "Save & Apply Settings".
- Try the 3LO sample query to fetch private playlists.
