# GCP Auth Sample

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

### 5. Test API key and 2LO auth provider using ADK web client

```bash
adk web contributing/samples
```

- On the ADK web UI, select the agent named `gcp_auth` from the dropdown.
- Sample queries to try:
  - API key (Google Maps tool): "What is the current weather in New York?"
  - 2LO key (Spotify tool): "Tell me about the song: Waving Flag"

### 6. Test 3LO auth provider using custom web client

> **Note:** If the agent backend is running on a different port or host other
> than `localhost:8000`, please set the `AGENT_BACKEND_URL` environment variable
> before starting the client (e.g.,
> `export AGENT_BACKEND_URL="http://localhost:9000"`).

- In a separate shell, activate environment

```bash
cd adk-python
python3 -m venv .venv
source .venv/bin/activate
```

- Navigate to the client directory and install dependencies

```bash
cd contributing/samples/gcp_auth/client
pip install -r requirements.txt
```

- Start the client application

```bash
uvicorn main:app --port 8080 --reload
```

- Open `http://localhost:8080`. (**Note:** You must use `localhost` and not `127.0.0.1`, as the OAuth redirect URL specifically requires it.)
- On the login screen, enter an arbitrary user ID (e.g. test_user123).
- Sample queries to try:
  - 3LO key (Spotify tool): "What are my private Spotify playlists?"
