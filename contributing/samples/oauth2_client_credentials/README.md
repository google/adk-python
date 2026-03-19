# OAuth2 Client Credentials Weather Agent

This sample demonstrates OAuth2 client credentials flow with ADK's `AuthenticatedFunctionTool` using a practical weather assistant agent.

## Overview

The OAuth2 client credentials grant type is used for server-to-server authentication where no user interaction is required. This demo shows:

- How to configure OAuth2 client credentials in ADK
- Using `AuthenticatedFunctionTool` for automatic token management
- Transparent authentication in a practical weather assistant
- Testing the OAuth2 client credentials implementation

## Architecture

```
[WeatherAssistant] -> [AuthenticatedFunctionTool] -> [OAuth2CredentialExchanger] -> [OAuth2 Server] -> [Weather API]
```

1. **WeatherAssistant** calls weather tool when user asks for weather data
2. **AuthenticatedFunctionTool** automatically handles OAuth2 flow
3. **OAuth2CredentialExchanger** exchanges client credentials for access token
4. **Authenticated requests** are made to weather API

## Files

### `agent.py` - WeatherAssistant Agent

Weather assistant agent that demonstrates OAuth2 client credentials flow transparently:

- **OAuth2 Configuration**: Client credentials setup with token URL and scopes
- **Weather Tool**: Single `get_weather_data` tool for fetching weather information
- **Agent Definition**: ADK LLM agent focused on providing weather information

**Key Features:**
- Automatic token exchange using client ID and secret
- Bearer token authentication
- Transparent OAuth2 handling (invisible to the model)
- Practical use case demonstrating machine-to-machine authentication

### `main.py` - CLI Interface

Command-line interface for running the WeatherAssistant agent:

```bash
# Ask for weather
python contributing/samples/oauth2_client_credentials/main.py "What's the weather in Tokyo?"
```

**Requirements:**
- LLM API key (Google AI or Vertex AI)
- OAuth2 test server running

### `oauth2_test_server.py` - Local OAuth2 Server

Mock OAuth2 server for testing the client credentials flow:

```bash
python contributing/samples/oauth2_client_credentials/oauth2_test_server.py
```

**Features:**
- OIDC discovery endpoint (`/.well-known/openid_configuration`)
- Client credentials token exchange (`/token`)
- Protected weather API (`/api/weather`)
- Supports both `authorization_code` and `client_credentials` grant types
- Test credentials: `client_id="test_client"`, `client_secret="test_secret"`

**Endpoints:**
- `GET /.well-known/openid_configuration` - OIDC discovery
- `POST /token` - Token exchange
- `GET /api/weather` - Weather API (requires Bearer token)
- `GET /` - Server info

## Quick Start

1. **Start the OAuth2 server:**
   ```bash
   python contributing/samples/oauth2_client_credentials/oauth2_test_server.py &
   ```
2. Create a `.env` file in the project root with your API credentials:

```bash
# Choose Model Backend: 0 -> ML Dev, 1 -> Vertex
GOOGLE_GENAI_USE_VERTEXAI=1

# ML Dev backend config
GOOGLE_API_KEY=your_google_api_key_here

# Vertex backend config
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

3. **Run the agent:**
   ```bash
   # Ask for weather
   python contributing/samples/oauth2_client_credentials/main.py "What's the weather in Tokyo?"
   ```

3. **Interactive demo (use ADK commands):**
   ```bash
   # Interactive CLI
   adk run contributing/samples/oauth2_client_credentials

   # Interactive web UI
   adk web contributing/samples
   ```

## OAuth2 Configuration

The agent uses these OAuth2 settings (configured in `agent.py`):

```python
flows = OAuthFlows(
    clientCredentials=OAuthFlowClientCredentials(
        tokenUrl="http://localhost:8000/token",
        scopes={
            "read": "Read access to weather data",
            "write": "Write access for data updates",
            "admin": "Administrative access",
        },
    )
)

raw_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="test_client",
        client_secret="test_secret",
    ),
)
```

## Authentication Flow

1. **Weather Request**: User asks WeatherAssistant for weather information
2. **Tool Invocation**: Agent calls `get_weather_data` authenticated function tool
3. **Credential Loading**: CredentialManager loads OAuth2 configuration
4. **Token Exchange**: OAuth2CredentialExchanger uses client credentials to get access token
5. **Request Enhancement**: AuthenticatedFunctionTool adds `Authorization: Bearer <token>` header
6. **API Call**: Weather API accessed with valid token
7. **Response**: Weather data returned to user
