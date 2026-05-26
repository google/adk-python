# ADK Workflow Auth Config Sample

## Overview

This sample demonstrates how to use `auth_config` on a `FunctionNode` to require user authentication before the node runs.

When a node has `auth_config`, the workflow automatically:

1. Pauses the node and emits an `adk_request_credential` FunctionCall event
1. The invocation ends — the node is marked as waiting
1. The client sends a new request with the credential as a FunctionResponse
1. The workflow stores the credential in session state and re-runs the node

The **ADK web UI** (`adk web`) handles step 3 automatically — it recognizes auth
requests and presents an auth dialog. If you use a custom client, you need to
handle the `adk_request_credential` FunctionCall and respond with the credential
yourself.

This sample uses **API key** authentication (the simplest credential type).

## No External Setup Required

This sample uses a mock weather lookup. No external API key or server is needed. When the auth UI prompts for a key, you can enter any value (e.g., `my-test-key-123`).

## Sample Inputs

Send any message (e.g., `go`) to start the workflow.

## Graph

```mermaid
graph TD
    START --> fetch_weather[fetch_weather <br/>pauses for auth on first run]
    fetch_weather --> summarize
```

## How To

1. Define an `AuthConfig` with the auth scheme and credential type:

   ```python
   from google.adk.auth.auth_tool import AuthConfig
   from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes

   auth_config = AuthConfig(
       auth_scheme=APIKey(**{'in': APIKeyIn.header, 'name': 'X-Api-Key'}),
       raw_auth_credential=AuthCredential(
           auth_type=AuthCredentialTypes.API_KEY,
           api_key='placeholder',
       ),
       credential_key='weather_api_key',
   )
   ```

1. Use the `@node` decorator with `auth_config` and `rerun_on_resume=True`:

   ```python
   @node(auth_config=auth_config, rerun_on_resume=True)
   def fetch_weather(ctx: Context):
       ...
   ```

1. Inside the function, retrieve the credential from `ctx`:

   ```python
   def fetch_weather(ctx: Context):
       cred = ctx.get_auth_response(auth_config)
       api_key = cred.api_key
       # Use api_key to call your API...
   ```

## OAuth2

The same `auth_config` pattern works with OAuth2 and OpenID Connect. The key
differences:

- **Auth scheme**: Use `OAuth2` (from `fastapi.openapi.models`) instead of
  `APIKey`. Configure the authorization and token URLs in the OAuth2 flows.
- **Raw credential**: Set `auth_type=AuthCredentialTypes.OAUTH2` and provide
  `client_id`, `client_secret`, and `redirect_uri` in the `oauth2` field.
- **Web UI flow**: The ADK web UI recognizes OAuth2 auth requests and opens
  an authorization popup automatically. The user authenticates with the
  provider, and the UI sends the full `AuthConfig` response back. No special
  handling is needed in the node.
- **Token exchange**: The framework automatically exchanges the authorization
  code for an access token via `AuthHandler.exchange_auth_token()`.

```python
from fastapi.openapi.models import OAuth2, OAuthFlowAuthorizationCode, OAuthFlows

auth_config = AuthConfig(
    auth_scheme=OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl='https://provider.com/authorize',
                tokenUrl='https://provider.com/token',
                scopes={'read': 'Read access'},
            )
        )
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id='YOUR_CLIENT_ID',
            client_secret='YOUR_CLIENT_SECRET',
            redirect_uri='http://localhost:8000/callback',
        ),
    ),
    credential_key='my_oauth_credential',
)
```
