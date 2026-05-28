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
"""
Weather API OAuth2 Test Server

A simple FastAPI server that implements OAuth2 flows for weather API testing:
- Client Credentials Flow
- Authorization Code Flow

Usage:
    python oauth2_test_server.py

Endpoints:
    GET  /auth                 - Authorization endpoint (auth code flow)
    POST /token                - Token endpoint (both flows)
    GET  /.well-known/openid_configuration - OpenID Connect discovery
    GET  /api/weather          - Weather API (requires Bearer token)
"""

import secrets
import time
from typing import Dict
from typing import Optional

from fastapi import FastAPI
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import status
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

app = FastAPI(title="Weather API OAuth2 Server", version="1.0.0")

# In-memory storage (for testing only)
clients = {
    "test_client": {
        "client_secret": "test_secret",
        "redirect_uris": [
            "http://localhost:8080/callback",
            "urn:ietf:wg:oauth:2.0:oob",
        ],
        "scopes": ["read", "write", "admin"],
    }
}

authorization_codes = {}  # code -> {client_id, redirect_uri, scope, expires_at}
access_tokens = {}  # token -> {client_id, scope, expires_at, token_type}


class TokenResponse(BaseModel):
  access_token: str
  token_type: str = "Bearer"
  expires_in: int = 3600
  refresh_token: Optional[str] = None
  scope: Optional[str] = None


@app.get("/.well-known/openid_configuration")
async def openid_configuration():
  """OpenID Connect Discovery endpoint."""
  return {
      "issuer": "http://localhost:8080",
      "authorization_endpoint": "http://localhost:8080/auth",
      "token_endpoint": "http://localhost:8080/token",
      "userinfo_endpoint": "http://localhost:8080/userinfo",
      "revocation_endpoint": "http://localhost:8080/revoke",
      "scopes_supported": ["openid", "read", "write", "admin"],
      "response_types_supported": ["code"],
      "grant_types_supported": ["authorization_code", "client_credentials"],
      "token_endpoint_auth_methods_supported": [
          "client_secret_basic",
          "client_secret_post",
      ],
      "subject_types_supported": ["public"],
  }


@app.get("/auth")
async def authorize(
    response_type: str = Query(...),
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    scope: str = Query(default="read"),
    state: str = Query(default=""),
):
  """Authorization endpoint for OAuth2 authorization code flow."""

  # Validate client
  if client_id not in clients:
    raise HTTPException(status_code=400, detail="Invalid client_id")

  client = clients[client_id]
  if redirect_uri not in client["redirect_uris"]:
    raise HTTPException(status_code=400, detail="Invalid redirect_uri")

  if response_type != "code":
    raise HTTPException(status_code=400, detail="Unsupported response_type")

  # Generate authorization code
  auth_code = secrets.token_urlsafe(32)
  authorization_codes[auth_code] = {
      "client_id": client_id,
      "redirect_uri": redirect_uri,
      "scope": scope,
      "expires_at": time.time() + 600,  # 10 minutes
  }

  # Simulate user consent - in real implementation, this would show a consent form
  params = f"code={auth_code}"
  if state:
    params += f"&state={state}"

  return RedirectResponse(url=f"{redirect_uri}?{params}")


@app.post("/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    client_id: str = Form(default=None),
    client_secret: str = Form(default=None),
    code: str = Form(default=None),
    redirect_uri: str = Form(default=None),
    scope: str = Form(default="read"),
):
  """Token endpoint for both client credentials and authorization code flows."""

  # Support both HTTP Basic auth and form-based client authentication
  auth_header = request.headers.get("Authorization")

  if auth_header and auth_header.startswith("Basic "):
    # HTTP Basic authentication
    import base64

    try:
      encoded_credentials = auth_header[6:]  # Remove "Basic " prefix
      decoded = base64.b64decode(encoded_credentials).decode("utf-8")
      basic_client_id, basic_client_secret = decoded.split(":", 1)
      client_id = client_id or basic_client_id
      client_secret = client_secret or basic_client_secret
    except Exception:
      raise HTTPException(
          status_code=401, detail="Invalid authorization header"
      )

  if not client_id or not client_secret:
    raise HTTPException(status_code=400, detail="Client credentials required")

  # Validate client credentials
  if client_id not in clients:
    raise HTTPException(status_code=401, detail="Invalid client")

  client = clients[client_id]
  if client["client_secret"] != client_secret:
    raise HTTPException(status_code=401, detail="Invalid client credentials")

  if grant_type == "client_credentials":
    return await handle_client_credentials(client_id, scope)
  elif grant_type == "authorization_code":
    return await handle_authorization_code(client_id, code, redirect_uri, scope)
  else:
    raise HTTPException(status_code=400, detail="Unsupported grant_type")


async def handle_client_credentials(
    client_id: str, scope: str
) -> TokenResponse:
  """Handle client credentials flow."""

  # Generate access token
  access_token = secrets.token_urlsafe(32)
  expires_at = time.time() + 3600  # 1 hour

  # Store token
  access_tokens[access_token] = {
      "client_id": client_id,
      "scope": scope,
      "expires_at": expires_at,
      "token_type": "Bearer",
  }

  return TokenResponse(
      access_token=access_token,
      token_type="Bearer",
      expires_in=3600,
      scope=scope,
  )


async def handle_authorization_code(
    client_id: str, code: str, redirect_uri: str, scope: str
) -> TokenResponse:
  """Handle authorization code flow."""

  if not code:
    raise HTTPException(status_code=400, detail="Missing authorization code")

  if code not in authorization_codes:
    raise HTTPException(status_code=400, detail="Invalid authorization code")

  auth_data = authorization_codes[code]

  # Validate authorization code
  if time.time() > auth_data["expires_at"]:
    del authorization_codes[code]
    raise HTTPException(status_code=400, detail="Authorization code expired")

  if auth_data["client_id"] != client_id:
    raise HTTPException(status_code=400, detail="Client mismatch")

  if redirect_uri and auth_data["redirect_uri"] != redirect_uri:
    raise HTTPException(status_code=400, detail="Redirect URI mismatch")

  # Generate tokens
  access_token = secrets.token_urlsafe(32)
  refresh_token = secrets.token_urlsafe(32)
  expires_at = time.time() + 3600  # 1 hour

  # Store token
  access_tokens[access_token] = {
      "client_id": client_id,
      "scope": auth_data["scope"],
      "expires_at": expires_at,
      "token_type": "Bearer",
  }

  # Clean up authorization code (one-time use)
  del authorization_codes[code]

  return TokenResponse(
      access_token=access_token,
      token_type="Bearer",
      expires_in=3600,
      refresh_token=refresh_token,
      scope=auth_data["scope"],
  )


@app.get("/api/weather")
async def get_weather(
    request: Request, city: str = "San Francisco", units: str = "metric"
):
  """Weather API endpoint that returns weather data for a city."""

  # Check authentication
  auth_header = request.headers.get("Authorization")
  if not auth_header or not auth_header.startswith("Bearer "):
    raise HTTPException(
        status_code=401, detail="Missing or invalid authorization header"
    )

  token = auth_header[7:]  # Remove "Bearer " prefix

  if token not in access_tokens:
    raise HTTPException(status_code=401, detail="Invalid access token")

  token_data = access_tokens[token]

  if time.time() > token_data["expires_at"]:
    del access_tokens[token]
    raise HTTPException(status_code=401, detail="Access token expired")

  # Return weather data (simulated)
  from datetime import datetime
  import random

  conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]

  weather_data = {
      "city": city,
      "temperature": random.randint(15, 30),
      "condition": random.choice(conditions),
      "humidity": random.randint(40, 80),
      "wind_speed": random.randint(5, 25),
      "timestamp": datetime.now().isoformat(),
      "units": units,
      "api_client": token_data["client_id"],
  }

  return weather_data


@app.get("/")
async def root():
  """Root endpoint with server information."""
  return HTMLResponse("""
    <html>
        <head><title>Weather API OAuth2 Server</title></head>
        <body>
            <h1>Weather API OAuth2 Server</h1>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><strong>GET /auth</strong> - Authorization endpoint</li>
                <li><strong>POST /token</strong> - Token endpoint</li>
                <li><strong>GET /.well-known/openid_configuration</strong> - Discovery</li>
                <li><strong>GET /api/weather</strong> - Weather API (requires Bearer token)</li>
            </ul>

            <h2>Test Client Credentials:</h2>
            <ul>
                <li><strong>Client ID:</strong> test_client</li>
                <li><strong>Client Secret:</strong> test_secret</li>
                <li><strong>Scopes:</strong> read, write, admin</li>
            </ul>

            <h2>Example cURL Commands:</h2>
            <h3>Client Credentials Flow:</h3>
            <pre>
curl -X POST http://localhost:8080/token \\
  -d "grant_type=client_credentials" \\
  -d "client_id=test_client" \\
  -d "client_secret=test_secret" \\
  -d "scope=read write"
            </pre>

            <h3>Test Weather API:</h3>
            <pre>
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\
  "http://localhost:8080/api/weather?city=Tokyo"
            </pre>
        </body>
    </html>
    """)


if __name__ == "__main__":
  import uvicorn

  print("🌤️  Starting Weather API OAuth2 Server...")
  print("📖 Documentation: http://localhost:8080/docs")
  print("🏠 Server Info: http://localhost:8080")
  print(
      '🔧 Test with: curl -H "Authorization: Bearer TOKEN"'
      ' "http://localhost:8080/api/weather?city=Tokyo"'
  )
  uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
