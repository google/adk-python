# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest import mock
import urllib.parse

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.testclient import TestClient
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.cli.fast_api import get_fast_api_app
import httpx  # For mocking userinfo response
import pytest

# Mock environment variables for OAuth client ID and secret
MOCK_CLIENT_ID = "test_client_id_123"
MOCK_CLIENT_SECRET = "test_client_secret_456"
MOCK_USERINFO = {"email": "test.user@example.com", "name": "Test User"}
MOCK_OAUTH_STATE = "mock_state_for_testing"
MOCK_AUTH_CODE = "mock_authorization_code"
MOCK_OAUTH_REDIRECT_URI = "http://testserver/auth/google/callback"
MOCK_SESSION_SECRET = "secret"


@pytest.fixture(scope="module")
def test_app_client():
  """Fixture to create a TestClient for the FastAPI app with web=True."""
  # Patch os.environ for the duration of this fixture setup
  # Also mock starlette.config.Config to ensure these values are used
  with mock.patch.dict(
      os.environ,
      {
          "GOOGLE_CLIENT_ID": MOCK_CLIENT_ID,
          "GOOGLE_CLIENT_SECRET": MOCK_CLIENT_SECRET,
          "GOOGLE_OAUTH_REDIRECT_URI": MOCK_OAUTH_REDIRECT_URI,
          "SESSION_SECRET": MOCK_SESSION_SECRET,
      },
  ), mock.patch(
      "google.adk.cli.fast_api.config",
      new=lambda key, default=None: os.environ.get(key, default),
  ):
    app = get_fast_api_app(
        agents_dir=".", web=True
    )  # Assuming a dummy agents_dir

    @app.get("/test-auth-check")
    async def _test_auth_check(request: Request):
      if "user" in request.session:
        return JSONResponse({
            "authenticated": True,
            "user_email": request.session["user"].get("email"),
        })
      return RedirectResponse("/login", status_code=302)

    with TestClient(app) as client:
      yield client


@pytest.mark.asyncio
@mock.patch("google.adk.auth.auth_handler.AuthHandler.generate_auth_uri")
async def test_login_redirects_to_google(
    mock_generate_auth_uri: mock.MagicMock, test_app_client: TestClient
):
  """Tests if /login redirects to Google's authorization URL and attempts to set state."""
  mock_auth_uri_response = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          auth_uri=f"https://accounts.google.com/o/oauth2/v2/auth?client_id={MOCK_CLIENT_ID}&redirect_uri={urllib.parse.quote(MOCK_OAUTH_REDIRECT_URI)}&response_type=code&scope=openid+email+profile+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform&state={MOCK_OAUTH_STATE}",
          state=MOCK_OAUTH_STATE,
      ),
  )
  mock_generate_auth_uri.return_value = mock_auth_uri_response

  response = test_app_client.get("/login", follow_redirects=False)

  assert response.status_code == 302
  redirect_location = response.headers["location"]
  assert "https://accounts.google.com/o/oauth2/v2/auth" in redirect_location

  parsed_redirect_url = urllib.parse.urlparse(redirect_location)
  query_params = urllib.parse.parse_qs(parsed_redirect_url.query)

  assert query_params["client_id"][0] == MOCK_CLIENT_ID
  # TestClient uses http://testserver as base_url
  assert query_params["redirect_uri"][0] == MOCK_OAUTH_REDIRECT_URI
  assert query_params["response_type"][0] == "code"
  assert (
      query_params["scope"][0]
      == "openid email profile https://www.googleapis.com/auth/cloud-platform"
  )
  assert query_params["state"][0] == MOCK_OAUTH_STATE

  # The state should be in the session after this,
  # which will be used by the callback test.
  # We can verify its use in the callback test.


@pytest.mark.asyncio
@mock.patch("google.adk.auth.auth_handler.AuthHandler.exchange_auth_token")
@mock.patch("httpx.AsyncClient.get")
async def test_auth_google_callback_success_and_session(
    mock_httpx_get: mock.MagicMock,
    mock_exchange_auth_token: mock.MagicMock,
    test_app_client: TestClient,
):
  """Tests successful OAuth callback, token exchange, userinfo fetch, and session setup."""
  # 1. Setup: Ensure 'oauth_state' is in the session by calling /login first
  # We mock generate_auth_uri to control the state and auth_uri
  mock_auth_uri_response_for_login = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          auth_uri="mock_google_auth_uri", state=MOCK_OAUTH_STATE
      ),
  )
  with mock.patch(
      "google.adk.auth.auth_handler.AuthHandler.generate_auth_uri",
      return_value=mock_auth_uri_response_for_login,
  ):
    login_response = test_app_client.get("/login", follow_redirects=False)
    assert login_response.status_code == 302  # Login sets the state in session

  # 2. Configure mocks for the callback flow
  # AuthHandler.exchange_auth_token returns the exchanged AuthCredential (as
  # dict for session)
  mock_exchanged_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          access_token="mock_access_token_123",
          refresh_token="mock_refresh_token_456",
          expires_at=1700000000,
      ),
  )
  # Note: parse_and_store_auth_response expects exchange_auth_token to return
  # AuthCredential obj
  mock_exchange_auth_token.return_value = mock_exchanged_credential

  # httpx.AsyncClient.get for userinfo
  mock_userinfo_response = httpx.Response(200, json=MOCK_USERINFO)
  mock_httpx_get.return_value = mock_userinfo_response

  # 3. Make the call to the callback endpoint
  callback_response = test_app_client.get(
      f"/auth/google/callback?code={MOCK_AUTH_CODE}&state={MOCK_OAUTH_STATE}",
      follow_redirects=False,  # We want to check the redirect itself
  )

  # 4. Assertions for callback
  assert callback_response.status_code == 302
  assert (
      callback_response.headers["location"] == "/"
  )  # Redirects to root on success

  mock_exchange_auth_token.assert_called_once()
  mock_httpx_get.assert_called_once()
  userinfo_url_called = mock_httpx_get.call_args[0][0]
  assert "openidconnect.googleapis.com/v1/userinfo" in userinfo_url_called
  userinfo_headers_called = mock_httpx_get.call_args[1]["headers"]
  assert (
      userinfo_headers_called["Authorization"] == "Bearer mock_access_token_123"
  )

  # 5. Verify session by accessing a protected route (/list-projects)
  auth_check_response = test_app_client.get("/test-auth-check")
  assert auth_check_response.status_code == 200
  assert auth_check_response.json() == {
      "authenticated": True,
      "user_email": MOCK_USERINFO["email"],
  }


@pytest.mark.asyncio
async def test_logout_clears_session_and_redirects(test_app_client: TestClient):
  """Tests if /logout clears the session and redirects, then protected route fails."""
  # 1. First, establish a logged-in session (simplified setup)
  # We'll manually set session cookies that would exist
  # after a successful login.
  # This is a bit of a shortcut for an integration test,
  # but avoids re-running the full login mock setup.
  # A more robust way would be to call the login/callback
  # sequence with all mocks.
  # For this test, let's ensure some token and user exist in
  # session before logout.

  # To set session, we make a call that would set it, like the
  # end of a mocked callback.
  # This requires all mocks for callback to be in place.
  with mock.patch(
      "google.adk.auth.auth_handler.AuthHandler.generate_auth_uri",
      return_value=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              auth_uri="mock_uri", state="login_for_logout_state"
          ),
      ),
  ), mock.patch(
      "google.adk.auth.auth_handler.AuthHandler.exchange_auth_token",
      return_value=AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(access_token="logout_token"),
      ),
  ), mock.patch(
      "httpx.AsyncClient.get",
      return_value=httpx.Response(200, json=MOCK_USERINFO),
  ):

    test_app_client.get("/login", follow_redirects=False)  # Sets oauth_state
    test_app_client.get(
        "/auth/google/callback?code=anycode&state=login_for_logout_state",
        follow_redirects=True,
    )  # Completes login, sets user/token

  # Verify logged-in state (optional, but good for confidence)
  auth_check_response_before_logout = test_app_client.get("/test-auth-check")
  assert auth_check_response_before_logout.status_code == 200
  assert auth_check_response_before_logout.json()["authenticated"]

  # 2. Perform logout
  logout_response = test_app_client.get("/logout", follow_redirects=False)
  assert logout_response.status_code == 302
  assert logout_response.headers["location"] == "/"

  # 3. Verify session is cleared by trying to access a protected route
  # /list-projects should now redirect to /login
  auth_check_response_after_logout = test_app_client.get(
      "/test-auth-check", follow_redirects=False
  )
  assert auth_check_response_after_logout.status_code == 302
  assert auth_check_response_after_logout.headers["location"] == "/login"
