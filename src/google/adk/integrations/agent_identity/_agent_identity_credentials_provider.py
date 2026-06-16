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

"""Credentials Provider using the Agent Identity service."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.api_core.client_options import ClientOptions

try:
  from google.cloud.agentidentitycredentials_v1 import AuthProviderCredentialsServiceClient as Client
  from google.cloud.agentidentitycredentials_v1 import RetrieveCredentialsRequest
  from google.cloud.agentidentitycredentials_v1 import RetrieveCredentialsResponse
except ImportError as e:
  raise ImportError(
      "Missing required dependencies for Agent Identity Auth Manager. "
      'Please install with: pip install "google-adk[agent-identity]"'
  ) from e

from .gcp_auth_provider_scheme import GcpAuthProviderScheme

# TODO: Catch specific exceptions instead of generic ones.

logger = logging.getLogger("google_adk." + __name__)

NON_INTERACTIVE_TOKEN_POLL_INTERVAL_SEC: float = 1.0
NON_INTERACTIVE_TOKEN_POLL_TIMEOUT_SEC: float = 10.0


def _construct_auth_credential(
    response: RetrieveCredentialsResponse,
) -> AuthCredential:
  """Constructs a simplified HTTP auth credential from the header-token tuple
  returned by the upstream service.
  """
  if not response.success.header or not response.success.token:
    raise ValueError(
        "Received either empty header or token from Agent Identity"
        " Credentials service."
    )

  header_name, _, header_value = response.success.header.partition(":")
  if (
      header_name.strip().lower() == "authorization"
      and header_value.strip().lower().startswith("bearer")
  ):
    return AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="Bearer",
            credentials=HttpCredentials(token=response.success.token),
        ),
    )

  # Handle custom header.
  return AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          # For custom headers, scheme and credentials fields are not used.
          scheme="",
          credentials=HttpCredentials(),
          additional_headers={
              response.success.header: response.success.token,
              "X-GOOG-API-KEY": response.success.token,
          },
      ),
  )


class _AgentIdentityCredentialsProvider:
  """Auth provider implementation using Agent Identity credentials service."""

  _client: Client | None = None

  def __init__(self, client: Client | None = None):
    self._client = client

  def _get_client(self) -> Client:
    """Lazy loads the client to avoid unnecessary setup on startup."""
    if self._client is None:
      client_options = None
      if host := os.environ.get("AGENT_IDENTITY_CREDENTIALS_TARGET_HOST"):
        client_options = ClientOptions(api_endpoint=host)
      self._client = Client(client_options=client_options, transport="rest")
    return self._client

  async def _retrieve_credentials(
      self,
      user_id: str,
      auth_scheme: GcpAuthProviderScheme,
  ) -> RetrieveCredentialsResponse:
    request = RetrieveCredentialsRequest(
        auth_provider=auth_scheme.name,
        user_id=user_id,
        scopes=auth_scheme.scopes,
        continue_uri=auth_scheme.continue_uri or "",
    )
    # TODO: Use async client once available. Temporarily using threading to
    # prevent blocking the event loop.
    return await asyncio.to_thread(
        self._get_client().retrieve_credentials, request
    )

  async def _poll_credentials(
      self, user_id: str, auth_scheme: GcpAuthProviderScheme, timeout: float
  ) -> RetrieveCredentialsResponse:
    end_time = time.time() + timeout
    while time.time() < end_time:
      response = await self._retrieve_credentials(user_id, auth_scheme)
      if (
          "success" in response
          or "uri_consent_required" in response
          or "consent_rejected" in response
      ):
        return response
      await asyncio.sleep(NON_INTERACTIVE_TOKEN_POLL_INTERVAL_SEC)
    raise TimeoutError("Timeout waiting for credentials.")

  @staticmethod
  def _is_consent_completed(context: CallbackContext) -> bool:
    """Checks if the user consent flow is completed for the current function

    call.
    """
    if not context.function_call_id:
      return False

    if not context.session:
      return False

    events = context.session.events
    target_tool_call_id = context.function_call_id

    # Find all relevant function calls and responses
    euc_calls = {}
    euc_responses = {}

    for event in events:
      for call in event.get_function_calls():
        if call.name == REQUEST_EUC_FUNCTION_CALL_NAME:
          euc_calls[call.id] = call
      for response in event.get_function_responses():
        if response.name == REQUEST_EUC_FUNCTION_CALL_NAME:
          euc_responses[response.id] = response

    # Check for a response that matches a call for the current tool invocation.
    for call_id, _ in euc_responses.items():
      if call_id in euc_calls:
        call = euc_calls[call_id]
        if call.args and call.args.get("functionCallId") == target_tool_call_id:
          return True
    return False

  async def get_auth_credential(
      self,
      auth_scheme: GcpAuthProviderScheme,
      context: CallbackContext | None = None,
  ) -> AuthCredential:
    """Retrieves credentials using the Agent Identity Credentials service.

    Args:
      auth_scheme: The GcpAuthProviderScheme.
      context: Optional context for the callback.

    Returns:
      An AuthCredential instance.

    Raises:
      RuntimeError: If credential retrieval or polling fails.
    """

    if context is None or context.user_id is None:
      raise ValueError(
          "GcpAuthProvider requires a context with a valid user_id."
      )

    user_id = context.user_id

    try:
      response = await self._retrieve_credentials(user_id, auth_scheme)
    except Exception as e:
      raise RuntimeError(
          f"Failed to retrieve credential for user '{user_id}' on"
          f" provider '{auth_scheme.name}'."
      ) from e

    if "consent_rejected" in response:
      raise RuntimeError("Operation failed: User consent rejected.")

    if "success" in response:
      logger.debug("Auth credential obtained immediately.")
      return _construct_auth_credential(response)

    if "pending" in response:
      # Get 2-legged OAuth token. Allow enough time for token exchange.
      try:
        response = await self._poll_credentials(
            user_id,
            auth_scheme,
            timeout=NON_INTERACTIVE_TOKEN_POLL_TIMEOUT_SEC,
        )
        if "consent_rejected" in response:
          raise RuntimeError("Operation failed: User consent rejected.")
        if "success" in response:
          logger.debug("Auth credential obtained after polling.")
          return _construct_auth_credential(response)
      except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve credential for user '{user_id}' on"
            f" provider '{auth_scheme.name}'."
        ) from e

    if "uri_consent_required" in response:
      if self._is_consent_completed(context):
        raise RuntimeError("Failed to retrieve consent based credential.")

      # Return AuthCredential with only auth_uri to trigger user consent
      # flow.
      return AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              auth_uri=response.uri_consent_required.authorization_uri,
              nonce=response.uri_consent_required.consent_nonce,
          ),
      )
