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
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.base_auth_provider import BaseAuthProvider
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.api_core.client_options import ClientOptions
from google.cloud.iamconnectorcredentials_v1alpha import IAMConnectorCredentialsServiceClient as Client
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsMetadata
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsRequest
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsResponse
from google.longrunning.operations_pb2 import Operation
from typing_extensions import override

from .gcp_auth_provider_scheme import GcpAuthProviderScheme

# Notes on the current Agent Identity Credentials service implementation:
# 1. The service does not yet support LROs, so even though the
#    retrieve_credentials method returns an Operation object, the methods like
#    operation.done() and operation.result() will not work yet.
# 2. For API key flows, the returned Operation contains the credentials.
# 3. For 2-legged OAuth flows, the returned Operation contains pending status,
#    client needs to retry the request until response with credentials is
#    returned or timeout occurs.
# 4. For 3-legged OAuth flows, the returned Operation contains consent pending
#    status along with the authorization URI.

# TODO: Catch specific exceptions instead of generic ones.

logger = logging.getLogger("google_adk." + __name__)

NON_INTERACTIVE_TOKEN_POLL_INTERVAL_SEC: float = 1.0
NON_INTERACTIVE_TOKEN_POLL_TIMEOUT_SEC: float = 10.0


def _construct_auth_credential(
    response: RetrieveCredentialsResponse,
) -> AuthCredential:
  """Constructs a simplified HTTP auth credential from the header-token tuple returned by the upstream service."""
  if not response.header or not response.token:
    raise ValueError(
        "Received either empty header or token from Agent Identity Credentials"
        " service."
    )

  header_name, _, header_value = response.header.partition(":")
  if (
      header_name.strip().lower() == "authorization"
      and header_value.strip().lower().startswith("bearer")
  ):
    return AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(token=response.token),
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
              response.header: response.token,
              "X-GOOG-API-KEY": response.token,
          },
      ),
  )


class GcpAuthProvider(BaseAuthProvider):
  """An auth provider that uses the Agent Identity Credentials service to generate access tokens."""

  _client: Client | None = None

  def __init__(self, client: Client | None = None):
    self._client = client

  @property
  @override
  def supported_auth_schemes(self) -> tuple[type[GcpAuthProviderScheme], ...]:
    return (GcpAuthProviderScheme,)

  def _get_client(self) -> Client:
    """Lazy loads the client to avoid unnecessary setup on startup."""
    if self._client is None:
      client_options = None
      if host := os.environ.get("IAM_CONNECTOR_CREDENTIALS_TARGET_HOST"):
        client_options = ClientOptions(api_endpoint=host)
      self._client = Client(client_options=client_options, transport="rest")
    return self._client

  async def _retrieve_credentials(
      self,
      user_id: str,
      auth_scheme: GcpAuthProviderScheme,
  ) -> Operation:
    request = RetrieveCredentialsRequest(
        connector=auth_scheme.name,
        user_id=user_id,
        scopes=auth_scheme.scopes,
        continue_uri=auth_scheme.continue_uri or "",
        force_refresh=False,
    )
    # TODO: Use async client once available. Temporarily using threading to
    # prevent blocking the event loop.
    operation = await asyncio.to_thread(
        self._get_client().retrieve_credentials, request
    )
    return operation.operation

  def _unpack_operation(
      self, operation: Operation
  ) -> tuple[
      RetrieveCredentialsResponse | None, RetrieveCredentialsMetadata | None
  ]:
    """Deserializes the response and metadata from the operation."""
    response = None
    metadata = None
    if operation.response:
      response = RetrieveCredentialsResponse.deserialize(
          operation.response.value
      )
    if operation.metadata:
      metadata = RetrieveCredentialsMetadata.deserialize(
          operation.metadata.value
      )
    return response, metadata

  async def _poll_credentials(
      self, user_id: str, auth_scheme: GcpAuthProviderScheme, timeout: float
  ) -> Operation:
    end_time = time.time() + timeout
    while time.time() < end_time:
      operation = await self._retrieve_credentials(user_id, auth_scheme)
      if operation.done:
        return operation
      await asyncio.sleep(NON_INTERACTIVE_TOKEN_POLL_INTERVAL_SEC)
    raise TimeoutError("Timeout waiting for credentials.")

  @staticmethod
  def _is_consent_completed(context: CallbackContext) -> bool:
    """Checks if the user consent flow is completed for the current function call."""
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

    # Check for a response that matches a call for the current tool invocation
    for call_id, _ in euc_responses.items():
      if call_id in euc_calls:
        call = euc_calls[call_id]
        if (
            call.args
            and call.args.get("function_call_id") == target_tool_call_id
        ):
          return True
    return False

  @override
  async def get_auth_credential(
      self,
      auth_config: AuthConfig,
      context: CallbackContext | None = None,
  ) -> AuthCredential:
    """Retrieves credentials using the Agent Identity Credentials service.

    Args:
      auth_config: The authentication configuration.
      context: Optional context for the callback.

    Returns:
      An AuthCredential instance.

    Raises:
      ValueError: If auth_scheme is not a GcpAuthProviderScheme.
      RuntimeError: If credential retrieval or polling fails.
    """

    auth_scheme = auth_config.auth_scheme
    if not isinstance(auth_scheme, GcpAuthProviderScheme):
      raise ValueError(
          f"Expected GcpAuthProviderScheme, got {type(auth_scheme)}"
      )

    if context is None or context.user_id is None:
      raise ValueError(
          "GcpAuthProvider requires a context with a valid user_id."
      )

    user_id = context.user_id

    try:
      operation = await self._retrieve_credentials(user_id, auth_scheme)
    except Exception as e:
      raise RuntimeError(
          f"Failed to retrieve credential for user '{user_id}' on connector"
          f" '{auth_scheme.name}'."
      ) from e

    response, metadata = self._unpack_operation(operation)

    if operation.HasField("error"):
      raise RuntimeError(f"Operation failed: {operation.error.message}")

    if operation.done:
      logger.debug("Auth credential obtained immediately.")
      return _construct_auth_credential(response)

    if metadata and metadata.consent_pending:
      # Get 2-legged OAuth token. Allow enough time for token exchange.
      try:
        operation = await self._poll_credentials(
            user_id,
            auth_scheme,
            timeout=NON_INTERACTIVE_TOKEN_POLL_TIMEOUT_SEC,
        )
        if operation.HasField("error"):
          raise RuntimeError(f"Operation failed: {operation.error.message}")
        if operation.done:
          logger.debug("Auth credential obtained after polling.")
          response, _ = self._unpack_operation(operation)
          return _construct_auth_credential(response)
      except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve credential for user '{user_id}' on connector"
            f" '{auth_scheme.name}'."
        ) from e

    if metadata is not None and metadata.uri_consent_required:
      if self._is_consent_completed(context):
        raise RuntimeError("Failed to retrieve consent based credential.")

      # Return AuthCredential with only auth_uri to trigger user consent flow.
      return AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          oauth2=OAuth2Auth(
              auth_uri=metadata.uri_consent_required.authorization_uri,
              nonce=metadata.uri_consent_required.consent_nonce,
          ),
      )
