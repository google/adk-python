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

"""Client caching and management for Eventarc publishing."""

from __future__ import annotations

import collections
import hashlib
import inspect
import os
import threading
import time
import typing

from google.api_core.gapic_v1 import client_info

if typing.TYPE_CHECKING:
  from google.cloud import eventarc_publishing_v1  # type: ignore
  from google.cloud.eventarc_publishing_v1 import PublisherAsyncClient  # type: ignore
else:
  try:
    from google.cloud import eventarc_publishing_v1  # type: ignore

    PublisherAsyncClient = getattr(
        eventarc_publishing_v1, "PublisherAsyncClient", typing.Any
    )
  except ImportError:
    eventarc_publishing_v1 = None
    PublisherAsyncClient = typing.Any

try:
  from google.adk import version  # type: ignore

  USER_AGENT = f"adk-eventarc-tool google-adk/{version.__version__}"
except ImportError:
  USER_AGENT = "adk-eventarc-tool google-adk/0.0.0"

_CACHE_TTL = 1800  # 30 minutes
_CACHE_MAX_SIZE = 10

_publisher_client_cache: collections.OrderedDict[
    tuple[str | None, str, int, str], tuple[PublisherAsyncClient, float]
] = collections.OrderedDict()
_publisher_client_lock = threading.Lock()


async def _close_client(client: typing.Any) -> None:
  """Explicitly closes the gRPC transport channel of the client."""
  transport = getattr(client, "transport", None)
  if transport is not None and hasattr(transport, "close"):
    try:
      res = transport.close()
      if inspect.isawaitable(res):
        await res
    except Exception:  # pylint: disable=broad-except
      pass


def _get_credential_id(credentials: typing.Any) -> str:
  """Returns a string identifier for a credentials object in O(1) time."""
  if (
      getattr(credentials.__class__, "__module__", "")
      == "google.auth.compute_engine.credentials"
  ):
    return "ComputeEngineCredentials"

  sa_email = getattr(credentials, "service_account_email", None)
  if sa_email is not None:
    # This covers both standard ServiceAccountCredentials and ImpersonatedCredentials
    return str(sa_email)

  # Handle User Credentials (like local ADC) using refresh token hash
  module_name = getattr(credentials.__class__, "__module__", "")
  if (
      module_name.startswith("google.oauth2.credentials")
      and getattr(credentials, "refresh_token", None) is not None
  ):
    token_bytes = credentials.refresh_token.encode("utf-8")
    token_hash = hashlib.sha256(token_bytes).hexdigest()
    return f"UserCredentials:{token_hash}"

  # Handle Downscoped Credentials recursively
  if module_name == "google.auth.downscoped":
    source = getattr(credentials, "_source_credentials", None)
    boundary = getattr(credentials, "_credential_access_boundary", None)
    if source is not None and boundary is not None:
      source_id = _get_credential_id(source)
      boundary_json = boundary.to_json()
      if isinstance(boundary_json, dict):
        boundary_str = str(sorted(boundary_json.items()))
      else:
        boundary_str = str(boundary_json)
      return f"Downscoped:{source_id}:{boundary_str}"

  # Handle external account credentials (Workload Identity, Pluggable, etc.)
  if getattr(credentials, "_audience", None) is not None:
    audience = credentials._audience
    source = getattr(credentials, "_credential_source", None)
    supplier = getattr(credentials, "_subject_token_supplier", None)

    if source is not None:
      if isinstance(source, dict):
        source_str = str(sorted(source.items()))
      else:
        source_str = str(source)
      return f"ExternalAccount:{audience}:{source_str}"
    elif supplier is not None:
      return f"ExternalAccount:{audience}:supplier:{id(supplier)}"

  return str(id(credentials))


def _get_final_user_agent(user_agent: str | None) -> str:
  return f"{USER_AGENT} {user_agent}" if user_agent else USER_AGENT


def _get_cache_key(
    credentials: typing.Any,
    final_user_agent: str,
    project_id: str | None = None,
) -> tuple[str | None, str, int, str]:
  """Generates a deterministic cache key for the publisher client connection pool."""
  cred_id = _get_credential_id(credentials)
  return (project_id, final_user_agent, os.getpid(), cred_id)


async def get_publisher_client(
    *,
    credentials: typing.Any,
    user_agent: str | None = None,
    project_id: str | None = None,
) -> PublisherAsyncClient:
  """Gets or creates a publisher client for Eventarc."""
  if eventarc_publishing_v1 is None:
    raise RuntimeError("google-cloud-eventarc-publishing is not installed")

  final_user_agent = _get_final_user_agent(user_agent)
  cache_key = _get_cache_key(
      credentials=credentials,
      final_user_agent=final_user_agent,
      project_id=project_id,
  )
  current_time = time.time()
  old_client_to_close = None
  with _publisher_client_lock:
    client_entry = _publisher_client_cache.get(cache_key)
    if client_entry is not None:
      client, expiration = client_entry
      if expiration > current_time:
        _publisher_client_cache.move_to_end(cache_key)
        return client
      else:
        _publisher_client_cache.pop(cache_key, None)

    info = client_info.ClientInfo(user_agent=final_user_agent)  # type: ignore[no-untyped-call]

    client = typing.cast(
        PublisherAsyncClient,
        eventarc_publishing_v1.PublisherAsyncClient(
            credentials=credentials,
            client_info=info,
        ),
    )

    if len(_publisher_client_cache) >= _CACHE_MAX_SIZE:
      _, (old_client_to_close, _) = _publisher_client_cache.popitem(last=False)

    _publisher_client_cache[cache_key] = (client, current_time + _CACHE_TTL)

  if old_client_to_close is not None:
    await _close_client(old_client_to_close)

  return client


async def remove_publisher_client(
    *,
    credentials: typing.Any,
    user_agent: str | None = None,
    project_id: str | None = None,
) -> None:
  """Removes a publisher client from the cache."""
  final_user_agent = _get_final_user_agent(user_agent)
  cache_key = _get_cache_key(
      credentials=credentials,
      final_user_agent=final_user_agent,
      project_id=project_id,
  )

  with _publisher_client_lock:
    entry = _publisher_client_cache.pop(cache_key, None)
  if entry is not None:
    await _close_client(entry[0])


async def cleanup_clients() -> None:
  """Cleans up all cached publisher clients."""
  with _publisher_client_lock:
    clients = list(_publisher_client_cache.values())
    _publisher_client_cache.clear()
  for client, _ in clients:
    await _close_client(client)
