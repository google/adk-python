# Copyright 2025 Google LLC
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

import threading
import time

from google.api_core.gapic_v1.client_info import ClientInfo
from google.auth.credentials import Credentials
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import BatchSettings

from ... import version

USER_AGENT = f"adk-pubsub-tool google-adk/{version.__version__}"

_CACHE_TTL = 1800  # 30 minutes

_publisher_client_cache = {}
_publisher_client_lock = threading.Lock()


def get_publisher_client(
    *,
    credentials: Credentials,
    user_agent: str | list[str] | None = None,
    publisher_options: pubsub_v1.types.PublisherOptions | None = None,
) -> pubsub_v1.PublisherClient:
  """Get a Pub/Sub Publisher client.

  Args:
    credentials: The credentials to use for the request.
    user_agent: The user agent to use for the request.
    publisher_options: The publisher options to use for the request.

  Returns:
    A Pub/Sub Publisher client.
  """
  global _publisher_client_cache
  current_time = time.time()

  user_agents_key = None
  if user_agent:
    if isinstance(user_agent, str):
      user_agents_key = (user_agent,)
    else:
      user_agents_key = tuple(user_agent)

  # Use object identity for credentials and publisher_options as they are not hashable
  key = (id(credentials), user_agents_key, id(publisher_options))

  with _publisher_client_lock:
    if key in _publisher_client_cache:
      client, expiration = _publisher_client_cache[key]
      if expiration > current_time:
        return client

    user_agents = [USER_AGENT]
    if user_agent:
      if isinstance(user_agent, str):
        user_agents.append(user_agent)
      else:
        user_agents.extend(ua for ua in user_agent if ua)

    client_info = ClientInfo(user_agent=" ".join(user_agents))

    # Since we synchronously publish messages, we want to disable batching to
    # remove any delay.
    custom_batch_settings = BatchSettings(max_messages=1)
    publisher_client = pubsub_v1.PublisherClient(
        credentials=credentials,
        client_info=client_info,
        publisher_options=publisher_options,
        batch_settings=custom_batch_settings,
    )

    _publisher_client_cache[key] = (publisher_client, current_time + _CACHE_TTL)

    return publisher_client


_subscriber_client_cache = {}
_subscriber_client_lock = threading.Lock()


def get_subscriber_client(
    *,
    credentials: Credentials,
    user_agent: str | list[str] | None = None,
) -> pubsub_v1.SubscriberClient:
  """Get a Pub/Sub Subscriber client.

  Args:
    credentials: The credentials to use for the request.
    user_agent: The user agent to use for the request.

  Returns:
    A Pub/Sub Subscriber client.
  """
  global _subscriber_client_cache
  current_time = time.time()

  user_agents_key = None
  if user_agent:
    if isinstance(user_agent, str):
      user_agents_key = (user_agent,)
    else:
      user_agents_key = tuple(user_agent)

  # Use object identity for credentials as they are not hashable
  key = (id(credentials), user_agents_key)

  with _subscriber_client_lock:
    if key in _subscriber_client_cache:
      client, expiration = _subscriber_client_cache[key]
      if expiration > current_time:
        return client

    user_agents = [USER_AGENT]
    if user_agent:
      if isinstance(user_agent, str):
        user_agents.append(user_agent)
      else:
        user_agents.extend(ua for ua in user_agent if ua)

    client_info = ClientInfo(user_agent=" ".join(user_agents))

    subscriber_client = pubsub_v1.SubscriberClient(
        credentials=credentials,
        client_info=client_info,
    )

    _subscriber_client_cache[key] = (
        subscriber_client,
        current_time + _CACHE_TTL,
    )

    return subscriber_client


def cleanup_clients():
  """Clean up all cached Pub/Sub clients."""
  global _publisher_client_cache, _subscriber_client_cache

  with _publisher_client_lock:
    for client, _ in _publisher_client_cache.values():
      client.transport.close()
    _publisher_client_cache.clear()

  with _subscriber_client_lock:
    for client, _ in _subscriber_client_cache.values():
      client.close()
    _subscriber_client_cache.clear()
