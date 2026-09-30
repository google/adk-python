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

from unittest import mock

from google.adk.tools.pubsub import client
from google.cloud import pubsub_v1
from google.oauth2.credentials import Credentials
import pytest

# Save original Pub/Sub classes before patching.
# This is necessary because create_autospec cannot be used on a mock object,
# and mock.patch.object(..., autospec=True) replaces the class with a mock.
# We need the original class to create spec'd mocks in side_effect.
ORIG_PUBLISHER = pubsub_v1.PublisherClient
ORIG_SUBSCRIBER = pubsub_v1.SubscriberClient


@pytest.fixture(autouse=True)
def cleanup_pubsub_clients():
  """Automatically clean up Pub/Sub client caches after each test.

  This fixture runs automatically for all tests in this file,
  ensuring that client caches are cleared between tests to prevent
  state leakage and ensure test isolation.
  """
  yield
  client.cleanup_clients()


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client(mock_publisher_client):
  """Test get_publisher_client factory."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  client.get_publisher_client(credentials=mock_creds)

  mock_publisher_client.assert_called_once()
  _, kwargs = mock_publisher_client.call_args
  assert kwargs["credentials"] == mock_creds
  assert "client_info" in kwargs
  assert isinstance(kwargs["batch_settings"], pubsub_v1.types.BatchSettings)
  assert kwargs["batch_settings"].max_messages == 1


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client_with_options(mock_publisher_client):
  """Test get_publisher_client factory with options."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  mock_options = mock.create_autospec(
      pubsub_v1.types.PublisherOptions, instance=True, spec_set=True
  )
  client.get_publisher_client(
      credentials=mock_creds, publisher_options=mock_options
  )

  mock_publisher_client.assert_called_once()
  _, kwargs = mock_publisher_client.call_args
  assert kwargs["credentials"] == mock_creds
  assert kwargs["publisher_options"] == mock_options
  assert "client_info" in kwargs
  assert isinstance(kwargs["batch_settings"], pubsub_v1.types.BatchSettings)
  assert kwargs["batch_settings"].max_messages == 1


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client_caching(mock_publisher_client):
  """Test get_publisher_client caching behavior."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  mock_publisher_client.side_effect = [
      mock.create_autospec(ORIG_PUBLISHER, instance=True, spec_set=True),
      mock.create_autospec(ORIG_PUBLISHER, instance=True, spec_set=True),
  ]

  # First call - should create client
  client1 = client.get_publisher_client(credentials=mock_creds)
  mock_publisher_client.assert_called_once()

  # Second call with same args - should return cached client
  client2 = client.get_publisher_client(credentials=mock_creds)
  assert client1 is client2
  mock_publisher_client.assert_called_once()  # Still called only once

  # Call with different args - should create new client
  mock_creds2 = mock.create_autospec(Credentials, instance=True, spec_set=True)
  client3 = client.get_publisher_client(credentials=mock_creds2)
  assert client3 is not client1
  assert mock_publisher_client.call_count == 2


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client_caching_equivalent_options(mock_publisher_client):
  """Equivalent but distinct options objects should share one client."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  mock_publisher_client.side_effect = [
      mock.create_autospec(ORIG_PUBLISHER, instance=True, spec_set=True)
      for _ in range(3)
  ]

  # A fresh options object per call, as publish_message builds one per message.
  clients = [
      client.get_publisher_client(
          credentials=mock_creds,
          user_agent=["my-project", "publish_message"],
          publisher_options=pubsub_v1.types.PublisherOptions(
              enable_message_ordering=False
          ),
      )
      for _ in range(3)
  ]

  assert mock_publisher_client.call_count == 1
  assert clients[0] is clients[1]
  assert clients[0] is clients[2]


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client_caching_different_options(mock_publisher_client):
  """Options that differ in value should not share a client."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  mock_publisher_client.side_effect = [
      mock.create_autospec(ORIG_PUBLISHER, instance=True, spec_set=True)
      for _ in range(2)
  ]

  unordered_client = client.get_publisher_client(
      credentials=mock_creds,
      publisher_options=pubsub_v1.types.PublisherOptions(
          enable_message_ordering=False
      ),
  )
  ordered_client = client.get_publisher_client(
      credentials=mock_creds,
      publisher_options=pubsub_v1.types.PublisherOptions(
          enable_message_ordering=True
      ),
  )

  assert mock_publisher_client.call_count == 2
  assert ordered_client is not unordered_client


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client_cache_is_bounded(mock_publisher_client):
  """The cache should evict rather than grow without bound."""
  all_creds = [
      mock.create_autospec(Credentials, instance=True, spec_set=True)
      for _ in range(client._CACHE_MAX_SIZE + 5)
  ]

  for creds in all_creds:
    client.get_publisher_client(credentials=creds)

  assert mock_publisher_client.call_count == len(all_creds)
  assert len(client._publisher_client_cache) == client._CACHE_MAX_SIZE


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client_cache_evicts_least_recently_used(
    mock_publisher_client,
):
  """A re-used entry should outlive an older one on the next eviction."""
  mock_publisher_client.side_effect = lambda *args, **kwargs: (
      mock.create_autospec(ORIG_PUBLISHER, instance=True, spec_set=True)
  )
  all_creds = [
      mock.create_autospec(Credentials, instance=True, spec_set=True)
      for _ in range(client._CACHE_MAX_SIZE)
  ]
  for creds in all_creds:
    client.get_publisher_client(credentials=creds)

  # Re-touch the oldest entry, then overflow the cache by one.
  oldest_client = client.get_publisher_client(credentials=all_creds[0])
  client.get_publisher_client(
      credentials=mock.create_autospec(
          Credentials, instance=True, spec_set=True
      )
  )
  call_count = mock_publisher_client.call_count

  # The re-touched entry survived, and the one after it was evicted instead.
  assert client.get_publisher_client(credentials=all_creds[0]) is oldest_client
  assert mock_publisher_client.call_count == call_count
  client.get_publisher_client(credentials=all_creds[1])
  assert mock_publisher_client.call_count == call_count + 1


@mock.patch.object(pubsub_v1, "PublisherClient", autospec=True)
def test_get_publisher_client_unhashable_options(mock_publisher_client):
  """Options that cannot be hashed should be built fresh, never cached."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  mock_publisher_client.side_effect = [
      mock.create_autospec(ORIG_PUBLISHER, instance=True, spec_set=True)
      for _ in range(2)
  ]
  # `retry` takes an arbitrary object, so a list makes the options unhashable.
  unhashable_options = pubsub_v1.types.PublisherOptions(retry=[])

  client1 = client.get_publisher_client(
      credentials=mock_creds, publisher_options=unhashable_options
  )
  client2 = client.get_publisher_client(
      credentials=mock_creds, publisher_options=unhashable_options
  )

  assert mock_publisher_client.call_count == 2
  assert client1 is not client2
  assert not client._publisher_client_cache


@mock.patch.object(pubsub_v1, "SubscriberClient", autospec=True)
def test_get_subscriber_client(mock_subscriber_client):
  """Test get_subscriber_client factory."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  client.get_subscriber_client(credentials=mock_creds)

  mock_subscriber_client.assert_called_once()
  _, kwargs = mock_subscriber_client.call_args
  assert kwargs["credentials"] == mock_creds
  assert "client_info" in kwargs


@mock.patch.object(pubsub_v1, "SubscriberClient", autospec=True)
def test_get_subscriber_client_caching(mock_subscriber_client):
  """Test get_subscriber_client caching behavior."""
  mock_creds = mock.create_autospec(Credentials, instance=True, spec_set=True)
  mock_subscriber_client.side_effect = [
      mock.create_autospec(ORIG_SUBSCRIBER, instance=True, spec_set=True),
      mock.create_autospec(ORIG_SUBSCRIBER, instance=True, spec_set=True),
  ]

  # First call - should create client
  client1 = client.get_subscriber_client(credentials=mock_creds)
  mock_subscriber_client.assert_called_once()

  # Second call with same args - should return cached client
  client2 = client.get_subscriber_client(credentials=mock_creds)
  assert client1 is client2
  mock_subscriber_client.assert_called_once()  # Still called only once

  # Call with different args - should create new client
  mock_creds2 = mock.create_autospec(Credentials, instance=True, spec_set=True)
  client3 = client.get_subscriber_client(credentials=mock_creds2)
  assert client3 is not client1
  assert mock_subscriber_client.call_count == 2
