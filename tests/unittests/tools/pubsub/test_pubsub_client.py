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
