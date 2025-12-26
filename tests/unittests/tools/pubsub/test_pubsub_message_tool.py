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

import os
from unittest import mock

from google.adk.tools.pubsub import client as pubsub_client_lib
from google.adk.tools.pubsub import message_tool
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.api_core import future
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1 import types
from google.oauth2.credentials import Credentials
from google.protobuf import timestamp_pb2


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_v1.PublisherClient, "publish", autospec=True)
@mock.patch.object(pubsub_client_lib, "get_publisher_client", autospec=True)
def test_publish_message(mock_get_publisher_client, mock_publish):
  """Test publish_message tool invocation."""
  topic_name = "projects/my_project_id/topics/my_topic"
  message = "Hello World"
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_publisher_client = mock.create_autospec(
      pubsub_v1.PublisherClient, instance=True
  )
  mock_get_publisher_client.return_value = mock_publisher_client

  mock_future = mock.create_autospec(future.Future, instance=True)
  mock_future.result.return_value = "message_id"
  mock_publisher_client.publish.return_value = mock_future

  result = message_tool.publish_message(
      topic_name, message, mock_credentials, tool_settings
  )

  assert result["message_id"] == "message_id"
  mock_get_publisher_client.assert_called_once()
  mock_publisher_client.publish.assert_called_once()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_v1.PublisherClient, "publish", autospec=True)
@mock.patch.object(pubsub_client_lib, "get_publisher_client", autospec=True)
def test_publish_message_with_ordering_key(
    mock_get_publisher_client, mock_publish
):
  """Test publish_message tool invocation with ordering_key."""
  topic_name = "projects/my_project_id/topics/my_topic"
  message = "Hello World"
  ordering_key = "key1"
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_publisher_client = mock.create_autospec(
      pubsub_v1.PublisherClient, instance=True
  )
  mock_get_publisher_client.return_value = mock_publisher_client

  mock_future = mock.create_autospec(future.Future, instance=True)
  mock_future.result.return_value = "message_id"
  mock_publisher_client.publish.return_value = mock_future

  result = message_tool.publish_message(
      topic_name,
      message,
      mock_credentials,
      tool_settings,
      ordering_key=ordering_key,
  )

  assert result["message_id"] == "message_id"
  mock_get_publisher_client.assert_called_once()
  _, kwargs = mock_get_publisher_client.call_args
  assert kwargs["publisher_options"].enable_message_ordering is True

  mock_publisher_client.publish.assert_called_once()

  # Verify ordering_key was passed
  _, kwargs = mock_publisher_client.publish.call_args
  assert kwargs["ordering_key"] == ordering_key


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_v1.PublisherClient, "publish", autospec=True)
@mock.patch.object(pubsub_client_lib, "get_publisher_client", autospec=True)
def test_publish_message_with_attributes(
    mock_get_publisher_client, mock_publish
):
  """Test publish_message tool invocation with attributes."""
  topic_name = "projects/my_project_id/topics/my_topic"
  message = "Hello World"
  attributes = {"key1": "value1", "key2": "value2"}
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_publisher_client = mock.create_autospec(
      pubsub_v1.PublisherClient, instance=True
  )
  mock_get_publisher_client.return_value = mock_publisher_client

  mock_future = mock.create_autospec(future.Future, instance=True)
  mock_future.result.return_value = "message_id"
  mock_publisher_client.publish.return_value = mock_future

  result = message_tool.publish_message(
      topic_name,
      message,
      mock_credentials,
      tool_settings,
      attributes=attributes,
  )

  assert result["message_id"] == "message_id"
  mock_get_publisher_client.assert_called_once()
  mock_publisher_client.publish.assert_called_once()

  # Verify attributes were passed
  _, kwargs = mock_publisher_client.publish.call_args
  assert kwargs["key1"] == "value1"
  assert kwargs["key2"] == "value2"


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_v1.PublisherClient, "publish", autospec=True)
@mock.patch.object(pubsub_client_lib, "get_publisher_client", autospec=True)
def test_publish_message_exception(mock_get_publisher_client, mock_publish):
  """Test publish_message tool invocation when exception occurs."""
  topic_name = "projects/my_project_id/topics/my_topic"
  message = "Hello World"
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_publisher_client = mock.create_autospec(
      pubsub_v1.PublisherClient, instance=True
  )
  mock_get_publisher_client.return_value = mock_publisher_client

  # Simulate an exception during publish
  mock_publisher_client.publish.side_effect = Exception("Publish failed")

  result = message_tool.publish_message(
      topic_name,
      message,
      mock_credentials,
      tool_settings,
  )

  assert result["status"] == "ERROR"
  assert "Publish failed" in result["error_details"]
  mock_get_publisher_client.assert_called_once()
  mock_publisher_client.publish.assert_called_once()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_client_lib, "get_subscriber_client", autospec=True)
def test_pull_messages(mock_get_subscriber_client):
  """Test pull_messages tool invocation."""
  subscription_name = "projects/my_project_id/subscriptions/my_sub"
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_subscriber_client = mock.create_autospec(
      pubsub_v1.SubscriberClient, instance=True
  )
  mock_get_subscriber_client.return_value = mock_subscriber_client

  mock_response = mock.create_autospec(types.PullResponse, instance=True)
  mock_message = mock.MagicMock()
  mock_message.message.message_id = "123"
  mock_message.message.data = b"Hello"
  mock_message.message.attributes = {"key": "value"}
  mock_message.message.ordering_key = "ABC"
  mock_publish_time = mock.MagicMock()
  mock_publish_time.rfc3339.return_value = "2023-01-01T00:00:00Z"
  mock_message.message.publish_time = mock_publish_time
  mock_message.ack_id = "ack_123"
  mock_response.received_messages = [mock_message]
  mock_subscriber_client.pull.return_value = mock_response

  result = message_tool.pull_messages(
      subscription_name, mock_credentials, tool_settings
  )

  expected_message = {
      "message_id": "123",
      "data": "Hello",
      "attributes": {"key": "value"},
      "ordering_key": "ABC",
      "publish_time": "2023-01-01T00:00:00Z",
      "ack_id": "ack_123",
  }
  assert result["messages"] == [expected_message]

  mock_get_subscriber_client.assert_called_once()
  mock_subscriber_client.pull.assert_called_once_with(
      subscription=subscription_name, max_messages=1
  )
  mock_subscriber_client.acknowledge.assert_not_called()


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_client_lib, "get_subscriber_client", autospec=True)
def test_pull_messages_auto_ack(mock_get_subscriber_client):
  """Test pull_messages tool invocation with auto_ack."""
  subscription_name = "projects/my_project_id/subscriptions/my_sub"
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_subscriber_client = mock.create_autospec(
      pubsub_v1.SubscriberClient, instance=True
  )
  mock_get_subscriber_client.return_value = mock_subscriber_client

  mock_response = mock.create_autospec(types.PullResponse, instance=True)
  mock_message = mock.MagicMock()
  mock_message.message.message_id = "123"
  mock_message.message.data = b"Hello"
  mock_message.message.attributes = {}
  mock_publish_time = mock.MagicMock()
  mock_publish_time.rfc3339.return_value = "2023-01-01T00:00:00Z"
  mock_message.message.publish_time = mock_publish_time
  mock_message.ack_id = "ack_123"
  mock_response.received_messages = [mock_message]
  mock_subscriber_client.pull.return_value = mock_response

  result = message_tool.pull_messages(
      subscription_name,
      mock_credentials,
      tool_settings,
      max_messages=5,
      auto_ack=True,
  )

  assert len(result["messages"]) == 1
  mock_get_subscriber_client.assert_called_once()
  mock_subscriber_client.pull.assert_called_once_with(
      subscription=subscription_name, max_messages=5
  )
  mock_subscriber_client.acknowledge.assert_called_once_with(
      subscription=subscription_name, ack_ids=["ack_123"]
  )


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_client_lib, "get_subscriber_client", autospec=True)
def test_pull_messages_exception(mock_get_subscriber_client):
  """Test pull_messages tool invocation when exception occurs."""
  subscription_name = "projects/my_project_id/subscriptions/my_sub"
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_subscriber_client = mock.create_autospec(
      pubsub_v1.SubscriberClient, instance=True
  )
  mock_get_subscriber_client.return_value = mock_subscriber_client

  mock_subscriber_client.pull.side_effect = Exception("Pull failed")

  result = message_tool.pull_messages(
      subscription_name, mock_credentials, tool_settings
  )

  assert result["status"] == "ERROR"
  assert "Pull failed" in result["error_details"]


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_client_lib, "get_subscriber_client", autospec=True)
def test_acknowledge_messages(mock_get_subscriber_client):
  """Test acknowledge_messages tool invocation."""
  subscription_name = "projects/my_project_id/subscriptions/my_sub"
  ack_ids = ["ack1", "ack2"]
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_subscriber_client = mock.create_autospec(
      pubsub_v1.SubscriberClient, instance=True
  )
  mock_get_subscriber_client.return_value = mock_subscriber_client

  result = message_tool.acknowledge_messages(
      subscription_name, ack_ids, mock_credentials, tool_settings
  )

  assert result["status"] == "SUCCESS"
  mock_get_subscriber_client.assert_called_once()
  mock_subscriber_client.acknowledge.assert_called_once_with(
      subscription=subscription_name, ack_ids=ack_ids
  )


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(pubsub_client_lib, "get_subscriber_client", autospec=True)
def test_acknowledge_messages_exception(mock_get_subscriber_client):
  """Test acknowledge_messages tool invocation when exception occurs."""
  subscription_name = "projects/my_project_id/subscriptions/my_sub"
  ack_ids = ["ack1"]
  mock_credentials = mock.create_autospec(Credentials, instance=True)
  tool_settings = PubSubToolConfig(project_id="my_project_id")

  mock_subscriber_client = mock.create_autospec(
      pubsub_v1.SubscriberClient, instance=True
  )
  mock_get_subscriber_client.return_value = mock_subscriber_client

  mock_subscriber_client.acknowledge.side_effect = Exception("Ack failed")

  result = message_tool.acknowledge_messages(
      subscription_name, ack_ids, mock_credentials, tool_settings
  )

  assert result["status"] == "ERROR"
  assert "Ack failed" in result["error_details"]
