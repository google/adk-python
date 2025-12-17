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

import base64
from typing import Optional

from google.auth.credentials import Credentials
from google.cloud import pubsub_v1

from . import client
from .config import PubSubToolConfig


def publish_message(
    topic_name: str,
    message: str,
    credentials: Credentials,
    settings: PubSubToolConfig,
    attributes: Optional[dict[str, str]] = None,
    ordering_key: str = "",
) -> dict:
  """Publish a message to a Pub/Sub topic.

  Args:
      topic_name (str): The Pub/Sub topic name (e.g.
        projects/my-project/topics/my-topic).
      message (str): The message content to publish.
      credentials (Credentials): The credentials to use for the request.
      settings (PubSubToolConfig): The Pub/Sub tool settings.
      attributes (Optional[dict[str, str]]): Attributes to attach to the message.
      ordering_key (str): Ordering key for the message.

  Returns:
      dict: Dictionary with the message_id of the published message.
  """
  try:
    publisher_options = pubsub_v1.types.PublisherOptions(
        enable_message_ordering=bool(ordering_key)
    )
    publisher_client = client.get_publisher_client(
        credentials=credentials,
        user_agent=[settings.project_id, "publish_message"],
        publisher_options=publisher_options,
    )

    message_bytes = message.encode("utf-8")
    future = publisher_client.publish(
        topic_name,
        data=message_bytes,
        ordering_key=ordering_key,
        **(attributes or {}),
    )

    return {"message_id": future.result()}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": (
            f"Failed to publish message to topic '{topic_name}': {repr(ex)}"
        ),
    }


def _decode_message_data(data: bytes) -> str:
  """Decodes message data, trying UTF-8 and falling back to base64."""
  try:
    return data.decode("utf-8")
  except UnicodeDecodeError:
    # If UTF-8 decoding fails, encode as base64 string
    return base64.b64encode(data).decode("ascii")


def pull_messages(
    subscription_name: str,
    credentials: Credentials,
    settings: PubSubToolConfig,
    *,
    max_messages: int = 1,
    auto_ack: bool = False,
) -> dict:
  """Pull messages from a Pub/Sub subscription.

  Args:
      subscription_name (str): The Pub/Sub subscription name (e.g.
        projects/my-project/subscriptions/my-sub).
      credentials (Credentials): The credentials to use for the request.
      settings (PubSubToolConfig): The Pub/Sub tool settings.
      max_messages (int): The maximum number of messages to pull. Defaults to 1.
      auto_ack (bool): Whether to automatically acknowledge the messages.
        Defaults to False.

  Returns:
      dict: Dictionary with the list of pulled messages.
  """
  try:
    subscriber_client = client.get_subscriber_client(
        credentials=credentials,
        user_agent=[settings.project_id, "pull_messages"],
    )

    response = subscriber_client.pull(
        subscription=subscription_name,
        max_messages=max_messages,
    )

    messages = []
    ack_ids = []
    for received_message in response.received_messages:
      message_data = _decode_message_data(received_message.message.data)
      messages.append({
          "message_id": received_message.message.message_id,
          "data": message_data,
          "attributes": dict(received_message.message.attributes),
          "ordering_key": received_message.message.ordering_key,
          "publish_time": received_message.message.publish_time.rfc3339(),
          "ack_id": received_message.ack_id,
      })
      ack_ids.append(received_message.ack_id)

    if auto_ack and ack_ids:
      subscriber_client.acknowledge(
          subscription=subscription_name,
          ack_ids=ack_ids,
      )

    return {"messages": messages}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": (
            f"Failed to pull messages from subscription '{subscription_name}':"
            f" {repr(ex)}"
        ),
    }


def acknowledge_messages(
    subscription_name: str,
    ack_ids: list[str],
    credentials: Credentials,
    settings: PubSubToolConfig,
) -> dict:
  """Acknowledge messages on a Pub/Sub subscription.

  Args:
      subscription_name (str): The Pub/Sub subscription name (e.g.
        projects/my-project/subscriptions/my-sub).
      ack_ids (list[str]): List of acknowledgment IDs to acknowledge.
      credentials (Credentials): The credentials to use for the request.
      settings (PubSubToolConfig): The Pub/Sub tool settings.

  Returns:
      dict: Status of the operation.
  """
  try:
    subscriber_client = client.get_subscriber_client(
        credentials=credentials,
        user_agent=[settings.project_id, "acknowledge_messages"],
    )

    subscriber_client.acknowledge(
        subscription=subscription_name,
        ack_ids=ack_ids,
    )

    return {"status": "SUCCESS"}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": (
            "Failed to acknowledge messages on subscription"
            f" '{subscription_name}': {repr(ex)}"
        ),
    }
