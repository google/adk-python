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

"""Remote integration tests for the /apps/trigger_echo_agent/trigger/pubsub endpoint.

Tests cover:

    2. Full pipeline tests — publish to a Pub/Sub topic with a push
       subscription pointing at the Cloud Run service, then verify via
       Cloud Logging that the requests reached the service.

Prerequisites:
    - GCP project with Pub/Sub, Cloud Run, and Cloud Logging APIs enabled.
    - ``gcloud`` CLI authenticated.
    - Terraform >= 1.5 installed.

Run:
    GCP_PROJECT_ID=my-project pytest tests/remote/test_trigger_pubsub.py -v -s
"""

from __future__ import annotations

import base64
import datetime
import json
import time

from google.cloud import logging as cloud_logging
from google.cloud import pubsub_v1
import pytest
import requests

# ---------------------------------------------------------------------------
# Full Pub/Sub pipeline tests
# ---------------------------------------------------------------------------


class TestPubSubPipeline:
  """Test the full Pub/Sub → push subscription → Cloud Run pipeline.

  Verification is done by checking Cloud Run request logs for successful
  POST requests to /apps/trigger_echo_agent/trigger/pubsub. We look for
  httpRequest entries with
  status 200 that arrived after we published.
  """

  @pytest.fixture(autouse=True)
  def _setup(self, pubsub_topic, project_id, cloud_run_url):
    self.publisher = pubsub_v1.PublisherClient()
    self.topic_path = pubsub_topic
    self.project_id = project_id
    self.cloud_run_url = cloud_run_url
    self.logging_client = cloud_logging.Client(project=project_id)

  def _publish_and_wait(
      self,
      data: str,
      attributes: dict[str, str] | None = None,
      wait_seconds: int = 30,
  ) -> None:
    """Publish a message and wait for it to be delivered."""
    future = self.publisher.publish(
        self.topic_path,
        data.encode("utf-8"),
        **(attributes or {}),
    )
    future.result(timeout=30)
    time.sleep(wait_seconds)

  def _count_successful_requests(
      self,
      path: str,
      since: datetime.datetime,
  ) -> int:
    """Count successful HTTP requests to the given path in Cloud Run logs."""
    timestamp = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    filter_str = (
        'resource.type="cloud_run_revision" '
        f'httpRequest.requestUrl:"{path}" '
        "httpRequest.status=200 "
        f'timestamp>="{timestamp}"'
    )
    entries = list(
        self.logging_client.list_entries(
            filter_=filter_str,
            page_size=200,
        )
    )
    return len(entries)

  def test_publish_text_message(self):
    """Publish a text message and verify it reaches Cloud Run."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    self._publish_and_wait("hello-pipeline-test")
    count = self._count_successful_requests(
        "/apps/trigger_echo_agent/trigger/pubsub", start_time
    )
    assert count >= 1, (
        "Expected at least 1 successful request to"
        f" /apps/trigger_echo_agent/trigger/pubsub, found {count}"
    )

  def test_publish_json_message(self):
    """Publish a JSON message and verify processing."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    data = json.dumps({"type": "json", "value": 42})
    self._publish_and_wait(data)
    count = self._count_successful_requests(
        "/apps/trigger_echo_agent/trigger/pubsub", start_time
    )
    assert count >= 1, (
        "Expected at least 1 successful request to"
        f" /apps/trigger_echo_agent/trigger/pubsub, found {count}"
    )

  def test_publish_with_attributes(self):
    """Publish a message with attributes and verify processing."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    self._publish_and_wait(
        "attr-pipeline-test",
        attributes={"test_attr": "pytest"},
    )
    count = self._count_successful_requests(
        "/apps/trigger_echo_agent/trigger/pubsub", start_time
    )
    assert count >= 1, (
        "Expected at least 1 successful request to"
        f" /apps/trigger_echo_agent/trigger/pubsub, found {count}"
    )

  def test_high_volume(self):
    """Publish 20 messages and verify they are processed."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    n = 20
    futures = []
    for i in range(n):
      future = self.publisher.publish(
          self.topic_path,
          f"volume-test-{i}".encode("utf-8"),
      )
      futures.append(future)

    for f in futures:
      f.result(timeout=30)

    # Wait for push delivery.
    time.sleep(15)

    count = self._count_successful_requests(
        "/apps/trigger_echo_agent/trigger/pubsub", start_time
    )
    # Allow some tolerance — at least 80% should arrive.
    assert (
        count >= n * 0.8
    ), f"Expected at least {int(n * 0.8)} successful requests, found {count}"
