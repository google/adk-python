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

"""Remote integration tests for the /apps/trigger_echo_agent/trigger/eventarc endpoint.

Tests cover:

       /apps/trigger_echo_agent/trigger/eventarc on Cloud Run.
    2. Full pipeline tests — publish to the Eventarc source topic and
       verify the request reached Cloud Run via Cloud Logging.

Prerequisites:
    - GCP project with Eventarc, Pub/Sub, Cloud Run, and Cloud Logging
      APIs enabled.
    - ``gcloud`` CLI authenticated.
    - Terraform >= 1.5 installed.

Run:
    GCP_PROJECT_ID=my-project pytest tests/remote/test_trigger_eventarc.py -v -s
"""

from __future__ import annotations

import datetime
import json
import time
import uuid

from google.cloud import logging as cloud_logging
from google.cloud import pubsub_v1
import pytest
import requests

# ---------------------------------------------------------------------------
# Full Eventarc pipeline tests
# ---------------------------------------------------------------------------


class TestEventarcPipeline:
  """Test the full Pub/Sub → Eventarc → Cloud Run pipeline.

  Verification is done by checking Cloud Run request logs for successful
  POST requests to /apps/trigger_echo_agent/trigger/eventarc.
  """

  @pytest.fixture(autouse=True)
  def _setup(self, eventarc_topic, project_id):
    self.publisher = pubsub_v1.PublisherClient()
    self.topic_path = eventarc_topic
    self.project_id = project_id
    self.logging_client = cloud_logging.Client(project=project_id)

  def _publish_event(self, data: str, wait_seconds: int = 45) -> None:
    """Publish a message to the Eventarc source topic and wait."""
    future = self.publisher.publish(
        self.topic_path,
        data.encode("utf-8"),
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
            page_size=50,
        )
    )
    return len(entries)

  def test_eventarc_pubsub_trigger(self):
    """Publish to Eventarc source topic and verify Cloud Run processes it."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    self._publish_event(json.dumps({"test": "eventarc-pipeline"}))
    count = self._count_successful_requests(
        "/apps/trigger_echo_agent/trigger/eventarc", start_time
    )
    assert count >= 1, (
        "Expected at least 1 successful request to"
        f" /apps/trigger_echo_agent/trigger/eventarc, found {count}"
    )

  def test_eventarc_multiple_events(self):
    """Publish multiple events and verify they are processed."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    for i in range(3):
      future = self.publisher.publish(
          self.topic_path,
          json.dumps({"seq": i}).encode("utf-8"),
      )
      future.result(timeout=30)

    # Eventarc delivery + log ingestion can be slow; wait longer.
    time.sleep(90)

    count = self._count_successful_requests(
        "/apps/trigger_echo_agent/trigger/eventarc", start_time
    )
    assert count >= 1, (
        "Expected at least 1 successful request to"
        f" /apps/trigger_echo_agent/trigger/eventarc, found {count}"
    )
