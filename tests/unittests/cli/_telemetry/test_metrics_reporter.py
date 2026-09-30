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

"""Unit tests for ADK CLI telemetry metrics reporter."""

import json
import os
import socket
import tempfile
import time
import unittest
from unittest import mock
import urllib.error

# Import the module directly from the namespace package location
from google.adk.cli._telemetry import _metrics_reporter as metrics_reporter

# Create a temporary directory for tests to avoid writing to user home directory
_TEMP_DIR = tempfile.mkdtemp()
_QUEUE_FILE = os.path.join(_TEMP_DIR, "telemetry_queue.jsonl")
_LOCK_FILE = os.path.join(_TEMP_DIR, "clearcut_lock")


class CliMetricsReporterTest(unittest.TestCase):
  """Tests for the background telemetry metrics reporter daemon."""

  def setUp(self):
    super().setUp()

    # Patch paths per test to prevent leakage and global pollution across
    # modules.
    self.queue_patcher = mock.patch.object(
        metrics_reporter._constants, "QUEUE_FILE", _QUEUE_FILE
    )
    self.lock_patcher = mock.patch.object(
        metrics_reporter._constants, "LOCK_FILE", _LOCK_FILE
    )
    self.ep_patcher = mock.patch.object(
        metrics_reporter, "_CLEARCUT_ENDPOINT_PROD", "http://localhost/mock"
    )
    self.queue_patcher.start()
    self.lock_patcher.start()
    self.ep_patcher.start()

    os.makedirs(_TEMP_DIR, exist_ok=True)
    if os.path.exists(_QUEUE_FILE):
      os.remove(_QUEUE_FILE)
    if os.path.exists(_LOCK_FILE):
      os.remove(_LOCK_FILE)

  def tearDown(self):
    self.queue_patcher.stop()
    self.lock_patcher.stop()
    self.ep_patcher.stop()
    if os.path.exists(_QUEUE_FILE):
      os.remove(_QUEUE_FILE)
    if os.path.exists(_LOCK_FILE):
      os.remove(_LOCK_FILE)
    try:
      os.rmdir(_TEMP_DIR)
    except OSError:
      pass
    super().tearDown()

  def test_reporter_rate_limiting_429(self):
    """Verify that HTTP 429 triggers the local rate limit lock file."""
    # Setup mock queue data
    os.makedirs(os.path.dirname(_QUEUE_FILE), exist_ok=True)
    event_data = {"event_time_ms": 1234, "source_extension_json": "{}"}
    with open(_QUEUE_FILE, "w") as f:
      f.write(json.dumps(event_data) + "\n")

    # Setup mock HTTPError for 429
    mock_response = mock.MagicMock()
    mock_response.code = 429
    mock_response.read.return_value = b"\x08\xc0\xd4\x03"

    http_error = urllib.error.HTTPError(
        url="http://clearcut",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=mock_response,
    )

    with mock.patch.object(
        metrics_reporter.urllib.request, "urlopen", side_effect=http_error
    ):
      # Run reporter
      metrics_reporter.report_metrics()

    # Queue should be deleted and rate limit lock should be set ~60s.
    self.assertFalse(os.path.exists(_QUEUE_FILE))
    self.assertTrue(os.path.exists(_LOCK_FILE))
    with open(_LOCK_FILE, "r") as lf:
      lock_time = float(lf.read().strip())
      self.assertGreater(lock_time, time.time() + 50)

  def test_reporter_retry_on_timeout(self):
    """Verify timeout errors trigger retries and eventual abandonment."""
    os.makedirs(os.path.dirname(_QUEUE_FILE), exist_ok=True)
    with open(_QUEUE_FILE, "w") as f:
      f.write(
          json.dumps({"event_time_ms": 1234, "source_extension_json": "{}"})
          + "\n"
      )

    url_error = urllib.error.URLError(socket.timeout("timed out"))

    with (
        mock.patch.object(
            metrics_reporter.urllib.request, "urlopen", side_effect=url_error
        ) as mock_urlopen,
        mock.patch.object(metrics_reporter.time, "sleep") as mock_sleep,
    ):
      metrics_reporter.report_metrics()
      # Since lock wait times is (1, 2), we make 1 initial attempt +
      # 2 retries = 3 attempts total.
      self.assertEqual(mock_urlopen.call_count, 3)
      self.assertEqual(mock_sleep.call_count, 2)
      mock_sleep.assert_has_calls([mock.call(1), mock.call(2)])

  def test_reporter_no_retry_on_offline(self):
    """Verify that offline errors (e.g. DNS) fail-fast without retrying."""
    os.makedirs(os.path.dirname(_QUEUE_FILE), exist_ok=True)
    with open(_QUEUE_FILE, "w") as f:
      f.write(
          json.dumps({"event_time_ms": 1234, "source_extension_json": "{}"})
          + "\n"
      )

    # socket.gaierror is raised when name resolution fails (e.g. offline)
    url_error = urllib.error.URLError(
        socket.gaierror(-2, "Name or service not known")
    )

    with (
        mock.patch.object(
            metrics_reporter.urllib.request, "urlopen", side_effect=url_error
        ) as mock_urlopen,
        mock.patch.object(metrics_reporter.time, "sleep") as mock_sleep,
    ):
      metrics_reporter.report_metrics()
      # Should only attempt once and fail immediately
      self.assertEqual(mock_urlopen.call_count, 1)
      self.assertEqual(mock_sleep.call_count, 0)


if __name__ == "__main__":
  unittest.main()
