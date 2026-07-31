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

"""Unit tests for ADK CLI usage telemetry collection."""

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from google.adk.cli._telemetry import _metrics_collector as metrics

# Create a temporary directory for tests to avoid writing to user home.
_TEMP_DIR = tempfile.mkdtemp()
_QUEUE_FILE = os.path.join(_TEMP_DIR, "telemetry_queue.jsonl")
_LOCK_FILE = os.path.join(_TEMP_DIR, "clearcut_lock")
_CONFIG_FILE = os.path.join(_TEMP_DIR, "config.json")
_TELEMETRY_SESSIONS_DIR = os.path.join(_TEMP_DIR, "telemetry_sessions")


class CliMetricsTest(unittest.TestCase):
  """Tests for ADK CLI usage metrics collection."""

  def setUp(self):
    super().setUp()

    # Patch paths per test to prevent leakage across modules
    self.queue_patcher = mock.patch.object(
        metrics._constants, "QUEUE_FILE", _QUEUE_FILE
    )
    self.lock_patcher = mock.patch.object(
        metrics._constants, "LOCK_FILE", _LOCK_FILE
    )
    self.sessions_patcher = mock.patch.object(
        metrics._constants, "TELEMETRY_SESSIONS_DIR", _TELEMETRY_SESSIONS_DIR
    )
    self.queue_patcher.start()
    self.lock_patcher.start()
    self.sessions_patcher.start()

    os.makedirs(_TEMP_DIR, exist_ok=True)
    if os.path.exists(_QUEUE_FILE):
      os.remove(_QUEUE_FILE)
    if os.path.exists(_LOCK_FILE):
      os.remove(_LOCK_FILE)
    if os.path.exists(_CONFIG_FILE):
      os.remove(_CONFIG_FILE)
    if os.path.exists(_TELEMETRY_SESSIONS_DIR):
      shutil.rmtree(_TELEMETRY_SESSIONS_DIR)

  def tearDown(self):
    self.queue_patcher.stop()
    self.lock_patcher.stop()
    self.sessions_patcher.stop()
    if os.path.exists(_QUEUE_FILE):
      os.remove(_QUEUE_FILE)
    if os.path.exists(_LOCK_FILE):
      os.remove(_LOCK_FILE)
    if os.path.exists(_CONFIG_FILE):
      os.remove(_CONFIG_FILE)
    if os.path.exists(_TELEMETRY_SESSIONS_DIR):
      shutil.rmtree(_TELEMETRY_SESSIONS_DIR)
    try:
      os.rmdir(_TEMP_DIR)
    except OSError:
      pass
    super().tearDown()

  def test_rate_limited_defensive_fail_closed_on_exception(self):
    """Verify that reading exceptions in rate limit log defaults to True."""
    # Create the lock file so exists check succeeds
    with open(_LOCK_FILE, "w") as f:
      f.write("invalid-non-float-lock-time")

    # Trigger ValueError during float casting to verify fail closed.
    # Verify it defaults to True (meaning it fails closed and rate limited).
    self.assertTrue(metrics.MetricsCollector._is_rate_limited())

  def test_record_command_run(self):
    """Verify command execution logs are correctly parsed and queued."""
    collector = metrics.MetricsCollector()

    # Exit the patch block so standard path checks run cleanly.
    with mock.patch.object(
        collector,
        "_gather_flags_from_click",
        return_value=["--debug", "--project", "-v", "--user"],
    ):
      collector.record_command_run(
          command="deploy",
          subcommand="create",
          exit_code=0,
          duration_ms=450,
          exception_type="",
      )

    # Verify it's written in queue file
    self.assertTrue(os.path.exists(_QUEUE_FILE))
    with open(_QUEUE_FILE, "r", encoding="utf-8") as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 1)
      event = json.loads(lines[0])
      self.assertIn("source_extension_json", event)

      source = json.loads(event["source_extension_json"])
      self.assertEqual(source["command_run"]["command"], "deploy")
      self.assertEqual(source["command_run"]["subcommand"], "create")
      self.assertEqual(source["command_run"]["exit_code"], 0)
      self.assertEqual(source["command_run"]["duration_ms"], 450)
      self.assertEqual(
          source["command_run"]["flags"],
          ["--debug", "--project", "-v", "--user"],
      )

  def test_record_command_run_with_click(self):
    """Verify that flags are correctly extracted from Click context."""
    collector = metrics.MetricsCollector()

    # Mock Click context and parameters
    mock_ctx = mock.MagicMock()

    # 1. Option passed on command line
    opt1 = mock.MagicMock(spec=metrics.click.Option)
    opt1.name = "debug"
    opt1.opts = ["--debug"]

    # 2. Option NOT passed on command line (default)
    opt2 = mock.MagicMock(spec=metrics.click.Option)
    opt2.name = "project"
    opt2.opts = ["--project"]

    # 3. Positional argument passed on command line
    arg1 = mock.MagicMock(spec=metrics.click.Argument)
    arg1.name = "agent_path"

    mock_ctx.command.params = [opt1, opt2, arg1]

    # Setup parameter source lookups
    COMMANDLINE = metrics.click.core.ParameterSource.COMMANDLINE
    DEFAULT = metrics.click.core.ParameterSource.DEFAULT
    mock_ctx.get_parameter_source.side_effect = (
        lambda name: COMMANDLINE if name in ["debug", "agent_path"] else DEFAULT
    )

    with mock.patch.object(
        metrics.click, "get_current_context", return_value=mock_ctx
    ):
      collector.record_command_run(
          command="deploy",
          subcommand="create",
          exit_code=0,
          duration_ms=450,
      )

    # Verify it's written in queue file with click flags
    self.assertTrue(os.path.exists(_QUEUE_FILE))
    with open(_QUEUE_FILE, "r", encoding="utf-8") as f:
      lines = f.readlines()
      self.assertEqual(len(lines), 1)
      event = json.loads(lines[0])
      source = json.loads(event["source_extension_json"])
      self.assertEqual(
          source["command_run"]["flags"],
          ["--debug", "<agent_path>"],
      )

  @mock.patch("os.getppid", return_value=12345)
  @mock.patch("time.time", return_value=1000.0)
  def test_session_lifecycle(self, _mock_time, _mock_getppid):
    """Test standard session persistence and sequence matching state."""
    collector = metrics.MetricsCollector()
    initial_session_id = collector._session_id
    self.assertEqual(collector._sequence_number, 0)

    collector.record_command_run(command="deploy", exit_code=0)
    self.assertEqual(collector._sequence_number, 1)

    next_collector = metrics.MetricsCollector()
    self.assertEqual(next_collector._session_id, initial_session_id)
    self.assertEqual(next_collector._sequence_number, 1)

    next_collector.record_command_run(command="run", exit_code=0)
    self.assertEqual(next_collector._sequence_number, 2)

  @mock.patch("os.getppid", return_value=12345)
  def test_session_reset_after_inactivity_timeout(self, _mock_getppid):
    """Test that session ID gets reset if inactivity timer limit is exceeded."""
    with mock.patch("time.time", return_value=1000.0):
      collector = metrics.MetricsCollector()
      first_session = collector._session_id
      collector.record_command_run(command="deploy", exit_code=0)
      self.assertEqual(collector._sequence_number, 1)

    with mock.patch("time.time", return_value=1000.0 + 7200.0):
      next_collector = metrics.MetricsCollector()
      self.assertNotEqual(next_collector._session_id, first_session)
      self.assertEqual(next_collector._sequence_number, 0)

  @mock.patch("time.time", return_value=1000.0)
  def test_session_reset_if_ppid_changes(self, _mock_time):
    """Test that session gets reset if the PPID changes."""
    with mock.patch("os.getppid", return_value=12345):
      collector = metrics.MetricsCollector()
      first_session = collector._session_id
      collector.record_command_run(command="deploy", exit_code=0)

    with mock.patch("os.getppid", return_value=67890):
      next_collector = metrics.MetricsCollector()
      self.assertNotEqual(next_collector._session_id, first_session)
      self.assertEqual(next_collector._sequence_number, 0)

  @mock.patch("os.getppid", return_value=12345)
  def test_session_pruning_removes_old_sessions(self, _mock_getppid):
    """Test that writing the session file prunes aged sessions."""
    os.makedirs(_TELEMETRY_SESSIONS_DIR, exist_ok=True)
    active_file = os.path.join(_TELEMETRY_SESSIONS_DIR, "12345.json")
    expired_file = os.path.join(_TELEMETRY_SESSIONS_DIR, "67890.json")
    expired_tmp_file = os.path.join(_TELEMETRY_SESSIONS_DIR, "67890.json.tmp")

    corrupted_file = os.path.join(_TELEMETRY_SESSIONS_DIR, "corrupted.json")

    with open(active_file, "w", encoding="utf-8") as f:
      json.dump(
          {
              "session_id": "active-session-id",
              "sequence_number": 5,
              "last_activity": 1000.0,
          },
          f,
      )
    os.utime(active_file, (1000.0, 1000.0))

    with open(expired_file, "w", encoding="utf-8") as f:
      json.dump(
          {
              "session_id": "expired-session-id",
              "sequence_number": 10,
              "last_activity": 0.0,
          },
          f,
      )
    os.utime(expired_file, (0.0, 0.0))

    with open(expired_tmp_file, "w", encoding="utf-8") as f:
      f.write("{}")
    os.utime(expired_tmp_file, (0.0, 0.0))

    with open(corrupted_file, "w", encoding="utf-8") as f:
      f.write("invalid json content")

    with mock.patch("time.time", return_value=4000.0):
      # 4000.0 - 1000.0 = 3000 (kept since < 3600)
      # 4000.0 - 0.0 = 4000 (pruned since > 3600)
      collector = metrics.MetricsCollector()
      collector.record_command_run(command="deploy", exit_code=0)

      self.assertTrue(os.path.exists(active_file))
      self.assertFalse(os.path.exists(expired_file))
      self.assertFalse(os.path.exists(expired_tmp_file))
      self.assertFalse(os.path.exists(corrupted_file))

  def test_metrics_collector_independent_instances(self):
    """Verify that multiple instantiations yield separate objects in memory."""
    collector_1 = metrics.MetricsCollector()
    collector_2 = metrics.MetricsCollector()

    self.assertIsNot(collector_1, collector_2)
    self.assertIsNot(collector_1._lock, collector_2._lock)

  @mock.patch("os.getppid", return_value=12345)
  @mock.patch("time.time", return_value=1000.0)
  def test_session_fallback_if_session_id_is_missing(
      self, _mock_time, _mock_getppid
  ):
    """Test that a new UUID is generated if the session file has missing/empty key."""
    os.makedirs(_TELEMETRY_SESSIONS_DIR, exist_ok=True)
    session_file = os.path.join(_TELEMETRY_SESSIONS_DIR, "12345.json")
    with open(session_file, "w", encoding="utf-8") as f:
      json.dump(
          {
              "session_id": "",
              "sequence_number": 5,
              "last_activity": 1000.0,
          },
          f,
      )

    collector = metrics.MetricsCollector()
    self.assertIsNotNone(collector._session_id)
    self.assertNotEqual(collector._session_id, "")
    self.assertEqual(collector._sequence_number, 0)


if __name__ == "__main__":
  unittest.main()
