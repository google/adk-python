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

"""Collection utility for capturing and queueing ADK CLI telemetry."""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from typing import List
from typing import Optional
import uuid

import click
from google.adk.cli._telemetry import _constants
import google.adk.version

# 1 hour of quiet time marks the end of a logical work session
_SESSION_INACTIVITY_TIMEOUT_SECONDS = 3600

# Constants protecting telemetry storage and preventing client/endpoint abuse.
# Prevents haywire scripts or malicious inputs from flooding log files
# or sending excessively bloated payloads to the public Clearcut server.
_MAX_QUEUE_SIZE_BYTES = 1048576  # 1 MB queue size boundary.
_MAX_STRING_LENGTH = 64  # Max length for command/flag names.
_MAX_EXCEPTION_LENGTH = 128  # Max length for exception type strings.
_MAX_FLAGS_COUNT = 50  # Max number of options recorded.

logger = logging.getLogger("google_adk." + __name__)


def _prune_expired_sessions() -> None:
  """Prunes session files modified more than 1 hour ago."""
  if not os.path.exists(_constants.TELEMETRY_SESSIONS_DIR):
    return
  current_time = time.time()
  try:
    for filename in os.listdir(_constants.TELEMETRY_SESSIONS_DIR):
      file_path = os.path.join(_constants.TELEMETRY_SESSIONS_DIR, filename)
      try:
        if filename.endswith(".json"):
          try:
            with open(file_path, "r", encoding="utf-8") as f:
              info = json.load(f)
              last_activity = info.get("last_activity", 0)
          except (OSError, json.JSONDecodeError, KeyError, TypeError):
            # If the JSON file is unreadable/corrupted, prune it.
            last_activity = 0
          if (
              current_time - last_activity
          ) > _SESSION_INACTIVITY_TIMEOUT_SECONDS:
            os.remove(file_path)
        elif filename.endswith(".tmp"):
          os.remove(file_path)
      except OSError:
        pass
  except OSError:
    pass


def _load_session_state() -> tuple[str, int]:
  """Retrieves the session state, resetting it if idle for over an hour."""
  session_file = os.path.join(
      _constants.TELEMETRY_SESSIONS_DIR, f"{os.getppid()}.json"
  )
  try:
    with open(session_file, "r", encoding="utf-8") as f:
      info = json.load(f)
      if isinstance(info, dict):
        last_activity = info.get("last_activity", 0)
        if (time.time() - last_activity) < _SESSION_INACTIVITY_TIMEOUT_SECONDS:
          session_id = info.get("session_id")
          if not session_id:
            return str(uuid.uuid4()), 0
          return session_id, info.get("sequence_number", 0)
  except (OSError, json.JSONDecodeError, KeyError, TypeError):
    pass
  return str(uuid.uuid4()), 0


def _write_session_state(session_id: str, sequence_number: int) -> None:
  """Saves the current session metadata back to local disk storage."""
  _prune_expired_sessions()
  session_file = os.path.join(
      _constants.TELEMETRY_SESSIONS_DIR, f"{os.getppid()}.json"
  )
  temp_file = None
  try:
    info = {
        "session_id": session_id,
        "sequence_number": sequence_number,
        "last_activity": time.time(),
    }

    temp_file = f"{session_file}.{os.getpid()}.tmp"
    os.makedirs(os.path.dirname(session_file), exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
      json.dump(info, f)
    os.replace(temp_file, session_file)
    temp_file = None
  except OSError:
    pass
  finally:
    if temp_file is not None:
      try:
        os.remove(temp_file)
      except OSError:
        pass


class MetricsCollector:
  """Collector for capturing and queueing ADK CLI telemetry."""

  @staticmethod
  def _is_rate_limited() -> bool:
    """Check if Clearcut backoff was requested."""
    if not os.path.exists(_constants.LOCK_FILE):
      return False
    try:
      with open(_constants.LOCK_FILE, "r") as f:
        lock_time = float(f.read().strip())
        # If current time is less than the lock endpoint, we are rate limited.
        return time.time() < lock_time
    except Exception:  # pylint: disable=broad-exception-caught
      # Fail closed defensively to protect the server on file read errs.
      return True

  def __init__(self) -> None:
    self._lock = threading.Lock()
    # Load session metadata matching parent terminal process
    # We generate an ephemeral session ID and sequence number to group commands
    # executed in the same terminal session for insightful metrics analytics.
    self._session_id, self._sequence_number = _load_session_state()

    self._environment = {
        "os_type": platform.system().lower(),
        "language": "python",
        "language_version": platform.python_version(),
        "adk_version": google.adk.version.__version__,
        "is_tty": sys.stdout.isatty() if sys.stdout else False,
    }
    logger.debug(
        "Initialized ADK metrics collector with session %s (seq %d)",
        self._session_id,
        self._sequence_number,
    )
    atexit.register(self.shutdown)

  @staticmethod
  def _gather_flags_from_click() -> Optional[List[str]]:
    """Gathers used flags and positional argument names from Click context."""
    ctx = click.get_current_context(silent=True)
    if not ctx:
      return None

    used_params: list[str] = []
    for param in ctx.command.params:
      if len(used_params) >= _MAX_FLAGS_COUNT:
        break
      if (
          ctx.get_parameter_source(param.name)
          == click.core.ParameterSource.COMMANDLINE
      ):
        if isinstance(param, click.Option):
          if param.opts:
            used_params.append(param.opts[0][:_MAX_STRING_LENGTH])
        elif isinstance(param, click.Argument):
          # Log positional variable name (e.g. '<agent_path>')
          # instead of PII value.
          used_params.append(f"<{param.name[:_MAX_STRING_LENGTH]}>")

    return used_params if used_params else None

  def record_command_run(
      self,
      command: str,
      subcommand: str = "",
      exit_code: int = 0,
      duration_ms: int = 0,
      exception_type: str = "",
      express_mode_action: str = "",
  ) -> None:
    """Records a command execution and safely appends to local disk queue."""
    with self._lock:
      self._sequence_number += 1
      # Save the updated sequence number back to the master file
      _write_session_state(self._session_id, self._sequence_number)
    # Enforce string length constraints
    command = command[:_MAX_STRING_LENGTH] if command else ""
    subcommand = subcommand[:_MAX_STRING_LENGTH] if subcommand else ""

    command_run = {
        "command": command,
        "subcommand": subcommand,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
    }
    flags = self._gather_flags_from_click()
    if flags:
      command_run["flags"] = flags
    if exception_type:
      # Enforce string length limit on exception type name
      command_run["exception_type"] = exception_type[:_MAX_EXCEPTION_LENGTH]
    if express_mode_action:
      command_run["express_mode_action"] = express_mode_action[
          :_MAX_STRING_LENGTH
      ]

    source_extension = {
        "client_session_id": self._session_id,
        "sequence_number": self._sequence_number,
        "environment": self._environment,
        "command_run": command_run,
    }

    log_event = {
        "event_time_ms": int(time.time() * 1000),
        "source_extension_json": json.dumps(
            source_extension, separators=(",", ":")
        ),
    }

    # Instantly append to local json queue safely
    try:
      # Enforce queue file limit to protect disk bounds
      if (
          os.path.exists(_constants.QUEUE_FILE)
          and os.path.getsize(_constants.QUEUE_FILE) > _MAX_QUEUE_SIZE_BYTES
      ):
        return
      os.makedirs(os.path.dirname(_constants.QUEUE_FILE), exist_ok=True)
      with open(_constants.QUEUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_event) + "\n")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.debug("Failed to record metric: %s", e)

  def shutdown(self) -> None:
    """Checks rate-limit and spins off metrics_reporter to flush queue."""
    if self._is_rate_limited():
      return

    if (
        not os.path.exists(_constants.QUEUE_FILE)
        or os.path.getsize(_constants.QUEUE_FILE) == 0
    ):
      return

    reporter_path = os.path.join(
        os.path.dirname(__file__), "_metrics_reporter.py"
    )
    env = os.environ.copy()
    env["ADK_VERSION"] = google.adk.version.__version__

    try:
      # Detach the subprocess to run in background
      subprocess.Popen(
          [sys.executable, reporter_path],
          env=env,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
          stdin=subprocess.DEVNULL,
          start_new_session=True,  # Detach from terminal
      )
      logger.debug("Metrics reporter subprocess launched.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.debug("Failed to launch metrics reporter: %s", e)
