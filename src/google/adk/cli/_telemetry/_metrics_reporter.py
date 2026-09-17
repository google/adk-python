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

"""Background reporter utility that flushes local metrics to Clearcut."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
import uuid

try:
  from google.adk.cli._telemetry import _constants
except ImportError:
  import _constants  # type: ignore[no-redef]

# Exponential backoff retry intervals (in seconds) for connection retries.
_RETRY_BACKOFF_WAIT_TIMES = (1, 2)
# Max network connection and read timeout (in seconds) for HTTP requests.
_TIMEOUT_IN_SEC = 5


# Clearcut registration details for Google ADK logs.
_LOG_SOURCE_INT = 3007
_CLIENT_TYPE = "PYTHON"

# Clearcut collection server endpoints.
_CLEARCUT_ENDPOINT_PROD = "https://play.googleapis.com/log"


def _extract_next_request_wait_millis(data: bytes) -> int | None:
  """Safely extracts field 1 (varint) from binary protobuffer response."""
  idx = 0
  while idx < len(data):
    tag = 0
    shift = 0
    while True:
      if idx >= len(data):
        return None
      b = data[idx]
      idx += 1
      tag |= (b & 0x7F) << shift
      if not (b & 0x80):
        break
      shift += 7
    field_num = tag >> 3
    wire_type = tag & 0x07
    if field_num == 1 and wire_type == 0:
      val = 0
      shift = 0
      while True:
        if idx >= len(data):
          return None
        b = data[idx]
        idx += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
          return val
        shift += 7
    if wire_type == 0:
      while idx < len(data):
        b = data[idx]
        idx += 1
        if not (b & 0x80):
          break
    elif wire_type == 1:
      idx += 8
    elif wire_type == 2:
      length = 0
      shift = 0
      while True:
        if idx >= len(data):
          return None
        b = data[idx]
        idx += 1
        length |= (b & 0x7F) << shift
        if not (b & 0x80):
          break
        shift += 7
      idx += length
    elif wire_type == 5:
      idx += 4
    else:
      return None
  return None


def _set_rate_limit_timestamp(wait_ms: int) -> None:
  """Sets a rate limit lock until time.time() + wait_ms."""
  if wait_ms is not None and isinstance(wait_ms, (int, float)) and wait_ms > 0:
    try:
      lock_time = time.time() + (wait_ms / 1000.0)
      os.makedirs(os.path.dirname(_constants.LOCK_FILE), exist_ok=True)
      with open(_constants.LOCK_FILE, "w") as lf:
        lf.write(str(lock_time))
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def _send_request_with_retry(req: urllib.request.Request) -> bool:
  """Sends the request with a retry backoff loop. Returns True on success."""
  retries = 0
  success = False
  while not success and retries <= len(_RETRY_BACKOFF_WAIT_TIMES):
    try:
      res = urllib.request.urlopen(req, timeout=_TIMEOUT_IN_SEC)
      if res.getcode() == 200:
        try:
          res_bytes = res.read()
          wait_ms = _extract_next_request_wait_millis(res_bytes)
          if wait_ms is not None and wait_ms > 0:
            _set_rate_limit_timestamp(wait_ms)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      success = True
    except urllib.error.HTTPError as e:
      # Handle rate limiting (HTTP 429) or other client errors
      if e.code == 429:
        try:
          res_bytes = e.read()
          wait_ms = _extract_next_request_wait_millis(res_bytes)
          if wait_ms is None or wait_ms <= 0:
            # Fallback default: lock for 5 mins (300,000 ms)
            wait_ms = 300000
          _set_rate_limit_timestamp(wait_ms)
        except Exception:  # pylint: disable=broad-exception-caught
          _set_rate_limit_timestamp(300000)
        # Abort retries immediately on rate limiting (429)
        break
      elif 400 <= e.code < 500:
        # Direct client errors are permanent, do not retry
        break
      else:
        # Server side errors (5xx)
        retries += 1
        if retries <= len(_RETRY_BACKOFF_WAIT_TIMES):
          time.sleep(_RETRY_BACKOFF_WAIT_TIMES[retries - 1])
        else:
          break
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
      # If the error is a timeout, we retry.
      # If the error is due to being completely offline
      # (e.g., DNS resolution failure
      # or network unreachable), we fail-fast immediately.
      reason = getattr(e, "reason", e)
      is_timeout = isinstance(
          reason, (socket.timeout, TimeoutError)
      ) or "timed out" in str(reason)
      if is_timeout:
        retries += 1
        if retries <= len(_RETRY_BACKOFF_WAIT_TIMES):
          time.sleep(_RETRY_BACKOFF_WAIT_TIMES[retries - 1])
          continue
      break
    except Exception:  # pylint: disable=broad-exception-caught
      # Under no circumstances should backend errors crash target CLI
      break
  return success


def report_metrics() -> None:
  """Consumes the queue and sends telemetry events to Clearcut."""
  if not os.path.exists(_constants.QUEUE_FILE):
    return

  temp_processing_file = f"{_constants.QUEUE_FILE}.tmp.{uuid.uuid4().hex}"
  try:
    # Atomically seize the queue
    os.rename(_constants.QUEUE_FILE, temp_processing_file)
  except OSError:
    return

  log_events = []
  try:
    with open(temp_processing_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line:
          log_events.append(json.loads(line))
  except (json.JSONDecodeError, UnicodeDecodeError):
    # File is heavily corrupted, clear it out.
    os.remove(temp_processing_file)
    return

  # Delete immediately to prevent ghost retries across different CLI commands
  # if this process crashes mid-execution.
  os.remove(temp_processing_file)

  if not log_events:
    return

  clearcut_request = {
      "log_source": _LOG_SOURCE_INT,
      "client_info": {"client_type": _CLIENT_TYPE},
      "request_time_ms": int(time.time() * 1000),
      "log_event": log_events,
  }

  # Fetch adk_version from environment to form User-Agent
  adk_version = os.environ.get("ADK_VERSION", "1.0")
  user_agent = f"ADK-CLI/{adk_version}"

  endpoint = _CLEARCUT_ENDPOINT_PROD
  # Note: Clearcut endpoints are called directly by public OSS clients and do
  # not use/require mTLS (compliance check bypass: play.mtls.googleapis.com).
  data = json.dumps(clearcut_request, separators=(",", ":")).encode("utf-8")
  headers = {"User-Agent": user_agent, "Content-Type": "application/json"}

  req = urllib.request.Request(
      endpoint, data=data, headers=headers, method="POST"
  )

  _send_request_with_retry(req)


if __name__ == "__main__":
  try:
    report_metrics()
  except Exception:  # pylint: disable=broad-exception-caught
    # Failsafe: Never surface background errors to the CLI user
    pass
