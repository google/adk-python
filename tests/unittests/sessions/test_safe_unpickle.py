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

"""Tests for _safe_unpickle RestrictedUnpickler."""

from __future__ import annotations

import io
import os
import pickle
import struct
import unittest

from google.adk.events.event_actions import EventActions
from google.adk.sessions.schemas._safe_unpickle import safe_loads


def _make_global_payload(module: str, func: str, *args: str) -> bytes:
  """Craft a pickle stream that calls module.func(*args)."""
  buf = io.BytesIO()
  buf.write(pickle.PROTO + struct.pack("B", 2))
  buf.write(b"c" + f"{module}\n{func}\n".encode())
  buf.write(b"(")
  for arg in args:
    encoded = arg.encode("utf-8")
    buf.write(
        pickle.SHORT_BINUNICODE + struct.pack("<B", len(encoded)) + encoded
    )
  buf.write(b"t")
  buf.write(b"R")
  buf.write(b".")
  return buf.getvalue()


class TestBlockedPayloads(unittest.TestCase):
  """Malicious pickle payloads must be blocked."""

  def test_os_system(self):
    with self.assertRaises(pickle.UnpicklingError):
      safe_loads(_make_global_payload("os", "system", "echo pwned"))

  def test_subprocess_popen(self):
    with self.assertRaises(pickle.UnpicklingError):
      safe_loads(_make_global_payload("subprocess", "Popen", "id"))

  def test_builtins_import(self):
    with self.assertRaises(pickle.UnpicklingError):
      safe_loads(_make_global_payload("builtins", "__import__", "os"))

  def test_posix_system(self):
    with self.assertRaises(pickle.UnpicklingError):
      safe_loads(_make_global_payload("posix", "system", "whoami"))

  def test_nt_system(self):
    with self.assertRaises(pickle.UnpicklingError):
      safe_loads(_make_global_payload("nt", "system", "whoami"))

  def test_builtins_eval(self):
    with self.assertRaises(pickle.UnpicklingError):
      safe_loads(
          _make_global_payload(
              "builtins", "eval", "__import__('os').system('id')"
          )
      )


class TestEventActionsRoundTrip(unittest.TestCase):
  """Legitimate EventActions data must survive pickle -> safe_loads."""

  def _round_trip(self, obj):
    return safe_loads(pickle.dumps(obj))

  def test_string_values(self):
    original = {"state_delta": {"key": "value"}, "artifact_delta": {}}
    self.assertEqual(self._round_trip(original), original)

  def test_nested_dict(self):
    original = {
        "state_delta": {
            "user_prefs": {"theme": "dark", "lang": "en"},
            "counter": 42,
        },
        "artifact_delta": {"files": ["a.txt", "b.txt"]},
    }
    self.assertEqual(self._round_trip(original), original)

  def test_none_and_bool(self):
    original = {
        "skip_summarization": True,
        "requested_auth_configs": None,
        "escalate": False,
    }
    self.assertEqual(self._round_trip(original), original)

  def test_empty_dict(self):
    self.assertEqual(self._round_trip({}), {})


class TestRealEventActionsRoundTrip(unittest.TestCase):
  """Smoke test: real EventActions instances survive pickle -> safe_loads."""

  def _round_trip(self, obj):
    return safe_loads(pickle.dumps(obj))

  def test_minimal_event_actions(self):
    original = EventActions()
    result = self._round_trip(original)
    self.assertIsInstance(result, EventActions)
    self.assertEqual(result.state_delta, {})
    self.assertEqual(result.artifact_delta, {})

  def test_event_actions_with_state_delta(self):
    original = EventActions(
        state_delta={"user_name": "alice", "turn_count": 3, "active": True},
        artifact_delta={"report.pdf": 2},
    )
    result = self._round_trip(original)
    self.assertIsInstance(result, EventActions)
    self.assertEqual(result.state_delta, original.state_delta)
    self.assertEqual(result.artifact_delta, original.artifact_delta)

  def test_event_actions_with_transfer_and_escalate(self):
    original = EventActions(
        transfer_to_agent="specialist_agent",
        escalate=True,
        skip_summarization=True,
    )
    result = self._round_trip(original)
    self.assertIsInstance(result, EventActions)
    self.assertEqual(result.transfer_to_agent, "specialist_agent")
    self.assertTrue(result.escalate)
    self.assertTrue(result.skip_summarization)

  def test_event_actions_with_complex_state_values(self):
    original = EventActions(
        state_delta={
            "nested": {"a": [1, 2, 3], "b": None},
            "count": 42,
            "tags": ["ml", "security"],
        },
    )
    result = self._round_trip(original)
    self.assertIsInstance(result, EventActions)
    self.assertEqual(result.state_delta["nested"]["a"], [1, 2, 3])
    self.assertIsNone(result.state_delta["nested"]["b"])


class TestEnvVarFallback(unittest.TestCase):
  """ADK_ALLOW_UNSAFE_V0_PICKLE=1 must bypass RestrictedUnpickler."""

  _ENV_KEY = "ADK_ALLOW_UNSAFE_V0_PICKLE"
  _PAYLOAD = _make_global_payload("collections", "Counter")

  def test_blocked_without_env_var(self):
    old = os.environ.pop(self._ENV_KEY, None)
    try:
      with self.assertRaises(pickle.UnpicklingError):
        safe_loads(self._PAYLOAD)
    finally:
      if old is not None:
        os.environ[self._ENV_KEY] = old

  def test_allowed_with_env_var(self):
    old = os.environ.get(self._ENV_KEY)
    try:
      os.environ[self._ENV_KEY] = "1"
      from collections import Counter

      result = safe_loads(self._PAYLOAD)
      self.assertIsInstance(result, Counter)
    finally:
      if old is None:
        os.environ.pop(self._ENV_KEY, None)
      else:
        os.environ[self._ENV_KEY] = old


if __name__ == "__main__":
  unittest.main()
