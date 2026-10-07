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

"""Tests for _ConformanceTestGemini's handling of missing replay recordings."""

from google.adk.cli.conformance._conformance_test_google_llm import _ConformanceTestGemini
from google.adk.cli.conformance._conformance_test_google_llm import ReplayVerificationError
from google.adk.cli.plugins.recordings_schema import Recordings
import pytest


def test_missing_recordings_raises_actionable_error():
  """Replaying without the replay plugin loaded should fail clearly.

  `config['_adk_replay_recordings']` is only populated by ReplayPlugin's
  before_run_callback. If the server serving `adk conformance test` wasn't
  started with that plugin, the key is absent and `config.get(...)` returns
  None. Silently indexing into `None.recordings` used to raise a bare
  AttributeError; it should now explain what's missing.
  """
  config = {
      'user_message_index': 0,
      'agent_name': 'root_agent',
      'current_replay_index': 0,
  }

  with pytest.raises(ReplayVerificationError, match='ReplayPlugin'):
    _ConformanceTestGemini(config=config)


def test_present_recordings_does_not_raise():
  """Sanity check: a properly loaded config constructs without error."""
  config = {
      '_adk_replay_recordings': Recordings(recordings=[]),
      'user_message_index': 0,
      'agent_name': 'root_agent',
      'current_replay_index': 0,
  }

  model = _ConformanceTestGemini(config=config)

  assert model._agent_llm_recordings == []
