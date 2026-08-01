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

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import google.adk.cli.agent_test_runner as agent_test_runner
from google.adk.events.event import Event
from google.genai import types


def test_rebuild_tests_preserves_non_ascii_event_text(tmp_path, monkeypatch):
  """Rebuilt test files preserve non-ASCII event text."""
  agent_dir = tmp_path / "test_agent"
  tests_dir = agent_dir / "tests"
  tests_dir.mkdir(parents=True)
  (agent_dir / "agent.py").write_text("", encoding="utf-8")
  test_file = tests_dir / "unicode.json"
  session_data = {
      "events": [{
          "author": "user",
          "content": {
              "role": "user",
              "parts": [{"text": "日本語の質問"}],
          },
      }]
  }
  test_file.write_text(
      json.dumps(session_data, ensure_ascii=False), encoding="utf-8"
  )

  class _Runner:

    def __init__(self):
      self.session = SimpleNamespace(user_id="test_user", id="test_session")
      self.runner = self

    async def run_async(self, **kwargs):
      del kwargs
      yield Event(
          author="test_agent",
          invocation_id="live-invocation",
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text="日本語の回答")],
          ),
      )

  loader = mock.create_autospec(
      agent_test_runner.AgentLoader, instance=True, spec_set=True
  )
  loader.load_agent.return_value = object()
  loader_factory = mock.create_autospec(
      agent_test_runner.AgentLoader, spec_set=True, return_value=loader
  )
  monkeypatch.setattr(
      agent_test_runner,
      "AgentLoader",
      loader_factory,
  )
  runner_factory = mock.create_autospec(
      agent_test_runner.InMemoryRunner,
      spec_set=True,
      return_value=_Runner(),
  )
  monkeypatch.setattr(
      agent_test_runner,
      "InMemoryRunner",
      runner_factory,
  )

  agent_test_runner.rebuild_tests(str(agent_dir))

  rebuilt = test_file.read_text(encoding="utf-8")
  assert "日本語の質問" in rebuilt
  assert "日本語の回答" in rebuilt
