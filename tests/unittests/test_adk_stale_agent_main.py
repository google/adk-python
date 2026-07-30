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

"""Tests for the adk_stale_agent sample's batch outcome accounting.

Regression test for a bug where `process_single_issue` swallowed exceptions
and still reported success, letting `main()` log "Successfully processed N
issues" and exit 0 even when every issue in the batch failed.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ADK_TEAM_DIR = (
    Path(__file__).parent.parent.parent
    / "contributing"
    / "samples"
    / "adk_team"
)


class _FailingRunner:
  """A fake InMemoryRunner whose run_async always raises."""

  def __init__(self, **kwargs):
    self.session_service = self

  async def create_session(self, **kwargs):
    return SimpleNamespace(id="synthetic-session")

  async def run_async(self, **kwargs):
    if False:  # pragma: no cover - makes this an async generator.
      yield None
    raise RuntimeError("synthetic runner failure")


def _unload_adk_stale_agent():
  for mod_name in list(sys.modules):
    if mod_name == "adk_stale_agent" or mod_name.startswith("adk_stale_agent."):
      del sys.modules[mod_name]


@pytest.fixture
def stale_agent_main(monkeypatch):
  """Imports adk_stale_agent.main with a fake GitHub token in place."""
  monkeypatch.setenv("GITHUB_TOKEN", "test-token")
  monkeypatch.syspath_prepend(str(ADK_TEAM_DIR))
  _unload_adk_stale_agent()

  main = importlib.import_module("adk_stale_agent.main")
  yield main

  _unload_adk_stale_agent()


def test_main_returns_false_when_every_issue_fails(
    stale_agent_main, monkeypatch
):
  main = stale_agent_main
  monkeypatch.setattr(main, "InMemoryRunner", _FailingRunner)
  monkeypatch.setattr(
      main,
      "get_old_open_issue_numbers",
      lambda *args, **kwargs: [101, 202, 303],
  )
  monkeypatch.setattr(main, "get_api_call_count", lambda: 0)
  monkeypatch.setattr(main, "reset_api_call_count", lambda: None)

  assert asyncio.run(main.main()) is False


def test_process_single_issue_reports_failure(stale_agent_main, monkeypatch):
  main = stale_agent_main
  monkeypatch.setattr(main, "InMemoryRunner", _FailingRunner)
  monkeypatch.setattr(main, "get_api_call_count", lambda: 0)

  success, _, _ = asyncio.run(main.process_single_issue(101))

  assert success is False
