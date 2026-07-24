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

"""Unit tests for utilities in cli_eval."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from google.adk.cli.cli_eval import get_root_agent
import pytest


@pytest.mark.parametrize(
    ("input_eval_set", "expected"),
    [
        pytest.param(
            r"C:\tmp\agent\eval.evalset.json",
            {r"C:\tmp\agent\eval.evalset.json": []},
            id="windows-backslash-path-without-selectors",
        ),
        pytest.param(
            r"C:\tmp\agent\eval.evalset.json:case1",
            {r"C:\tmp\agent\eval.evalset.json": ["case1"]},
            id="windows-backslash-path-with-one-selector",
        ),
        pytest.param(
            r"C:\tmp\agent\eval.evalset.json:case1,case2",
            {r"C:\tmp\agent\eval.evalset.json": ["case1", "case2"]},
            id="windows-backslash-path-with-multiple-selectors",
        ),
        pytest.param(
            "C:/tmp/agent/eval.evalset.json:case1",
            {"C:/tmp/agent/eval.evalset.json": ["case1"]},
            id="windows-forward-slash-path",
        ),
        pytest.param(
            r"d:\tmp\agent\eval.evalset.json:case1",
            {r"d:\tmp\agent\eval.evalset.json": ["case1"]},
            id="lowercase-windows-drive",
        ),
        pytest.param(
            "/tmp/agent/eval.evalset.json:case1,case2",
            {"/tmp/agent/eval.evalset.json": ["case1", "case2"]},
            id="posix-path-with-selectors",
        ),
        pytest.param(
            "my_eval_set:case1,case2",
            {"my_eval_set": ["case1", "case2"]},
            id="eval-set-id-with-selectors",
        ),
        pytest.param(
            "my_eval_set",
            {"my_eval_set": []},
            id="eval-set-id-without-selectors",
        ),
        pytest.param(
            "/tmp/agent/eval.evalset.json",
            {"/tmp/agent/eval.evalset.json": []},
            id="posix-path-without-selectors",
        ),
    ],
)
def test_parse_and_get_evals_to_run_parses_eval_set_and_selectors(
    input_eval_set: str, expected: dict[str, list[str]]
):
  """Eval-set paths and IDs retain their optional case selectors."""
  from google.adk.cli.cli_eval import parse_and_get_evals_to_run

  assert parse_and_get_evals_to_run([input_eval_set]) == expected


def test_get_eval_sets_manager_local(monkeypatch):
  mock_local_manager = mock.MagicMock()
  monkeypatch.setattr(
      "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager",
      lambda *a, **k: mock_local_manager,
  )
  from google.adk.cli.cli_eval import get_eval_sets_manager

  manager = get_eval_sets_manager(eval_storage_uri=None, agents_dir="some/dir")
  assert manager == mock_local_manager


def test_get_eval_sets_manager_gcs(monkeypatch):
  mock_gcs_manager = mock.MagicMock()
  mock_create_gcs = mock.MagicMock()
  mock_create_gcs.return_value = SimpleNamespace(
      eval_sets_manager=mock_gcs_manager
  )
  monkeypatch.setattr(
      "google.adk.cli.utils.evals.create_gcs_eval_managers_from_uri",
      mock_create_gcs,
  )
  from google.adk.cli.cli_eval import get_eval_sets_manager

  manager = get_eval_sets_manager(
      eval_storage_uri="gs://bucket", agents_dir="some/dir"
  )
  assert manager == mock_gcs_manager
  mock_create_gcs.assert_called_once_with("gs://bucket")


@pytest.mark.asyncio
async def test_get_root_agent_supports_root_agent(monkeypatch):
  root_agent = mock.MagicMock()
  agent_module = SimpleNamespace(agent=SimpleNamespace(root_agent=root_agent))
  monkeypatch.setattr(
      "google.adk.cli.cli_eval._get_agent_module",
      lambda _agent_module_file_path: agent_module,
  )
  assert await get_root_agent("some/dir") == root_agent


@pytest.mark.asyncio
async def test_get_root_agent_supports_get_agent_async(monkeypatch):
  root_agent = mock.MagicMock()
  get_agent_async = mock.AsyncMock(return_value=(root_agent, object()))
  agent_module = SimpleNamespace(
      agent=SimpleNamespace(get_agent_async=get_agent_async)
  )
  monkeypatch.setattr(
      "google.adk.cli.cli_eval._get_agent_module",
      lambda _agent_module_file_path: agent_module,
  )
  assert await get_root_agent("some/dir") == root_agent
  get_agent_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_root_agent_raises_without_supported_entrypoint(monkeypatch):
  agent_module = SimpleNamespace(agent=SimpleNamespace())
  monkeypatch.setattr(
      "google.adk.cli.cli_eval._get_agent_module",
      lambda _agent_module_file_path: agent_module,
  )
  with pytest.raises(ValueError, match="root_agent|get_agent_async"):
    await get_root_agent("some/dir")


def test_parse_evals_preserves_windows_drive_in_file_path(tmp_path):
  from google.adk.cli.cli_eval import parse_and_get_evals_to_run

  eval_set_file = tmp_path / "evals.json"
  eval_set_file.write_text("{}", encoding="utf-8")

  assert parse_and_get_evals_to_run([str(eval_set_file)]) == {
      str(eval_set_file): []
  }


def test_parse_evals_preserves_missing_windows_drive_path():
  from google.adk.cli.cli_eval import parse_and_get_evals_to_run

  eval_set_file = r"C:\missing\evals.json"

  assert parse_and_get_evals_to_run([eval_set_file]) == {eval_set_file: []}


def test_parse_evals_splits_case_selector_from_right():
  from google.adk.cli.cli_eval import parse_and_get_evals_to_run

  assert parse_and_get_evals_to_run([r"C:\evals\set.json:case1,case2"]) == {
      r"C:\evals\set.json": ["case1", "case2"]
  }
