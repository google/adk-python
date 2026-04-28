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

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App


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


def _patch_agent_module(monkeypatch, agent_namespace):
  """Patches `_get_agent_module` to return a stub whose `.agent` matches."""
  monkeypatch.setattr(
      "google.adk.cli.cli_eval._get_agent_module",
      lambda _path: SimpleNamespace(agent=agent_namespace),
  )


def test_get_app_or_root_agent_with_app(monkeypatch):
  """When the module exposes an App, both app and its root_agent are returned."""
  root_agent = BaseAgent(name="root_agent")
  app = App(name="my_app", root_agent=root_agent)
  _patch_agent_module(monkeypatch, SimpleNamespace(root_agent=root_agent, app=app))

  from google.adk.cli.cli_eval import get_app_or_root_agent

  resolved_app, resolved_root = get_app_or_root_agent("some/path")
  assert resolved_app is app
  assert resolved_root is root_agent


def test_get_app_or_root_agent_without_app(monkeypatch):
  """When only `root_agent` is exposed, app is None."""
  root_agent = BaseAgent(name="root_agent")
  _patch_agent_module(monkeypatch, SimpleNamespace(root_agent=root_agent))

  from google.adk.cli.cli_eval import get_app_or_root_agent

  resolved_app, resolved_root = get_app_or_root_agent("some/path")
  assert resolved_app is None
  assert resolved_root is root_agent


def test_get_app_or_root_agent_app_attribute_not_an_app_instance(monkeypatch):
  """If `app` exists but is not an App, it is ignored and we fall back."""
  root_agent = BaseAgent(name="root_agent")
  _patch_agent_module(
      monkeypatch,
      SimpleNamespace(root_agent=root_agent, app="not-an-app"),
  )

  from google.adk.cli.cli_eval import get_app_or_root_agent

  resolved_app, resolved_root = get_app_or_root_agent("some/path")
  assert resolved_app is None
  assert resolved_root is root_agent


def test_get_root_agent_back_compat(monkeypatch):
  """Existing `get_root_agent` callers keep getting the bare agent back."""
  root_agent = BaseAgent(name="root_agent")
  app = App(name="my_app", root_agent=root_agent)
  _patch_agent_module(monkeypatch, SimpleNamespace(root_agent=root_agent, app=app))

  from google.adk.cli.cli_eval import get_root_agent

  assert get_root_agent("some/path") is root_agent
