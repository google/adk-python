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

"""Tests for ignore file support in cli_deploy."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import types
from typing import Any
from typing import Dict
from unittest import mock

import click
from google.adk.cli import cli_deploy
import pytest


@pytest.fixture(autouse=True)
def _mute_click(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppress click.echo to keep test output clean."""
  monkeypatch.setattr(click, "echo", lambda *_a, **_k: None)
  monkeypatch.setattr(click, "secho", lambda *_a, **_k: None)


def _fake_vertexai_module() -> types.ModuleType:
  """Returns a stand-in for the vertexai module that records nothing."""
  fake_vertexai = types.ModuleType("vertexai")

  class _FakeAgentEngines:

    def create(self, *, config: Dict[str, Any]) -> Any:
      del config
      return types.SimpleNamespace(
          api_resource=types.SimpleNamespace(
              name="projects/p/locations/l/reasoningEngines/e"
          )
      )

    def update(self, *, name: str, config: Dict[str, Any]) -> None:
      del name
      del config

  class _FakeVertexClient:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      del args
      del kwargs
      self.agent_engines = _FakeAgentEngines()

  fake_vertexai.Client = _FakeVertexClient
  return fake_vertexai


def test_get_ignore_patterns_func_excludes_dot_adk_without_ignore_files(
    tmp_path: Path,
) -> None:
  """The .adk folder is excluded even when the agent has no ignore files."""
  ignore_func = cli_deploy._get_ignore_patterns_func(str(tmp_path))

  names = [".adk", "agent.py", "a", "d", "k", ".", "/"]
  ignored = ignore_func(str(tmp_path), names)

  assert ".adk" in ignored
  # A set built from the string '.adk/' would ignore each character instead,
  # so single-character names must survive.
  assert ignored == {".adk"}


def test_get_ignore_patterns_func_combines_ignore_files(tmp_path: Path) -> None:
  """Patterns from all three ignore files are combined and normalized."""
  (tmp_path / ".gitignore").write_text(
      "# a comment\n\nignored_by_git.txt\n/rooted.txt\nbuild/\n"
  )
  (tmp_path / ".gcloudignore").write_text("ignored_by_gcloud.txt\n")
  (tmp_path / ".ae_ignore").write_text("ignored_by_ae.txt\n")

  ignore_func = cli_deploy._get_ignore_patterns_func(str(tmp_path))

  names = [
      "agent.py",
      "ignored_by_git.txt",
      "rooted.txt",
      "build",
      "ignored_by_gcloud.txt",
      "ignored_by_ae.txt",
      ".adk",
  ]
  ignored = ignore_func(str(tmp_path), names)

  assert ignored == {
      "ignored_by_git.txt",
      "rooted.txt",
      "build",
      "ignored_by_gcloud.txt",
      "ignored_by_ae.txt",
      ".adk",
  }


def test_get_ignore_patterns_func_warns_on_unreadable_ignore_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """An unreadable ignore file warns instead of aborting the deployment."""
  (tmp_path / ".gitignore").write_text("ignored.txt\n")

  def _raise(*_a: Any, **_k: Any) -> Any:
    raise OSError("boom")

  monkeypatch.setattr("builtins.open", _raise)

  ignore_func = cli_deploy._get_ignore_patterns_func(str(tmp_path))

  assert ignore_func(str(tmp_path), ["ignored.txt", ".adk"]) == {".adk"}


def test_to_cloud_run_respects_ignore_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """to_cloud_run respects .gitignore, .gcloudignore and the .adk folder."""
  agent_dir = tmp_path / "agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").write_text("# agent")
  (agent_dir / "__init__.py").write_text("")
  (agent_dir / "ignored_by_git.txt").write_text("ignored")
  (agent_dir / "ignored_by_gcloud.txt").write_text("ignored")
  (agent_dir / "ignored_rooted.txt").write_text("ignored")
  (agent_dir / "not_ignored.txt").write_text("keep")
  (agent_dir / ".adk").mkdir()
  (agent_dir / ".adk" / "session.db").write_text("db")

  # Use a root-anchored pattern (leading slash) to ensure it is honored.
  (agent_dir / ".gitignore").write_text(
      "ignored_by_git.txt\n/ignored_rooted.txt\n"
  )
  (agent_dir / ".gcloudignore").write_text("ignored_by_gcloud.txt\n")

  temp_deploy_dir = tmp_path / "temp_deploy"

  # Mock subprocess.run to avoid actual gcloud call
  monkeypatch.setattr(subprocess, "run", mock.Mock())
  # Mock shutil.rmtree to keep the temp folder for verification
  original_rmtree = shutil.rmtree
  monkeypatch.setattr(
      shutil,
      "rmtree",
      lambda path, **kwargs: None
      if "temp_deploy" in str(path)
      else original_rmtree(path, **kwargs),
  )

  cli_deploy.to_cloud_run(
      agent_folder=str(agent_dir),
      project="proj",
      region="us-central1",
      service_name="svc",
      app_name="app",
      temp_folder=str(temp_deploy_dir),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
  )

  agent_src_path = temp_deploy_dir / "agents" / "app"

  assert (agent_src_path / "agent.py").exists()
  assert (agent_src_path / "not_ignored.txt").exists()

  # These should be ignored
  assert not (
      agent_src_path / "ignored_by_git.txt"
  ).exists(), "Should respect .gitignore"
  assert not (
      agent_src_path / "ignored_by_gcloud.txt"
  ).exists(), "Should respect .gcloudignore"
  assert not (
      agent_src_path / "ignored_rooted.txt"
  ).exists(), "Should respect root-anchored (leading slash) patterns"
  assert not (
      agent_src_path / ".adk"
  ).exists(), "Should exclude the local .adk folder"


def test_to_agent_engine_respects_multiple_ignore_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """to_agent_engine respects .gitignore, .gcloudignore and .ae_ignore."""
  project_dir = tmp_path / "project"
  project_dir.mkdir()
  monkeypatch.chdir(project_dir)

  agent_dir = project_dir / "my_agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").write_text("root_agent = None")
  (agent_dir / "__init__.py").write_text("from . import agent")
  (agent_dir / "ignored_by_git.txt").write_text("ignored")
  (agent_dir / "ignored_by_ae.txt").write_text("ignored")
  (agent_dir / ".adk").mkdir()
  (agent_dir / ".adk" / "session.db").write_text("db")

  (agent_dir / ".gitignore").write_text("ignored_by_git.txt\n")
  (agent_dir / ".ae_ignore").write_text("ignored_by_ae.txt\n")

  monkeypatch.setitem(sys.modules, "vertexai", _fake_vertexai_module())
  # Mock shutil.rmtree to keep the temp folder for verification
  original_rmtree = shutil.rmtree

  def mock_rmtree(path, **kwargs):
    if "_tmp" in str(path):
      return None
    return original_rmtree(path, **kwargs)

  monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

  cli_deploy.to_agent_engine(
      agent_folder=str(agent_dir),
      adk_app="adk_app",
      project="my-gcp-project",
      region="us-central1",
  )

  # Find the temp folder created by to_agent_engine
  temp_folders = [
      d for d in project_dir.iterdir() if d.is_dir() and "_tmp" in d.name
  ]
  assert len(temp_folders) == 1
  agent_src_path = temp_folders[0]

  assert (agent_src_path / "agent.py").exists()
  assert not (
      agent_src_path / "ignored_by_git.txt"
  ).exists(), "Should respect .gitignore"
  assert not (
      agent_src_path / "ignored_by_ae.txt"
  ).exists(), "Should respect .ae_ignore"
  assert not (
      agent_src_path / ".adk"
  ).exists(), "Should exclude the local .adk folder"


def test_to_gke_respects_ignore_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """to_gke respects ignore files."""
  agent_dir = tmp_path / "agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").write_text("# agent")
  (agent_dir / "__init__.py").write_text("")
  (agent_dir / "ignored.txt").write_text("ignored")
  (agent_dir / ".gitignore").write_text("ignored.txt\n")
  (agent_dir / ".adk").mkdir()
  (agent_dir / ".adk" / "session.db").write_text("db")

  temp_deploy_dir = tmp_path / "temp_deploy"

  # Mock subprocess.run to avoid actual gcloud call
  mock_run = mock.Mock()
  mock_run.return_value.stdout = "deployment created"
  monkeypatch.setattr(subprocess, "run", mock_run)
  # Mock shutil.rmtree to keep the temp folder for verification
  original_rmtree = shutil.rmtree
  monkeypatch.setattr(
      shutil,
      "rmtree",
      lambda path, **kwargs: None
      if "temp_deploy" in str(path)
      else original_rmtree(path, **kwargs),
  )

  cli_deploy.to_gke(
      agent_folder=str(agent_dir),
      project="proj",
      region="us-central1",
      cluster_name="cluster",
      service_name="svc",
      app_name="app",
      temp_folder=str(temp_deploy_dir),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      adk_version="1.0.0",
  )

  agent_src_path = temp_deploy_dir / "agents" / "app"

  assert (agent_src_path / "agent.py").exists()
  assert not (
      agent_src_path / "ignored.txt"
  ).exists(), "Should respect .gitignore"
  assert not (
      agent_src_path / ".adk"
  ).exists(), "Should exclude the local .adk folder"
