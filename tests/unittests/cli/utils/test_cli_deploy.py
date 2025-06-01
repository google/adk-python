# Copyright 2025 Google LLC
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

"""Tests for utilities in cli_deploy."""


from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from unittest import mock

import click
import google.adk.cli.cli_deploy as cli_deploy
from google.adk.cli.deployers.deployer_factory import DeployerFactory
import pytest


# Helpers
class _Recorder:
  """A callable object that records every invocation."""

  def __init__(self) -> None:
    self.calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    self.calls.append((args, kwargs))


# Fixtures
@pytest.fixture(autouse=True)
def _mute_click(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppress click.echo to keep test output clean."""
  monkeypatch.setattr(click, "echo", lambda *a, **k: None)


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Callable[[bool], Path]:
  """Return a factory that creates a dummy agent directory tree."""

  def _factory(include_requirements: bool) -> Path:
    base = tmp_path / "agent"
    base.mkdir()
    (base / "agent.py").write_text("# dummy agent")
    (base / "__init__.py").touch()
    if include_requirements:
      (base / "requirements.txt").write_text("pytest\n")
    return base

  return _factory


# _resolve_project
def test_resolve_project_with_option() -> None:
  """It should return the explicit project value untouched."""
  cloudRunDeployer = DeployerFactory.get_deployer("cloud_run")
  assert cloudRunDeployer._resolve_project("my-project") == "my-project"


def test_resolve_project_from_gcloud(monkeypatch: pytest.MonkeyPatch) -> None:
  """It should fall back to `gcloud config get-value project` when no value supplied."""
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda *a, **k: types.SimpleNamespace(stdout="gcp-proj\n"),
  )

  with mock.patch("click.echo") as mocked_echo:
    cloudRunDeployer = DeployerFactory.get_deployer("cloud_run")
    assert cloudRunDeployer._resolve_project(None) == "gcp-proj"
    mocked_echo.assert_called_once()


# cli_deploy.run with cloud_run
@pytest.mark.parametrize("include_requirements", [True, False])
def test_deploy_cloud_run_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool], Path],
    include_requirements: bool,
) -> None:
  """
  End-to-end execution test for `cli_deploy.run` with cloud_run covering both presence and
  absence of *requirements.txt*.
  """
  tmp_dir = Path(tempfile.mkdtemp())
  src_dir = agent_dir(include_requirements)

  copy_recorder = _Recorder()
  run_recorder = _Recorder()

  # Cache the ORIGINAL copytree before patching
  original_copytree = cli_deploy.shutil.copytree

  def _recording_copytree(*args: Any, **kwargs: Any):
    copy_recorder(*args, **kwargs)
    return original_copytree(*args, **kwargs)

  monkeypatch.setattr(cli_deploy.shutil, "copytree", _recording_copytree)
  # Skip actual cleanup so that we can inspect generated files later.
  monkeypatch.setattr(cli_deploy.shutil, "rmtree", lambda *_a, **_k: None)
  monkeypatch.setattr(subprocess, "run", run_recorder)

  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="cloud_run",
      project="proj",
      region="asia-northeast1",
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_dir),
      port=8080,
      trace_to_cloud=True,
      with_ui=True,
      verbosity="info",
      session_db_url="sqlite://",
      artifact_storage_uri="gs://bucket",
      adk_version="0.0.5",
      provider_args="TEST_ARG=ARG1",
      env="TEST_ENV=1",
  )

  # Assertions
  assert (
      len(copy_recorder.calls) == 1
  ), "Agent sources must be copied exactly once."
  assert run_recorder.calls, "gcloud command should be executed at least once."
  assert (tmp_dir / "Dockerfile").exists(), "Dockerfile must be generated."

  # Manual cleanup because we disabled rmtree in the monkeypatch.
  shutil.rmtree(tmp_dir, ignore_errors=True)


# cli_deploy.run with docker
@pytest.mark.parametrize("include_requirements", [True, False])
def test_deploy_run_docker_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool], Path],
    include_requirements: bool,
) -> None:
  """
  End-to-end execution test for `cli_deploy.run` with docker covering both presence and
  absence of *requirements.txt*.
  """
  tmp_dir = Path(tempfile.mkdtemp())
  src_dir = agent_dir(include_requirements)

  copy_recorder = _Recorder()
  run_recorder = _Recorder()

  # Cache the ORIGINAL copytree before patching
  original_copytree = cli_deploy.shutil.copytree

  def _recording_copytree(*args: Any, **kwargs: Any):
    copy_recorder(*args, **kwargs)
    return original_copytree(*args, **kwargs)

  monkeypatch.setattr(cli_deploy.shutil, "copytree", _recording_copytree)
  # Skip actual cleanup so that we can inspect generated files later.
  monkeypatch.setattr(cli_deploy.shutil, "rmtree", lambda *_a, **_k: None)
  monkeypatch.setattr(subprocess, "run", run_recorder)

  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="docker",
      project=None,
      region=None,
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_dir),
      port=8080,
      trace_to_cloud=True,
      with_ui=True,
      verbosity="info",
      session_db_url="sqlite://",
      artifact_storage_uri="gs://bucket",
      adk_version="0.0.5",
      provider_args="TEST_ARG=ARG1",
      env="TEST_ENV=1",
  )

  # Assertions
  assert (
      len(copy_recorder.calls) == 1
  ), "Agent sources must be copied exactly once."
  assert run_recorder.calls, "gcloud command should be executed at least once."
  assert (tmp_dir / "Dockerfile").exists(), "Dockerfile must be generated."

  # Manual cleanup because we disabled rmtree in the monkeypatch.
  shutil.rmtree(tmp_dir, ignore_errors=True)


def test_deploy_cloud_run_cleans_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool], Path],
) -> None:
  """`to_cloud_run` should always delete the temporary folder on exit."""
  tmp_dir = Path(tempfile.mkdtemp())
  src_dir = agent_dir(False)

  deleted: Dict[str, Path] = {}

  def _fake_rmtree(path: str | Path, *a: Any, **k: Any) -> None:
    deleted["path"] = Path(path)

  monkeypatch.setattr(cli_deploy.shutil, "rmtree", _fake_rmtree)
  monkeypatch.setattr(subprocess, "run", _Recorder())

  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="cloud_run",
      project="proj",
      region=None,
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_dir),
      port=8080,
      trace_to_cloud=False,
      with_ui=False,
      verbosity="info",
      session_db_url=None,
      artifact_storage_uri=None,
      adk_version="0.0.5",
      provider_args="TEST_ARG=ARG1",
      env="TEST_ENV=1",
  )

  assert deleted["path"] == tmp_dir
