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

import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple
from unittest import mock

import click
import pytest

import src.google.adk.cli.cli_deploy as cli_deploy


# Helpers
class _Recorder:
  """A callable object that records every invocation."""

  def __init__(self) -> None:
    self.calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    self.calls.append((args, kwargs))

  def get_last_call_args(self) -> Tuple[Any, ...]:
    """Returns the positional arguments of the last call."""
    if not self.calls:
      raise IndexError("No calls have been recorded.")
    return self.calls[-1][0]

  def get_last_call_kwargs(self) -> Dict[str, Any]:
    """Returns the keyword arguments of the last call."""
    if not self.calls:
      raise IndexError("No calls have been recorded.")
    return self.calls[-1][1]


# Fixtures
@pytest.fixture(autouse=True)
def _mute_click(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppress click.echo to keep test output clean."""
  monkeypatch.setattr(click, "echo", lambda *a, **k: None)
  monkeypatch.setattr(click, "secho", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def reload_cli_deploy():
  """Reload cli_deploy before each test."""
  importlib.reload(cli_deploy)
  yield  # This allows the test to run after the module has been reloaded.


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Callable[[bool, bool], Path]:
  """
  Return a factory that creates a dummy agent directory tree.
  """

  def _factory(include_requirements: bool, include_env: bool) -> Path:
    base = tmp_path / "agent"
    base.mkdir()
    (base / "agent.py").write_text("# dummy agent")
    (base / "__init__.py").touch()
    if include_requirements:
      (base / "requirements.txt").write_text("pytest\n")
    if include_env:
      (base / ".env").write_text('TEST_VAR="test_value"\n')
    return base

  return _factory


# _resolve_project
def test_resolve_project_with_option() -> None:
  """It should return the explicit project value untouched."""
  assert cli_deploy._resolve_project("my-project") == "my-project"


def test_resolve_project_from_gcloud(monkeypatch: pytest.MonkeyPatch) -> None:
  """It should fall back to `gcloud config get-value project` when no value supplied."""
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda *a, **k: types.SimpleNamespace(stdout="gcp-proj\n"),
  )

  with mock.patch("click.echo") as mocked_echo:
    assert cli_deploy._resolve_project(None) == "gcp-proj"
    mocked_echo.assert_called_once()


def test_resolve_project_from_gcloud_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """It should raise an exception if the gcloud command fails."""
  monkeypatch.setattr(
      subprocess,
      "run",
      mock.Mock(side_effect=subprocess.CalledProcessError(1, "cmd", "err")),
  )
  with pytest.raises(subprocess.CalledProcessError):
    cli_deploy._resolve_project(None)


@pytest.mark.parametrize(
    "adk_version, session_uri, artifact_uri, memory_uri, expected",
    [
        (
            "1.3.0",
            "sqlite://s",
            "gs://a",
            "rag://m",
            (
                "--session_service_uri=sqlite://s --artifact_service_uri=gs://a"
                " --memory_service_uri=rag://m"
            ),
        ),
        (
            "1.2.5",
            "sqlite://s",
            "gs://a",
            "rag://m",
            "--session_db_url=sqlite://s --artifact_storage_uri=gs://a",
        ),
        (
            "0.5.0",
            "sqlite://s",
            "gs://a",
            "rag://m",
            "--session_db_url=sqlite://s",
        ),
        (
            "1.3.0",
            "sqlite://s",
            None,
            None,
            "--session_service_uri=sqlite://s  ",
        ),
        (
            "1.3.0",
            None,
            "gs://a",
            "rag://m",
            " --artifact_service_uri=gs://a --memory_service_uri=rag://m",
        ),
        ("1.2.0", None, "gs://a", None, " --artifact_storage_uri=gs://a"),
    ],
)
def test_get_service_option_by_adk_version(
    adk_version: str,
    session_uri: str | None,
    artifact_uri: str | None,
    memory_uri: str | None,
    expected: str,
) -> None:
  """It should return the correct service URI flags for a given ADK version."""
  actual = cli_deploy._get_service_option_by_adk_version(
      adk_version=adk_version,
      session_uri=session_uri,
      artifact_uri=artifact_uri,
      memory_uri=memory_uri,
  )
  assert actual.rstrip() == expected.rstrip()


@pytest.mark.parametrize("include_requirements", [True, False])
def test_to_gke_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool, bool], Path],
    tmp_path: Path,
    include_requirements: bool,
) -> None:
  """
  Tests the happy path for the `to_gke` function.
  """
  src_dir = agent_dir(include_requirements, False)
  run_recorder = _Recorder()
  rmtree_recorder = _Recorder()

  def mock_subprocess_run(*args, **kwargs):
    run_recorder(*args, **kwargs)
    command_list = args[0]
    if command_list and command_list[0:2] == ["kubectl", "apply"]:
      fake_stdout = "deployment.apps/gke-svc created\nservice/gke-svc created"
      return types.SimpleNamespace(stdout=fake_stdout)
    return None

  monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)

  cli_deploy.to_gke(
      agent_folder=str(src_dir),
      project="gke-proj",
      region="us-east1",
      cluster_name="my-gke-cluster",
      service_name="gke-svc",
      app_name="agent",
      temp_folder=str(tmp_path),
      port=9090,
      trace_to_cloud=False,
      with_ui=True,
      log_level="debug",
      adk_version="1.2.0",
      allow_origins=["http://localhost:3000", "https://my-app.com"],
      session_service_uri="sqlite:///",
      artifact_service_uri="gs://gke-bucket",
  )

  dockerfile_path = tmp_path / "Dockerfile"
  assert dockerfile_path.is_file()
  dockerfile_content = dockerfile_path.read_text()
  assert "CMD adk web --port=9090" in dockerfile_content
  assert "RUN pip install google-adk==1.2.0" in dockerfile_content

  assert len(run_recorder.calls) == 3, "Expected 3 subprocess calls"

  build_args = run_recorder.calls[0][0][0]
  expected_build_args = [
      "gcloud",
      "builds",
      "submit",
      "--tag",
      "gcr.io/gke-proj/gke-svc",
      "--verbosity",
      "debug",
      str(tmp_path),
  ]
  assert build_args == expected_build_args

  creds_args = run_recorder.calls[1][0][0]
  expected_creds_args = [
      "gcloud",
      "container",
      "clusters",
      "get-credentials",
      "my-gke-cluster",
      "--region",
      "us-east1",
      "--project",
      "gke-proj",
  ]
  assert creds_args == expected_creds_args

  assert (
      "--allow_origins=http://localhost:3000,https://my-app.com"
      in dockerfile_content
  )

  apply_args = run_recorder.calls[2][0][0]
  expected_apply_args = ["kubectl", "apply", "-f", str(tmp_path)]
  assert apply_args == expected_apply_args

  deployment_yaml_path = tmp_path / "deployment.yaml"
  assert deployment_yaml_path.is_file()
  yaml_content = deployment_yaml_path.read_text()

  assert "kind: Deployment" in yaml_content
  assert "kind: Service" in yaml_content
  assert "name: gke-svc" in yaml_content
  assert "image: gcr.io/gke-proj/gke-svc" in yaml_content
  assert f"containerPort: 9090" in yaml_content
  assert f"targetPort: 9090" in yaml_content
  assert "type: LoadBalancer" in yaml_content

  # 4. Verify cleanup
  assert str(rmtree_recorder.get_last_call_args()[0]) == str(tmp_path)
