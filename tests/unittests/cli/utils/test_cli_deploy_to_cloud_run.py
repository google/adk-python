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

"""Tests for to_cloud_run functionality in cli_deploy."""


from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from typing import Dict
from typing import List
from typing import Protocol
from typing import Tuple
from unittest import mock

import click
import pytest

import src.google.adk.cli.cli_deploy as cli_deploy


class AgentDirFixture(Protocol):
  """Protocol for the agent_dir pytest fixture factory."""

  def __call__(self, *, include_requirements: bool, include_env: bool) -> Path:
    ...


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
  monkeypatch.setattr(click, "echo", lambda *_a, **_k: None)
  monkeypatch.setattr(click, "secho", lambda *_a, **_k: None)


@pytest.fixture()
def agent_dir(tmp_path: Path) -> AgentDirFixture:
  """
  Return a factory that creates a dummy agent directory tree.
  """

  def _factory(*, include_requirements: bool, include_env: bool) -> Path:
    base = tmp_path / "agent"
    base.mkdir()
    (base / "agent.py").write_text("# dummy agent")
    (base / "__init__.py").write_text("from . import agent")
    if include_requirements:
      (base / "requirements.txt").write_text("pytest\n")
    if include_env:
      (base / ".env").write_text('TEST_VAR="test_value"\n')
    return base

  return _factory


@pytest.mark.parametrize("include_requirements", [True, False])
@pytest.mark.parametrize("with_ui", [True, False])
def test_to_cloud_run_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
    include_requirements: bool,
    with_ui: bool,
) -> None:
  """
  End-to-end execution test for `to_cloud_run`.
  """
  src_dir = agent_dir(
      include_requirements=include_requirements, include_env=False
  )
  run_recorder = _Recorder()

  monkeypatch.setattr(subprocess, "run", run_recorder)
  rmtree_recorder = _Recorder()
  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)

  cli_deploy.to_cloud_run(
      agent_folder=str(src_dir),
      project="proj",
      region="asia-northeast1",
      service_name="svc",
      app_name="agent",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=True,
      otel_to_cloud=True,
      with_ui=with_ui,
      log_level="info",
      verbosity="info",
      allow_origins=["http://localhost:3000", "https://my-app.com"],
      session_service_uri="sqlite://",
      artifact_service_uri="gs://bucket",
      memory_service_uri="rag://",
      adk_version="1.3.0",
  )

  agent_dest_path = tmp_path / "agents" / "agent"
  assert (agent_dest_path / "agent.py").is_file()
  assert (agent_dest_path / "__init__.py").is_file()
  assert (
      agent_dest_path / "requirements.txt"
  ).is_file() == include_requirements

  dockerfile_path = tmp_path / "Dockerfile"
  assert dockerfile_path.is_file()
  dockerfile_content = dockerfile_path.read_text()

  expected_command = "web" if with_ui else "api_server"
  assert f"CMD adk {expected_command} --port=8080" in dockerfile_content
  assert "FROM python:3.11-slim" in dockerfile_content
  assert (
      'RUN adduser --disabled-password --gecos "" myuser' in dockerfile_content
  )
  assert "USER myuser" in dockerfile_content
  assert "ENV GOOGLE_CLOUD_PROJECT=proj" in dockerfile_content
  assert "ENV GOOGLE_CLOUD_LOCATION=asia-northeast1" in dockerfile_content
  assert "RUN pip install google-adk==1.3.0" in dockerfile_content
  assert "--trace_to_cloud" in dockerfile_content
  assert "--otel_to_cloud" in dockerfile_content

  # Check agent dependencies installation based on include_requirements
  if include_requirements:
    assert (
        'RUN pip install -r "/app/agents/agent/requirements.txt"'
        in dockerfile_content
    )
  else:
    assert "# No requirements.txt found." in dockerfile_content

  assert (
      "--allow_origins=http://localhost:3000,https://my-app.com"
      in dockerfile_content
  )

  assert len(run_recorder.calls) == 1
  gcloud_args = run_recorder.get_last_call_args()[0]

  expected_gcloud_command = [
      cli_deploy._GCLOUD_CMD,
      "run",
      "deploy",
      "svc",
      "--source",
      str(tmp_path),
      "--project",
      "proj",
      "--region",
      "asia-northeast1",
      "--port",
      "8080",
      "--verbosity",
      "info",
      "--labels",
      "created-by=adk",
  ]
  assert gcloud_args == expected_gcloud_command

  assert str(rmtree_recorder.get_last_call_args()[0]) == str(tmp_path)


def test_to_cloud_run_cleans_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
) -> None:
  """`to_cloud_run` should always delete the temporary folder on exit."""
  tmp_dir = Path(tempfile.mkdtemp())
  src_dir = agent_dir(include_requirements=False, include_env=False)

  deleted: Dict[str, Path] = {}

  def _fake_rmtree(path: str | Path, *_a: Any, **_k: Any) -> None:
    deleted["path"] = Path(path)

  monkeypatch.setattr(shutil, "rmtree", _fake_rmtree)
  monkeypatch.setattr(subprocess, "run", _Recorder())

  cli_deploy.to_cloud_run(
      agent_folder=str(src_dir),
      project="proj",
      region=None,
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_dir),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
      session_service_uri=None,
      artifact_service_uri=None,
      memory_service_uri=None,
  )

  assert deleted["path"] == tmp_dir


def test_to_cloud_run_cleans_temp_dir_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
) -> None:
  """`to_cloud_run` should delete the temp folder on exit, even if gcloud fails."""
  tmp_dir = Path(tempfile.mkdtemp())
  src_dir = agent_dir(include_requirements=False, include_env=False)

  rmtree_recorder = _Recorder()
  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)
  monkeypatch.setattr(
      subprocess,
      "run",
      mock.Mock(side_effect=subprocess.CalledProcessError(1, "gcloud")),
  )

  with pytest.raises(subprocess.CalledProcessError):
    cli_deploy.to_cloud_run(
        agent_folder=str(src_dir),
        project="proj",
        region="us-central1",
        service_name="svc",
        app_name="app",
        temp_folder=str(tmp_dir),
        port=8080,
        trace_to_cloud=False,
        otel_to_cloud=False,
        with_ui=False,
        log_level="info",
        verbosity="info",
        adk_version="1.0.0",
        session_service_uri=None,
        artifact_service_uri=None,
        memory_service_uri=None,
    )

  assert rmtree_recorder.calls, "shutil.rmtree should have been called"
  assert str(rmtree_recorder.get_last_call_args()[0]) == str(tmp_dir)


# Label merging tests
@pytest.mark.parametrize(
    "extra_gcloud_args, expected_labels",
    [
        # No user labels - should only have default ADK label
        (None, "created-by=adk"),
        ([], "created-by=adk"),
        # Single user label
        (["--labels=env=test"], "created-by=adk,env=test"),
        # Multiple user labels in same argument
        (
            ["--labels=env=test,team=myteam"],
            "created-by=adk,env=test,team=myteam",
        ),
        # User labels mixed with other args
        (
            ["--memory=1Gi", "--labels=env=test", "--cpu=1"],
            "created-by=adk,env=test",
        ),
        # Multiple --labels arguments
        (
            ["--labels=env=test", "--labels=team=myteam"],
            "created-by=adk,env=test,team=myteam",
        ),
        # Labels with other passthrough args
        (
            ["--timeout=300", "--labels=env=prod", "--max-instances=10"],
            "created-by=adk,env=prod",
        ),
    ],
)
def test_cloud_run_label_merging(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
    extra_gcloud_args: list[str] | None,
    expected_labels: str,
) -> None:
  """Test that user labels are properly merged with the default ADK label."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  run_recorder = _Recorder()

  monkeypatch.setattr(subprocess, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda _x: None)

  # Execute the function under test
  cli_deploy.to_cloud_run(
      agent_folder=str(src_dir),
      project="test-project",
      region="us-central1",
      service_name="test-service",
      app_name="test-app",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
      extra_gcloud_args=tuple(extra_gcloud_args) if extra_gcloud_args else None,
  )

  # Verify that the gcloud command was called
  assert len(run_recorder.calls) == 1
  gcloud_args = run_recorder.get_last_call_args()[0]

  # Find the labels argument
  labels_idx = gcloud_args.index("--labels")
  actual_labels = gcloud_args[labels_idx + 1]

  assert actual_labels == expected_labels
