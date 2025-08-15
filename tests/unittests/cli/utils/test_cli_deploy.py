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
import tempfile
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

  Args:
    tmp_path: The temporary path fixture provided by pytest.

  Returns:
    A factory function that takes two booleans:
    - include_requirements: Whether to include a `requirements.txt` file.
    - include_env: Whether to include a `.env` file.
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


@pytest.fixture
def mock_vertex_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[mock.MagicMock, None, None]:
  """Mocks the entire vertexai module and its sub-modules."""
  mock_vertexai = mock.MagicMock()
  mock_agent_engines = mock.MagicMock()
  mock_vertexai.agent_engines = mock_agent_engines
  mock_vertexai.init = mock.MagicMock()
  mock_agent_engines.create = mock.MagicMock()
  mock_agent_engines.ModuleAgent = mock.MagicMock(
      return_value="mock-agent-engine-object"
  )

  sys.modules["vertexai"] = mock_vertexai
  sys.modules["vertexai.agent_engines"] = mock_agent_engines

  # Also mock dotenv
  mock_dotenv = mock.MagicMock()
  mock_dotenv.dotenv_values = mock.MagicMock(return_value={"FILE_VAR": "value"})
  sys.modules["dotenv"] = mock_dotenv

  yield mock_vertexai

  # Cleanup: remove mocks from sys.modules
  del sys.modules["vertexai"]
  del sys.modules["vertexai.agent_engines"]
  del sys.modules["dotenv"]


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

# _get_service_option_by_adk_version
def test_get_service_option_by_adk_version(
    adk_version: str,
    session_uri: str | None,
    artifact_uri: str | None,
    memory_uri: str | None,
    expected: str,
) -> None:
  """It should return the correct service URI flags for a given ADK version."""
  assert (
      cli_deploy._get_service_option_by_adk_version(
          adk_version=adk_version,
          session_uri=session_uri,
          artifact_uri=artifact_uri,
          memory_uri=memory_uri,
      )
      == expected
  )


@pytest.mark.parametrize("include_requirements", [True, False])
@pytest.mark.parametrize("with_ui", [True, False])
def test_to_cloud_run_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool, bool], Path],
    tmp_path: Path,
    include_requirements: bool,
    with_ui: bool,
) -> None:
  """
  End-to-end execution test for `to_cloud_run`.

  This test verifies that for a given configuration:
  1. The agent source files are correctly copied to a temporary build context.
  2. A valid Dockerfile is generated with the correct parameters.
  3. The `gcloud run deploy` command is constructed with the correct arguments.
  """
  src_dir = agent_dir(include_requirements, False)
  run_recorder = _Recorder()

  monkeypatch.setattr(subprocess, "run", run_recorder)
  # Mock rmtree to prevent actual deletion during test run but record calls
  rmtree_recorder = _Recorder()
  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)

  # Execute the function under test
  cli_deploy.to_cloud_run(
      agent_folder=str(src_dir),
      project="proj",
      region="asia-northeast1",
      service_name="svc",
      app_name="agent",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=True,
      with_ui=with_ui,
      log_level="info",
      verbosity="info",
      allow_origins=["http://localhost:3000", "https://my-app.com"],
      session_service_uri="sqlite://",
      artifact_service_uri="gs://bucket",
      memory_service_uri="rag://",
      adk_version="1.3.0",
  )

  # 1. Assert that source files were copied correctly
  agent_dest_path = tmp_path / "agents" / "agent"
  assert (agent_dest_path / "agent.py").is_file()
  assert (agent_dest_path / "__init__.py").is_file()
  assert (
      agent_dest_path / "requirements.txt"
  ).is_file() == include_requirements

  # 2. Assert that the Dockerfile was generated correctly
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

  if include_requirements:
    assert (
        'RUN pip install -r "/app/agents/agent/requirements.txt"'
        in dockerfile_content
    )
  else:
    assert "RUN pip install -r" not in dockerfile_content

  assert (
      "--allow_origins=http://localhost:3000,https://my-app.com"
      in dockerfile_content
  )

  # 3. Assert that the gcloud command was constructed correctly
  assert len(run_recorder.calls) == 1
  gcloud_args = run_recorder.get_last_call_args()[0]

  expected_gcloud_command = [
      "gcloud",
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

  # 4. Assert cleanup was performed
  assert str(rmtree_recorder.get_last_call_args()[0]) == str(tmp_path)


def test_to_cloud_run_cleans_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool], Path],
) -> None:
  """`to_cloud_run` should always delete the temporary folder on exit."""
  tmp_dir = Path(tempfile.mkdtemp())
  src_dir = agent_dir(False, False)

  deleted: Dict[str, Path] = {}

  def _fake_rmtree(path: str | Path, *a: Any, **k: Any) -> None:
    deleted["path"] = Path(path)

  monkeypatch.setattr(cli_deploy.shutil, "rmtree", _fake_rmtree)
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
    agent_dir: Callable[[bool, bool], Path],
) -> None:
  """`to_cloud_run` should always delete the temporary folder on exit, even if gcloud fails."""
  tmp_dir = Path(tempfile.mkdtemp())
  src_dir = agent_dir(False, False)

  rmtree_recorder = _Recorder()
  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)
  # Make the gcloud command fail
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
        with_ui=False,
        log_level="info",
        verbosity="info",
        adk_version="1.0.0",
        session_service_uri=None,
        artifact_service_uri=None,
        memory_service_uri=None,
    )

  # Check that rmtree was called on the temp folder in the finally block
  assert rmtree_recorder.calls, "shutil.rmtree should have been called"
  assert str(rmtree_recorder.get_last_call_args()[0]) == str(tmp_dir)


@pytest.mark.usefixtures("mock_vertex_ai")
@pytest.mark.parametrize("has_reqs", [True, False])
@pytest.mark.parametrize("has_env", [True, False])
def test_to_agent_engine_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool, bool], Path],
    tmp_path: Path,
    has_reqs: bool,
    has_env: bool,
) -> None:
  """
  Tests the happy path for the `to_agent_engine` function.

  Verifies:
  1. Source files are copied.
  2. `adk_app.py` is created correctly.
  3. `requirements.txt` is handled (created if not present).
  4. `.env` file is read if present.
  5. `vertexai.init` and `agent_engines.create` are called with the correct args.
  6. Cleanup is performed.
  """
  src_dir = agent_dir(has_reqs, has_env)
  temp_folder = tmp_path / "build"
  app_name = src_dir.name
  rmtree_recorder = _Recorder()

  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)

  # Execute
  cli_deploy.to_agent_engine(
      agent_folder=str(src_dir),
      temp_folder=str(temp_folder),
      adk_app="my_adk_app",
      staging_bucket="gs://my-staging-bucket",
      trace_to_cloud=True,
      project="my-gcp-project",
      region="us-central1",
      display_name="My Test Agent",
      description="A test agent.",
  )

  # 1. Verify file operations
  assert (temp_folder / app_name / "agent.py").is_file()
  assert (temp_folder / app_name / "__init__.py").is_file()

  # 2. Verify adk_app.py creation
  adk_app_path = temp_folder / "my_adk_app.py"
  assert adk_app_path.is_file()
  content = adk_app_path.read_text()
  assert f"from {app_name}.agent import root_agent" in content
  assert "adk_app = AdkApp(" in content
  assert "enable_tracing=True" in content

  # 3. Verify requirements handling
  reqs_path = temp_folder / app_name / "requirements.txt"
  assert reqs_path.is_file()
  if not has_reqs:
    # It should have been created with the default content
    assert "google-cloud-aiplatform[adk,agent_engines]" in reqs_path.read_text()

  # 4. Verify Vertex AI SDK calls
  vertexai = sys.modules["vertexai"]
  vertexai.init.assert_called_once_with(
      project="my-gcp-project",
      location="us-central1",
      staging_bucket="gs://my-staging-bucket",
  )

  # 5. Verify env var handling
  dotenv = sys.modules["dotenv"]
  if has_env:
    dotenv.dotenv_values.assert_called_once()
    expected_env_vars = {"FILE_VAR": "value"}
  else:
    dotenv.dotenv_values.assert_not_called()
    expected_env_vars = None

  # 6. Verify agent_engines.create call
  vertexai.agent_engines.create.assert_called_once()
  create_kwargs = vertexai.agent_engines.create.call_args.kwargs
  assert create_kwargs["agent_engine"] == "mock-agent-engine-object"
  assert create_kwargs["display_name"] == "My Test Agent"
  assert create_kwargs["description"] == "A test agent."
  assert create_kwargs["requirements"] == str(reqs_path)
  assert create_kwargs["extra_packages"] == [str(temp_folder)]
  assert create_kwargs["env_vars"] == expected_env_vars

  # 7. Verify cleanup
  assert str(rmtree_recorder.get_last_call_args()[0]) == str(temp_folder)


@pytest.mark.parametrize("include_requirements", [True, False])
def test_to_gke_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool, bool], Path],
    tmp_path: Path,
    include_requirements: bool,
) -> None:
  """
  Tests the happy path for the `to_gke` function.

  Verifies:
  1. Source files are copied and Dockerfile is created.
  2. `gcloud builds submit` is called to build the image.
  3. `deployment.yaml` is created with the correct content.
  4. `gcloud container get-credentials` and `kubectl apply` are called.
  5. Cleanup is performed.
  """
  src_dir = agent_dir(include_requirements, False)
  run_recorder = _Recorder()
  rmtree_recorder = _Recorder()

  def mock_subprocess_run(*args, **kwargs):
    # We still use the recorder to check which commands were called
    run_recorder(*args, **kwargs)

    # The command is the first positional argument, e.g., ['kubectl', 'apply', ...]
    command_list = args[0]

    # Check if this is the 'kubectl apply' call
    if command_list and command_list[0:2] == ["kubectl", "apply"]:
      # If it is, return a fake process object with a .stdout attribute
      # This mimics the real output from kubectl.
      fake_stdout = "deployment.apps/gke-svc created\nservice/gke-svc created"
      return types.SimpleNamespace(stdout=fake_stdout)

    # For all other subprocess.run calls (like 'gcloud builds submit'),
    # we don't need a return value, so the default None is fine.
    return None

  monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)

  # Execute
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

  # 1. Verify Dockerfile (basic check)
  dockerfile_path = tmp_path / "Dockerfile"
  assert dockerfile_path.is_file()
  dockerfile_content = dockerfile_path.read_text()
  assert "CMD adk web --port=9090" in dockerfile_content
  assert "RUN pip install google-adk==1.2.0" in dockerfile_content

  # 2. Verify command executions by checking each recorded call
  assert len(run_recorder.calls) == 3, "Expected 3 subprocess calls"

  # Call 1: gcloud builds submit
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

  # Call 2: gcloud container clusters get-credentials
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

  # Call 3: kubectl apply
  apply_args = run_recorder.calls[2][0][0]
  expected_apply_args = ["kubectl", "apply", "-f", str(tmp_path)]
  assert apply_args == expected_apply_args

  # 3. Verify deployment.yaml content
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
