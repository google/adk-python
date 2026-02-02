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
    (base / "agent.py").write_text(
        "# dummy agent\nroot_agent = 'dummy_agent'\n"
    )
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
    "adk_version, session_uri, artifact_uri, memory_uri, use_local_storage, "
    "expected",
    [
        (
            "1.3.0",
            "sqlite://s",
            "gs://a",
            "rag://m",
            None,
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
            None,
            "--session_db_url=sqlite://s --artifact_storage_uri=gs://a",
        ),
        (
            "0.5.0",
            "sqlite://s",
            "gs://a",
            "rag://m",
            None,
            "--session_db_url=sqlite://s",
        ),
        (
            "1.3.0",
            "sqlite://s",
            None,
            None,
            None,
            "--session_service_uri=sqlite://s",
        ),
        (
            "1.3.0",
            None,
            "gs://a",
            "rag://m",
            None,
            "--artifact_service_uri=gs://a --memory_service_uri=rag://m",
        ),
        (
            "1.2.0",
            None,
            "gs://a",
            None,
            None,
            "--artifact_storage_uri=gs://a",
        ),
        (
            "1.21.0",
            None,
            None,
            None,
            False,
            "--no_use_local_storage",
        ),
        (
            "1.21.0",
            None,
            None,
            None,
            True,
            "--use_local_storage",
        ),
        (
            "1.21.0",
            "sqlite://s",
            "gs://a",
            None,
            False,
            "--session_service_uri=sqlite://s --artifact_service_uri=gs://a",
        ),
    ],
)
def test_get_service_option_by_adk_version(
    adk_version: str,
    session_uri: str | None,
    artifact_uri: str | None,
    memory_uri: str | None,
    use_local_storage: bool | None,
    expected: str,
) -> None:
  """It should return the correct service URI flags for a given ADK version."""
  actual = cli_deploy._get_service_option_by_adk_version(
      adk_version=adk_version,
      session_uri=session_uri,
      artifact_uri=artifact_uri,
      memory_uri=memory_uri,
      use_local_storage=use_local_storage,
  )
  assert actual.rstrip() == expected.rstrip()


def test_agent_engine_app_template_compiles_with_windows_paths() -> None:
  """It should not emit invalid Python when paths contain `\\u` segments."""
  rendered = cli_deploy._AGENT_ENGINE_APP_TEMPLATE.format(
      is_config_agent=True,
      agent_folder=r".\user_agent_tmp20260101_000000",
      adk_app_object="root_agent",
      adk_app_type="agent",
      trace_to_cloud_option=False,
      express_mode=False,
  )
  compile(rendered, "<agent_engine_app.py>", "exec")


@pytest.mark.parametrize("include_requirements", [True, False])
def test_to_agent_engine_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool, bool], Path],
    include_requirements: bool,
) -> None:
  """Tests the happy path for the `to_agent_engine` function."""
  rmtree_recorder = _Recorder()
  monkeypatch.setattr(shutil, "rmtree", rmtree_recorder)
  create_recorder = _Recorder()

  fake_vertexai = types.ModuleType("vertexai")

  class _FakeAgentEngines:

    def create(self, *, config: Dict[str, Any]) -> Any:
      create_recorder(config=config)
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
  monkeypatch.setitem(sys.modules, "vertexai", fake_vertexai)
  src_dir = agent_dir(include_requirements, False)
  tmp_dir = src_dir.parent / "tmp"
  cli_deploy.to_agent_engine(
      agent_folder=str(src_dir),
      temp_folder="tmp",
      adk_app="my_adk_app",
      trace_to_cloud=True,
      project="my-gcp-project",
      region="us-central1",
      display_name="My Test Agent",
      description="A test agent.",
  )
  agent_file = tmp_dir / "agent.py"
  assert agent_file.is_file()
  init_file = tmp_dir / "__init__.py"
  assert init_file.is_file()
  adk_app_file = tmp_dir / "my_adk_app.py"
  assert adk_app_file.is_file()
  content = adk_app_file.read_text()
  assert "from .agent import root_agent" in content
  assert "adk_app = AdkApp(" in content
  assert "agent=root_agent" in content
  assert "enable_tracing=True" in content
  reqs_path = tmp_dir / "requirements.txt"
  assert reqs_path.is_file()
  assert "google-cloud-aiplatform[adk,agent_engines]" in reqs_path.read_text()
  assert len(create_recorder.calls) == 1
  assert str(rmtree_recorder.get_last_call_args()[0]) == str(tmp_dir)


def test_to_agent_engine_skips_agent_import_validation_by_default(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool, bool], Path],
) -> None:
  """It should skip agent.py import validation by default."""
  validate_recorder = _Recorder()

  def _validate_agent_import(*args: Any, **kwargs: Any) -> None:
    validate_recorder(*args, **kwargs)
    raise AssertionError("_validate_agent_import should not be called")

  monkeypatch.setattr(
      cli_deploy, "_validate_agent_import", _validate_agent_import
  )

  fake_vertexai = types.ModuleType("vertexai")

  class _FakeAgentEngines:

    def create(self, *, config: Dict[str, Any]) -> Any:
      del config
      return types.SimpleNamespace(
          api_resource=types.SimpleNamespace(
              name="projects/p/locations/l/reasoningEngines/e"
          )
      )

  class _FakeVertexClient:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      del args
      del kwargs
      self.agent_engines = _FakeAgentEngines()

  fake_vertexai.Client = _FakeVertexClient
  monkeypatch.setitem(sys.modules, "vertexai", fake_vertexai)

  src_dir = agent_dir(False, False)
  cli_deploy.to_agent_engine(
      agent_folder=str(src_dir),
      temp_folder="tmp",
      adk_app="my_adk_app",
      trace_to_cloud=True,
      project="my-gcp-project",
      region="us-central1",
      display_name="My Test Agent",
      description="A test agent.",
  )

  assert validate_recorder.calls == []


def test_to_agent_engine_validates_agent_import_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: Callable[[bool, bool], Path],
) -> None:
  """It should run agent.py import validation when enabled."""
  validate_recorder = _Recorder()

  def _validate_agent_import(*args: Any, **kwargs: Any) -> None:
    validate_recorder(*args, **kwargs)

  monkeypatch.setattr(
      cli_deploy, "_validate_agent_import", _validate_agent_import
  )

  fake_vertexai = types.ModuleType("vertexai")

  class _FakeAgentEngines:

    def create(self, *, config: Dict[str, Any]) -> Any:
      del config
      return types.SimpleNamespace(
          api_resource=types.SimpleNamespace(
              name="projects/p/locations/l/reasoningEngines/e"
          )
      )

  class _FakeVertexClient:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      del args
      del kwargs
      self.agent_engines = _FakeAgentEngines()

  fake_vertexai.Client = _FakeVertexClient
  monkeypatch.setitem(sys.modules, "vertexai", fake_vertexai)

  src_dir = agent_dir(False, False)
  cli_deploy.to_agent_engine(
      agent_folder=str(src_dir),
      temp_folder="tmp",
      adk_app="my_adk_app",
      trace_to_cloud=True,
      project="my-gcp-project",
      region="us-central1",
      display_name="My Test Agent",
      description="A test agent.",
      skip_agent_import_validation=False,
  )

  assert len(validate_recorder.calls) == 1


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
      otel_to_cloud=False,
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


# _validate_agent_import tests
class TestValidateAgentImport:
  """Tests for the _validate_agent_import function."""

  def test_skips_config_agents(self, tmp_path: Path) -> None:
    """Config agents should skip validation."""
    # This should not raise even with no agent.py file
    cli_deploy._validate_agent_import(
        str(tmp_path), "root_agent", is_config_agent=True
    )

  def test_raises_on_missing_agent_module(self, tmp_path: Path) -> None:
    """Should raise when agent.py is missing."""
    with pytest.raises(click.ClickException) as exc_info:
      cli_deploy._validate_agent_import(
          str(tmp_path), "root_agent", is_config_agent=False
      )
    assert "Agent module not found" in str(exc_info.value)

  def test_raises_on_missing_export(self, tmp_path: Path) -> None:
    """Should raise when the expected export is missing."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("some_other_var = 'hello'\n")
    (tmp_path / "__init__.py").touch()

    with pytest.raises(click.ClickException) as exc_info:
      cli_deploy._validate_agent_import(
          str(tmp_path), "root_agent", is_config_agent=False
      )
    assert "does not export 'root_agent'" in str(exc_info.value)
    assert "some_other_var" in str(exc_info.value)

  def test_success_with_root_agent_export(self, tmp_path: Path) -> None:
    """Should succeed when root_agent is exported."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("root_agent = 'my_agent'\n")
    (tmp_path / "__init__.py").touch()

    # Should not raise
    cli_deploy._validate_agent_import(
        str(tmp_path), "root_agent", is_config_agent=False
    )

  def test_success_with_app_export(self, tmp_path: Path) -> None:
    """Should succeed when app is exported."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("app = 'my_app'\n")
    (tmp_path / "__init__.py").touch()

    # Should not raise
    cli_deploy._validate_agent_import(
        str(tmp_path), "app", is_config_agent=False
    )

  def test_success_with_relative_imports(self, tmp_path: Path) -> None:
    """Should succeed when agent.py uses relative imports."""
    (tmp_path / "helper.py").write_text("VALUE = 'my_agent'\n")
    (tmp_path / "__init__.py").touch()
    (tmp_path / "agent.py").write_text(
        "from .helper import VALUE\n\nroot_agent = VALUE\n"
    )

    cli_deploy._validate_agent_import(
        str(tmp_path), "root_agent", is_config_agent=False
    )

  def test_raises_on_import_error(self, tmp_path: Path) -> None:
    """Should raise with helpful message on ImportError."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("from nonexistent_module import something\n")
    (tmp_path / "__init__.py").touch()

    with pytest.raises(click.ClickException) as exc_info:
      cli_deploy._validate_agent_import(
          str(tmp_path), "root_agent", is_config_agent=False
      )
    assert "Failed to import agent module" in str(exc_info.value)
    assert "nonexistent_module" in str(exc_info.value)

  def test_raises_on_basellm_import_error(self, tmp_path: Path) -> None:
    """Should provide specific guidance for BaseLlm import errors."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text(
        "from google.adk.models.base_llm import NonexistentBaseLlm\n"
    )
    (tmp_path / "__init__.py").touch()

    with pytest.raises(click.ClickException) as exc_info:
      cli_deploy._validate_agent_import(
          str(tmp_path), "root_agent", is_config_agent=False
      )
    assert "BaseLlm-related error" in str(exc_info.value)
    assert "custom LLM" in str(exc_info.value)

  def test_raises_on_syntax_error(self, tmp_path: Path) -> None:
    """Should raise on syntax errors in agent.py."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("def invalid syntax here:\n")
    (tmp_path / "__init__.py").touch()

    with pytest.raises(click.ClickException) as exc_info:
      cli_deploy._validate_agent_import(
          str(tmp_path), "root_agent", is_config_agent=False
      )
    assert "Error while loading agent module" in str(exc_info.value)

  def test_cleans_up_sys_modules(self, tmp_path: Path) -> None:
    """Should clean up sys.modules after validation."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("root_agent = 'my_agent'\n")
    (tmp_path / "__init__.py").touch()

    module_name = tmp_path.name
    agent_module_key = f"{module_name}.agent"

    # Ensure module is not in sys.modules before
    assert module_name not in sys.modules
    assert agent_module_key not in sys.modules

    cli_deploy._validate_agent_import(
        str(tmp_path), "root_agent", is_config_agent=False
    )

    # Ensure module is cleaned up after
    assert module_name not in sys.modules
    assert agent_module_key not in sys.modules

  def test_restores_sys_path(self, tmp_path: Path) -> None:
    """Should restore sys.path after validation."""
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("root_agent = 'my_agent'\n")
    (tmp_path / "__init__.py").touch()

    original_path = sys.path.copy()

    cli_deploy._validate_agent_import(
        str(tmp_path), "root_agent", is_config_agent=False
    )

    assert sys.path == original_path
