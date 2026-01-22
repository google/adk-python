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

"""Tests for utilities in cli_tool_click."""


from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from unittest import mock

import click
from click.testing import CliRunner
from google.adk.agents.base_agent import BaseAgent
from google.adk.cli import cli_tools_click
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from pydantic import BaseModel
import pytest


class DummyAgent(BaseAgent):

  def __init__(self, name):
    super().__init__(name=name)
    self.sub_agents = []


root_agent = DummyAgent(name="dummy_agent")


@pytest.fixture
def mock_load_eval_set_from_file():
  with mock.patch(
      "google.adk.evaluation.local_eval_sets_manager.load_eval_set_from_file"
  ) as mock_func:
    yield mock_func


@pytest.fixture
def mock_get_root_agent():
  with mock.patch("google.adk.cli.cli_eval.get_root_agent") as mock_func:
    mock_func.return_value = root_agent
    yield mock_func


# Helpers
class _Recorder(BaseModel):
  """Callable that records every invocation."""

  calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

  def __call__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
    self.calls.append((args, kwargs))


# Fixtures
@pytest.fixture(autouse=True)
def _mute_click(request, monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppress click output during tests."""
  # Allow tests to opt-out of muting by using the 'unmute_click' marker
  if "unmute_click" in request.keywords:
    return
  monkeypatch.setattr(click, "echo", lambda *a, **k: None)
  # Keep secho for error messages
  # monkeypatch.setattr(click, "secho", lambda *a, **k: None)


# validate_exclusive
def test_validate_exclusive_allows_single() -> None:
  """Providing exactly one exclusive option should pass."""
  ctx = click.Context(cli_tools_click.cli_run)
  param = SimpleNamespace(name="replay")
  assert (
      cli_tools_click.validate_exclusive(ctx, param, "file.json") == "file.json"
  )


def test_validate_exclusive_blocks_multiple() -> None:
  """Providing two exclusive options should raise UsageError."""
  ctx = click.Context(cli_tools_click.cli_run)
  param1 = SimpleNamespace(name="replay")
  param2 = SimpleNamespace(name="resume")

  # First option registers fine
  cli_tools_click.validate_exclusive(ctx, param1, "replay.json")

  # Second option triggers conflict
  with pytest.raises(click.UsageError):
    cli_tools_click.validate_exclusive(ctx, param2, "resume.json")


# cli create
def test_cli_create_cmd_invokes_run_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """`adk create` should forward arguments to cli_create.run_cmd."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_create, "run_cmd", rec)

  app_dir = tmp_path / "my_app"
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      ["create", "--model", "gemini", "--api_key", "key123", str(app_dir)],
  )
  assert result.exit_code == 0, (result.output, repr(result.exception))
  assert rec.calls, "cli_create.run_cmd must be called"


# cli run
@pytest.mark.parametrize(
    "cli_args,expected_session_uri,expected_artifact_uri",
    [
        pytest.param(
            [
                "--session_service_uri",
                "memory://",
                "--artifact_service_uri",
                "memory://",
            ],
            "memory://",
            "memory://",
            id="memory_scheme_uris",
        ),
        pytest.param(
            [],
            None,
            None,
            id="default_uris_none",
        ),
    ],
)
def test_cli_run_service_uris(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_args: list,
    expected_session_uri: str,
    expected_artifact_uri: str,
) -> None:
  """`adk run` should forward service URIs correctly to run_cli."""
  agent_dir = tmp_path / "agent"
  agent_dir.mkdir()
  (agent_dir / "__init__.py").touch()
  (agent_dir / "agent.py").touch()

  # Capture the coroutine's locals before closing it
  captured_locals = []

  def capture_asyncio_run(coro):
    # Extract the locals before closing the coroutine
    if coro.cr_frame is not None:
      captured_locals.append(dict(coro.cr_frame.f_locals))
    coro.close()  # Properly close the coroutine to avoid warnings

  monkeypatch.setattr(cli_tools_click.asyncio, "run", capture_asyncio_run)

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      ["run", *cli_args, str(agent_dir)],
  )
  assert result.exit_code == 0, (result.output, repr(result.exception))
  assert len(captured_locals) == 1, "Expected asyncio.run to be called once"

  # Verify the kwargs passed to run_cli
  coro_locals = captured_locals[0]
  assert coro_locals.get("session_service_uri") == expected_session_uri
  assert coro_locals.get("artifact_service_uri") == expected_artifact_uri
  assert coro_locals["agent_folder_name"] == "agent"


# cli deploy cloud_run
def test_cli_deploy_cloud_run_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Successful path should call cli_deploy.to_cloud_run once."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_cloud_run", rec)

  agent_dir = tmp_path / "agent2"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "proj",
          "--region",
          "asia-northeast1",
          str(agent_dir),
      ],
  )
  assert result.exit_code == 0
  assert rec.calls, "cli_deploy.to_cloud_run must be invoked"


def test_cli_deploy_cloud_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Exception from to_cloud_run should be caught and surfaced via click.secho."""

  def _boom(*_a: Any, **_k: Any) -> None:  # noqa: D401
    raise RuntimeError("boom")

  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_cloud_run", _boom)

  agent_dir = tmp_path / "agent3"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main, ["deploy", "cloud_run", str(agent_dir)]
  )

  assert result.exit_code == 0
  assert "Deploy failed: boom" in result.output


def test_cli_deploy_cloud_run_passthrough_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Extra args after '--' should be passed through to the gcloud command."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_cloud_run", rec)

  agent_dir = tmp_path / "agent_passthrough"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "test-project",
          "--region",
          "us-central1",
          str(agent_dir),
          "--",
          "--labels=test-label=test",
          "--memory=1Gi",
          "--cpu=1",
      ],
  )
  # Print debug information if the test fails
  if result.exit_code != 0:
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

  assert result.exit_code == 0
  assert rec.calls, "cli_deploy.to_cloud_run must be invoked"

  # Check that extra_gcloud_args were passed correctly
  called_kwargs = rec.calls[0][1]
  extra_args = called_kwargs.get("extra_gcloud_args")
  assert extra_args is not None
  assert "--labels=test-label=test" in extra_args
  assert "--memory=1Gi" in extra_args
  assert "--cpu=1" in extra_args


def test_cli_deploy_cloud_run_rejects_args_without_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Args without '--' separator should be rejected with helpful error message."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_cloud_run", rec)

  agent_dir = tmp_path / "agent_no_sep"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "test-project",
          "--region",
          "us-central1",
          str(agent_dir),
          "--labels=test-label=test",  # This should be rejected
      ],
  )

  assert result.exit_code == 2
  assert "Unexpected arguments:" in result.output
  assert "Use '--' to separate gcloud arguments" in result.output
  assert not rec.calls, "cli_deploy.to_cloud_run should not be called"


def test_cli_deploy_cloud_run_rejects_args_before_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Args before '--' separator should be rejected."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_cloud_run", rec)

  agent_dir = tmp_path / "agent_before_sep"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "test-project",
          "--region",
          "us-central1",
          str(agent_dir),
          "unexpected_arg",  # This should be rejected
          "--",
          "--labels=test-label=test",
      ],
  )

  assert result.exit_code == 2
  assert (
      "Unexpected arguments after agent path and before '--':" in result.output
  )
  assert "unexpected_arg" in result.output
  assert not rec.calls, "cli_deploy.to_cloud_run should not be called"


def test_cli_deploy_cloud_run_allows_empty_gcloud_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """No gcloud args after '--' should be allowed."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_cloud_run", rec)

  agent_dir = tmp_path / "agent_empty_gcloud"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "test-project",
          "--region",
          "us-central1",
          str(agent_dir),
          "--",
          # No gcloud args after --
      ],
  )

  assert result.exit_code == 0
  assert rec.calls, "cli_deploy.to_cloud_run must be invoked"

  # Check that extra_gcloud_args is empty
  called_kwargs = rec.calls[0][1]
  extra_args = called_kwargs.get("extra_gcloud_args")
  assert extra_args == ()


# cli deploy agent_engine
def test_cli_deploy_agent_engine_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Successful path should call cli_deploy.to_agent_engine."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_agent_engine", rec)

  agent_dir = tmp_path / "agent_ae"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "agent_engine",
          "--project",
          "test-proj",
          "--region",
          "us-central1",
          str(agent_dir),
      ],
  )
  assert result.exit_code == 0
  assert rec.calls, "cli_deploy.to_agent_engine must be invoked"
  called_kwargs = rec.calls[0][1]
  assert called_kwargs.get("project") == "test-proj"
  assert called_kwargs.get("region") == "us-central1"


# cli deploy agent_engine with --otel_to_cloud
def test_cli_deploy_agent_engine_otel_to_cloud_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Successful path should call cli_deploy.to_agent_engine with --otel_to_cloud."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_agent_engine", rec)

  agent_dir = tmp_path / "agent_ae"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "agent_engine",
          "--project",
          "test-proj",
          "--region",
          "us-central1",
          "--otel_to_cloud",
          str(agent_dir),
      ],
  )
  assert result.exit_code == 0
  assert rec.calls, "cli_deploy.to_agent_engine must be invoked"
  called_kwargs = rec.calls[0][1]
  assert called_kwargs.get("project") == "test-proj"
  assert called_kwargs.get("region") == "us-central1"
  assert called_kwargs.get("otel_to_cloud")


# cli deploy gke
def test_cli_deploy_gke_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Successful path should call cli_deploy.to_gke."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click.cli_deploy, "to_gke", rec)

  agent_dir = tmp_path / "agent_gke"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "gke",
          "--project",
          "test-proj",
          "--region",
          "us-central1",
          "--cluster_name",
          "my-cluster",
          str(agent_dir),
      ],
  )
  assert result.exit_code == 0
  assert rec.calls, "cli_deploy.to_gke must be invoked"
  called_kwargs = rec.calls[0][1]
  assert called_kwargs.get("project") == "test-proj"
  assert called_kwargs.get("region") == "us-central1"
  assert called_kwargs.get("cluster_name") == "my-cluster"


# cli eval
def test_cli_eval_missing_deps_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """If cli_eval sub-module is missing, command should raise ClickException."""
  orig_import = builtins.__import__

  def _fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    if name == "google.adk.cli.cli_eval" or (level > 0 and "cli_eval" in name):
      raise ModuleNotFoundError(f"Simulating missing {name}")
    return orig_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", _fake_import)

  agent_dir = tmp_path / "agent_missing_deps"
  agent_dir.mkdir()
  (agent_dir / "__init__.py").touch()
  eval_file = tmp_path / "dummy.json"
  eval_file.touch()

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      ["eval", str(agent_dir), str(eval_file)],
  )
  assert result.exit_code != 0
  assert isinstance(result.exception, SystemExit)
  assert cli_tools_click.MISSING_EVAL_DEPENDENCIES_MESSAGE in result.output


# cli web & api_server (uvicorn patched)
@pytest.fixture()
def _patch_uvicorn(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
  """Patch uvicorn.Config/Server to avoid real network operations."""
  rec = _Recorder()

  class _DummyServer:

    def __init__(self, *a: Any, **k: Any) -> None:
      ...

    def run(self) -> None:
      rec()

  monkeypatch.setattr(
      cli_tools_click.uvicorn, "Config", lambda *a, **k: object()
  )
  monkeypatch.setattr(
      cli_tools_click.uvicorn, "Server", lambda *_a, **_k: _DummyServer()
  )
  return rec


def test_cli_web_invokes_uvicorn(
    tmp_path: Path, _patch_uvicorn: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
  """`adk web` should configure and start uvicorn.Server.run."""
  agents_dir = tmp_path / "agents"
  agents_dir.mkdir()
  monkeypatch.setattr(
      cli_tools_click, "get_fast_api_app", lambda **_k: object()
  )
  runner = CliRunner()
  result = runner.invoke(cli_tools_click.main, ["web", str(agents_dir)])
  assert result.exit_code == 0
  assert _patch_uvicorn.calls, "uvicorn.Server.run must be called"


def test_cli_api_server_invokes_uvicorn(
    tmp_path: Path, _patch_uvicorn: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
  """`adk api_server` should configure and start uvicorn.Server.run."""
  agents_dir = tmp_path / "agents_api"
  agents_dir.mkdir()
  monkeypatch.setattr(
      cli_tools_click, "get_fast_api_app", lambda **_k: object()
  )
  runner = CliRunner()
  result = runner.invoke(cli_tools_click.main, ["api_server", str(agents_dir)])
  assert result.exit_code == 0
  assert _patch_uvicorn.calls, "uvicorn.Server.run must be called"


def test_cli_web_passes_service_uris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_uvicorn: _Recorder
) -> None:
  """`adk web` should pass service URIs to get_fast_api_app."""
  agents_dir = tmp_path / "agents"
  agents_dir.mkdir()

  mock_get_app = _Recorder()
  monkeypatch.setattr(cli_tools_click, "get_fast_api_app", mock_get_app)

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "web",
          str(agents_dir),
          "--session_service_uri",
          "sqlite:///test.db",
          "--artifact_service_uri",
          "gs://mybucket",
          "--memory_service_uri",
          "rag://mycorpus",
      ],
  )
  assert result.exit_code == 0
  assert mock_get_app.calls
  called_kwargs = mock_get_app.calls[0][1]
  assert called_kwargs.get("session_service_uri") == "sqlite:///test.db"
  assert called_kwargs.get("artifact_service_uri") == "gs://mybucket"
  assert called_kwargs.get("memory_service_uri") == "rag://mycorpus"


@pytest.mark.unmute_click
def test_cli_web_warns_and_maps_deprecated_uris(
    tmp_path: Path,
    _patch_uvicorn: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """`adk web` should accept deprecated URI flags with warnings."""
  agents_dir = tmp_path / "agents"
  agents_dir.mkdir()

  mock_get_app = _Recorder()
  monkeypatch.setattr(cli_tools_click, "get_fast_api_app", mock_get_app)

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "web",
          str(agents_dir),
          "--session_db_url",
          "sqlite:///deprecated.db",
          "--artifact_storage_uri",
          "gs://deprecated",
      ],
  )

  assert result.exit_code == 0
  called_kwargs = mock_get_app.calls[0][1]
  assert called_kwargs.get("session_service_uri") == "sqlite:///deprecated.db"
  assert called_kwargs.get("artifact_service_uri") == "gs://deprecated"
  # Check output for deprecation warnings (CliRunner captures both stdout and stderr)
  assert "--session_db_url" in result.output
  assert "--artifact_storage_uri" in result.output


def test_cli_eval_with_eval_set_file_path(
    mock_load_eval_set_from_file,
    mock_get_root_agent,
    tmp_path,
):
  agent_path = tmp_path / "my_agent"
  agent_path.mkdir()
  (agent_path / "__init__.py").touch()

  eval_set_file = tmp_path / "my_evals.json"
  eval_set_file.write_text("{}")

  mock_load_eval_set_from_file.return_value = EvalSet(
      eval_set_id="my_evals",
      eval_cases=[EvalCase(eval_id="case1", conversation=[])],
  )

  result = CliRunner().invoke(
      cli_tools_click.cli_eval,
      [str(agent_path), str(eval_set_file)],
  )

  assert result.exit_code == 0
  # Assert that we wrote eval set results
  eval_set_results_manager = LocalEvalSetResultsManager(
      agents_dir=str(tmp_path)
  )
  eval_set_results = eval_set_results_manager.list_eval_set_results(
      app_name="my_agent"
  )
  assert len(eval_set_results) == 1


def test_cli_eval_with_eval_set_id(
    mock_get_root_agent,
    tmp_path,
):
  app_name = "test_app"
  eval_set_id = "test_eval_set_id"
  agent_path = tmp_path / app_name
  agent_path.mkdir()
  (agent_path / "__init__.py").touch()

  eval_sets_manager = LocalEvalSetsManager(agents_dir=str(tmp_path))
  eval_sets_manager.create_eval_set(app_name=app_name, eval_set_id=eval_set_id)
  eval_sets_manager.add_eval_case(
      app_name=app_name,
      eval_set_id=eval_set_id,
      eval_case=EvalCase(eval_id="case1", conversation=[]),
  )
  eval_sets_manager.add_eval_case(
      app_name=app_name,
      eval_set_id=eval_set_id,
      eval_case=EvalCase(eval_id="case2", conversation=[]),
  )

  result = CliRunner().invoke(
      cli_tools_click.cli_eval,
      [str(agent_path), "test_eval_set_id:case1,case2"],
  )

  assert result.exit_code == 0
  # Assert that we wrote eval set results
  eval_set_results_manager = LocalEvalSetResultsManager(
      agents_dir=str(tmp_path)
  )
  eval_set_results = eval_set_results_manager.list_eval_set_results(
      app_name=app_name
  )
  assert len(eval_set_results) == 2


def test_cli_create_eval_set(tmp_path: Path):
  app_name = "test_app"
  eval_set_id = "test_eval_set"
  agent_path = tmp_path / app_name
  agent_path.mkdir()
  (agent_path / "__init__.py").touch()

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      ["eval_set", "create", str(agent_path), eval_set_id],
  )

  assert result.exit_code == 0
  eval_set_file = agent_path / f"{eval_set_id}.evalset.json"
  assert eval_set_file.exists()
  with open(eval_set_file, "r") as f:
    eval_set_data = json.load(f)
  assert eval_set_data["eval_set_id"] == eval_set_id
  assert eval_set_data["eval_cases"] == []


def test_cli_add_eval_case_with_session(tmp_path: Path):
  app_name = "test_app_add_2"
  eval_set_id = "test_eval_set_add_2"
  agent_path = tmp_path / app_name
  agent_path.mkdir()
  (agent_path / "__init__.py").touch()

  scenarios_file = tmp_path / "scenarios2.json"
  scenarios_file.write_text(
      '{"scenarios": [{"starting_prompt": "hello", "conversation_plan":'
      ' "world"}]}'
  )
  session_file = tmp_path / "session2.json"
  session_file.write_text(
      '{"app_name": "test_app_add_2", "user_id": "test_user", "state": {}}'
  )

  runner = CliRunner()
  runner.invoke(
      cli_tools_click.main,
      ["eval_set", "create", str(agent_path), eval_set_id],
      catch_exceptions=False,
  )
  result = runner.invoke(
      cli_tools_click.main,
      [
          "eval_set",
          "add_eval_case",
          str(agent_path),
          eval_set_id,
          "--scenarios_file",
          str(scenarios_file),
          "--session_input_file",
          str(session_file),
      ],
      catch_exceptions=False,
  )

  assert result.exit_code == 0
  eval_set_file = agent_path / f"{eval_set_id}.evalset.json"
  assert eval_set_file.exists()
  with open(eval_set_file, "r") as f:
    eval_set_data = json.load(f)
  assert len(eval_set_data["eval_cases"]) == 1
  eval_case = eval_set_data["eval_cases"][0]
  assert eval_case["eval_id"] == "0a1a5048"
  assert eval_case["session_input"]["app_name"] == "test_app_add_2"


def test_cli_add_eval_case_skip_existing(tmp_path: Path):
  app_name = "test_app_add_3"
  eval_set_id = "test_eval_set_add_3"
  agent_path = tmp_path / app_name
  agent_path.mkdir()
  (agent_path / "__init__.py").touch()

  scenarios_file = tmp_path / "scenarios3.json"
  scenarios_file.write_text(
      '{"scenarios": [{"starting_prompt": "hello", "conversation_plan":'
      ' "world"}]}'
  )
  session_file = tmp_path / "session3.json"
  session_file.write_text(
      '{"app_name": "test_app_add_3", "user_id": "test_user", "state": {}}'
  )

  runner = CliRunner()
  runner.invoke(
      cli_tools_click.main,
      ["eval_set", "create", str(agent_path), eval_set_id],
      catch_exceptions=False,
  )
  result1 = runner.invoke(
      cli_tools_click.main,
      [
          "eval_set",
          "add_eval_case",
          str(agent_path),
          eval_set_id,
          "--scenarios_file",
          str(scenarios_file),
          "--session_input_file",
          str(session_file),
      ],
      catch_exceptions=False,
  )
  eval_set_file = agent_path / f"{eval_set_id}.evalset.json"
  with open(eval_set_file, "r") as f:
    eval_set_data1 = json.load(f)

  result2 = runner.invoke(
      cli_tools_click.main,
      [
          "eval_set",
          "add_eval_case",
          str(agent_path),
          eval_set_id,
          "--scenarios_file",
          str(scenarios_file),
          "--session_input_file",
          str(session_file),
      ],
      catch_exceptions=False,
  )
  with open(eval_set_file, "r") as f:
    eval_set_data2 = json.load(f)

  assert result1.exit_code == 0
  assert result2.exit_code == 0
  assert len(eval_set_data1["eval_cases"]) == 1
  assert len(eval_set_data2["eval_cases"]) == 1


def test_cli_deploy_cloud_run_gcloud_arg_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Extra gcloud args that conflict with ADK deploy args should raise ClickException."""

  def _mock_to_cloud_run(*_a, **kwargs):
    # Import and call the validation function
    from google.adk.cli.cli_deploy import _validate_gcloud_extra_args

    # Build the same set of managed args as the real function would
    adk_managed_args = {"--source", "--project", "--port", "--verbosity"}
    if kwargs.get("region"):
      adk_managed_args.add("--region")
    _validate_gcloud_extra_args(
        kwargs.get("extra_gcloud_args"), adk_managed_args
    )

  monkeypatch.setattr(
      cli_tools_click.cli_deploy, "to_cloud_run", _mock_to_cloud_run
  )

  agent_dir = tmp_path / "agent_conflict"
  agent_dir.mkdir()
  runner = CliRunner()

  # Test with conflicting --project arg
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "test-project",
          "--region",
          "us-central1",
          str(agent_dir),
          "--",
          "--project=conflict-project",  # This should conflict
      ],
  )

  expected_msg = (
      "The argument '--project' conflicts with ADK's automatic configuration."
      " ADK will set this argument automatically, so please remove it from your"
      " command."
  )
  assert expected_msg in result.output

  # Test with conflicting --port arg
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "test-project",
          str(agent_dir),
          "--",
          "--port=9000",  # This should conflict
      ],
  )

  expected_msg = (
      "The argument '--port' conflicts with ADK's automatic configuration. ADK"
      " will set this argument automatically, so please remove it from your"
      " command."
  )
  assert expected_msg in result.output

  # Test with conflicting --region arg
  result = runner.invoke(
      cli_tools_click.main,
      [
          "deploy",
          "cloud_run",
          "--project",
          "test-project",
          "--region",
          "us-central1",
          str(agent_dir),
          "--",
          "--region=us-west1",  # This should conflict
      ],
  )

  expected_msg = (
      "The argument '--region' conflicts with ADK's automatic configuration."
      " ADK will set this argument automatically, so please remove it from your"
      " command."
  )
  assert expected_msg in result.output
