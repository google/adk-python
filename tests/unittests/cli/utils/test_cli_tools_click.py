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

"""Tests for utilities in cli_tool_click."""


from __future__ import annotations

import builtins
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
def _mute_click(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppress click output during tests."""
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
  assert result.exit_code == 0
  assert rec.calls, "cli_create.run_cmd must be called"


# cli run
@pytest.mark.asyncio
async def test_cli_run_invokes_run_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """`adk run` should call run_cli via asyncio.run with correct parameters."""
  rec = _Recorder()
  monkeypatch.setattr(cli_tools_click, "run_cli", lambda **kwargs: rec(kwargs))
  monkeypatch.setattr(
      cli_tools_click.asyncio, "run", lambda coro: coro
  )  # pass-through

  # create dummy agent directory
  agent_dir = tmp_path / "agent"
  agent_dir.mkdir()
  (agent_dir / "__init__.py").touch()
  (agent_dir / "agent.py").touch()

  runner = CliRunner()
  result = runner.invoke(cli_tools_click.main, ["run", str(agent_dir)])
  assert result.exit_code == 0
  assert rec.calls and rec.calls[0][0][0]["agent_folder_name"] == "agent"


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
          "--staging_bucket",
          "gs://mybucket",
          str(agent_dir),
      ],
  )
  assert result.exit_code == 0
  assert rec.calls, "cli_deploy.to_agent_engine must be invoked"
  called_kwargs = rec.calls[0][1]
  assert called_kwargs.get("project") == "test-proj"
  assert called_kwargs.get("region") == "us-central1"
  assert called_kwargs.get("staging_bucket") == "gs://mybucket"


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


def test_cli_web_passes_deprecated_uris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_uvicorn: _Recorder
) -> None:
  """`adk web` should use deprecated URIs if new ones are not provided."""
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
  assert mock_get_app.calls
  called_kwargs = mock_get_app.calls[0][1]
  assert called_kwargs.get("session_service_uri") == "sqlite:///deprecated.db"
  assert called_kwargs.get("artifact_service_uri") == "gs://deprecated"


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
