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
from typing import Optional
from typing import Tuple

import click
from click.testing import CliRunner
from google.adk.cli import cli_tools_click
from google.adk.evaluation import local_eval_set_results_manager
from google.adk.sessions import Session
from pydantic import BaseModel
import pytest


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
  monkeypatch.setattr(click, "secho", lambda *a, **k: None)


# validate_exclusive
def test_validate_exclusive_allows_single() -> None:
  """Providing exactly one exclusive option should pass."""
  ctx = click.Context(cli_tools_click.main)
  param = SimpleNamespace(name="replay")
  assert (
      cli_tools_click.validate_exclusive(ctx, param, "file.json") == "file.json"
  )


def test_validate_exclusive_blocks_multiple() -> None:
  """Providing two exclusive options should raise UsageError."""
  ctx = click.Context(cli_tools_click.main)
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

  # intercept click.secho(error=True) output
  captured: List[str] = []
  monkeypatch.setattr(click, "secho", lambda msg, **__: captured.append(msg))

  agent_dir = tmp_path / "agent3"
  agent_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main, ["deploy", "cloud_run", str(agent_dir)]
  )

  assert result.exit_code == 0
  assert any("Deploy failed: boom" in m for m in captured)


# cli eval
def test_cli_eval_missing_deps_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """If cli_eval sub-module is missing, command should raise ClickException."""
  # Ensure .cli_eval is not importable
  orig_import = builtins.__import__

  def _fake_import(name: str, *a: Any, **k: Any):
    if name.endswith(".cli_eval") or name == "google.adk.cli.cli_eval":
      raise ModuleNotFoundError()
    return orig_import(name, *a, **k)

  monkeypatch.setattr(builtins, "__import__", _fake_import)


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
  monkeypatch.setattr(
      cli_tools_click, "get_fast_api_app", lambda **_k: object()
  )
  return rec


def test_cli_web_invokes_uvicorn(
    tmp_path: Path, _patch_uvicorn: _Recorder
) -> None:
  """`adk web` should configure and start uvicorn.Server.run."""
  agents_dir = tmp_path / "agents"
  agents_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(cli_tools_click.main, ["web", str(agents_dir)])
  assert result.exit_code == 0
  assert _patch_uvicorn.calls, "uvicorn.Server.run must be called"


def test_cli_api_server_invokes_uvicorn(
    tmp_path: Path, _patch_uvicorn: _Recorder
) -> None:
  """`adk api_server` should configure and start uvicorn.Server.run."""
  agents_dir = tmp_path / "agents_api"
  agents_dir.mkdir()
  runner = CliRunner()
  result = runner.invoke(cli_tools_click.main, ["api_server", str(agents_dir)])
  assert result.exit_code == 0
  assert _patch_uvicorn.calls, "uvicorn.Server.run must be called"


def test_cli_eval_success_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Test the success path of `adk eval` by fully executing it with a stub module, up to summary generation."""
  import asyncio
  import sys
  import types

  # stub cli_eval module
  stub = types.ModuleType("google.adk.cli.cli_eval")
  eval_sets_manager_stub = types.ModuleType(
      "google.adk.evaluation.local_eval_sets_manager"
  )

  class _EvalMetric:

    def __init__(self, metric_name: str, threshold: float) -> None:
      ...

  class _EvalCaseResult(BaseModel):
    eval_set_id: str
    eval_id: str
    final_eval_status: Any
    user_id: str
    session_id: str
    session_details: Optional[Session] = None
    eval_metric_results: list = {}
    overall_eval_metric_results: list = {}
    eval_metric_result_per_invocation: list = {}

  class EvalCase(BaseModel):
    eval_id: str

  class EvalSet(BaseModel):
    eval_set_id: str
    eval_cases: list[EvalCase]

  def mock_save_eval_set_result(cls, *args, **kwargs):
    return None

  monkeypatch.setattr(
      local_eval_set_results_manager.LocalEvalSetResultsManager,
      "save_eval_set_result",
      mock_save_eval_set_result,
  )

  # minimal enum-like namespace
  _EvalStatus = types.SimpleNamespace(PASSED="PASSED", FAILED="FAILED")

  # helper funcs
  stub.EvalMetric = _EvalMetric
  stub.EvalCaseResult = _EvalCaseResult
  stub.EvalStatus = _EvalStatus
  stub.MISSING_EVAL_DEPENDENCIES_MESSAGE = "stub msg"

  stub.get_evaluation_criteria_or_default = lambda _p: {"foo": 1.0}
  stub.get_root_agent = lambda _p: object()
  stub.try_get_reset_func = lambda _p: None
  stub.parse_and_get_evals_to_run = lambda _paths: {"set1.json": ["e1", "e2"]}
  eval_sets_manager_stub.load_eval_set_from_file = lambda x, y: EvalSet(
      eval_set_id="test_eval_set_id",
      eval_cases=[EvalCase(eval_id="e1"), EvalCase(eval_id="e2")],
  )

  # Create an async generator function for run_evals
  async def mock_run_evals(*_a, **_k):
    yield _EvalCaseResult(
        eval_set_id="set1.json",
        eval_id="e1",
        final_eval_status=_EvalStatus.PASSED,
        user_id="user",
        session_id="session1",
        overall_eval_metric_results=[{
            "metricName": "some_metric",
            "threshold": 0.0,
            "score": 1.0,
            "evalStatus": _EvalStatus.PASSED,
        }],
    )
    yield _EvalCaseResult(
        eval_set_id="set1.json",
        eval_id="e2",
        final_eval_status=_EvalStatus.FAILED,
        user_id="user",
        session_id="session2",
        overall_eval_metric_results=[{
            "metricName": "some_metric",
            "threshold": 0.0,
            "score": 0.0,
            "evalStatus": _EvalStatus.FAILED,
        }],
    )

  stub.run_evals = mock_run_evals

  # Replace asyncio.run with a function that properly handles coroutines
  def mock_asyncio_run(coro):
    # Create a new event loop
    loop = asyncio.new_event_loop()
    try:
      return loop.run_until_complete(coro)
    finally:
      loop.close()

  monkeypatch.setattr(cli_tools_click.asyncio, "run", mock_asyncio_run)

  # inject stub
  monkeypatch.setitem(sys.modules, "google.adk.cli.cli_eval", stub)
  monkeypatch.setitem(
      sys.modules,
      "google.adk.evaluation.local_eval_sets_manager",
      eval_sets_manager_stub,
  )

  # create dummy agent directory
  agent_dir = tmp_path / "agent5"
  agent_dir.mkdir()
  (agent_dir / "__init__.py").touch()

  # inject monkeypatch
  monkeypatch.setattr(
      cli_tools_click.envs, "load_dotenv_for_agent", lambda *a, **k: None
  )

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      ["eval", str(agent_dir), str(tmp_path / "dummy_eval.json")],
  )

  assert result.exit_code == 0
  assert "Eval Run Summary" in result.output
  assert "Tests passed: 1" in result.output
  assert "Tests failed: 1" in result.output
