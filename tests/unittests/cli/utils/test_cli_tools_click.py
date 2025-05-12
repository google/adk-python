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
import click
import pytest

from google.adk.cli import cli_tools_click

from pathlib import Path
from typing import Any, Dict, List, Tuple
from types import SimpleNamespace
from click.testing import CliRunner

# Helpers
class _Recorder:
    """Callable that records every invocation."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

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
    assert cli_tools_click.validate_exclusive(ctx, param, "file.json") == "file.json"


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
def test_cli_create_cmd_invokes_run_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
async def test_cli_run_invokes_run_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`adk run` should call run_cli via asyncio.run with correct parameters."""
    rec = _Recorder()
    monkeypatch.setattr(cli_tools_click, "run_cli", lambda **kwargs: rec(kwargs))
    monkeypatch.setattr(cli_tools_click.asyncio, "run", lambda coro: coro)  # pass-through

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
def test_cli_deploy_cloud_run_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_cli_deploy_cloud_run_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = runner.invoke(cli_tools_click.main, ["deploy", "cloud_run", str(agent_dir)])

    assert result.exit_code == 0
    assert any("Deploy failed: boom" in m for m in captured)


# cli eval
def test_cli_eval_missing_deps_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        def __init__(self, *a: Any, **k: Any) -> None: ...
        def run(self) -> None:
            rec()

    monkeypatch.setattr(cli_tools_click.uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(cli_tools_click.uvicorn, "Server", lambda *_a, **_k: _DummyServer())
    monkeypatch.setattr(cli_tools_click, "get_fast_api_app", lambda **_k: object())
    return rec


def test_cli_web_invokes_uvicorn(tmp_path: Path, _patch_uvicorn: _Recorder) -> None:
    """`adk web` should configure and start uvicorn.Server.run."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli_tools_click.main, ["web", str(agents_dir)])
    assert result.exit_code == 0
    assert _patch_uvicorn.calls, "uvicorn.Server.run must be called"


def test_cli_api_server_invokes_uvicorn(tmp_path: Path, _patch_uvicorn: _Recorder) -> None:
    """`adk api_server` should configure and start uvicorn.Server.run."""
    agents_dir = tmp_path / "agents_api"
    agents_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(cli_tools_click.main, ["api_server", str(agents_dir)])
    assert result.exit_code == 0
    assert _patch_uvicorn.calls, "uvicorn.Server.run must be called"


def test_cli_eval_success_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the success path of `adk eval` by fully executing it with a stub module, up to summary generation."""
    import sys, types

    # stub cli_eval module
    stub = types.ModuleType("google.adk.cli.cli_eval")

    class _EvalMetric:
        def __init__(self, metric_name: str, threshold: float) -> None: ...

    class _EvalResult:
        def __init__(self, eval_set_file: str, final_eval_status: str) -> None:
            self.eval_set_file = eval_set_file
            self.final_eval_status = final_eval_status

    # minimal enum-like namespace
    _EvalStatus = types.SimpleNamespace(PASSED="PASSED", FAILED="FAILED")

    # helper funcs
    stub.EvalMetric = _EvalMetric
    stub.EvalResult = _EvalResult
    stub.EvalStatus = _EvalStatus
    stub.MISSING_EVAL_DEPENDENCIES_MESSAGE = "stub msg"

    stub.get_evaluation_criteria_or_default = lambda _p: {"foo": 1.0}
    stub.get_root_agent = lambda _p: object()
    stub.try_get_reset_func = lambda _p: None
    stub.parse_and_get_evals_to_run = lambda _paths: {"set1.json": ["e1", "e2"]}
    stub.run_evals = lambda *_a, **_k: iter(
        [_EvalResult("set1.json", "PASSED"), _EvalResult("set1.json", "FAILED")]
    )

    monkeypatch.setattr(cli_tools_click.asyncio, "run", lambda coro: list(coro))

    # inject stub
    sys.modules["google.adk.cli.cli_eval"] = stub

    # create dummy agent directory
    agent_dir = tmp_path / "agent5"
    agent_dir.mkdir()
    (agent_dir / "__init__.py").touch()

    # inject monkeypatch
    monkeypatch.setattr(cli_tools_click.envs, "load_dotenv_for_agent", lambda *a, **k: None)

    runner = CliRunner()
    result = runner.invoke(
        cli_tools_click.main,
        ["eval", str(agent_dir), str(tmp_path / "dummy_eval.json")],
    )

    assert result.exit_code == 0
    assert "Eval Run Summary" in result.output
    assert "Tests passed: 1" in result.output
    assert "Tests failed: 1" in result.output
