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

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
from typing import AsyncIterator
from typing import Iterator

from google.adk.agents import config_agent_utils
from google.adk.apps.app import App
from google.adk.cli.agent_test_runner import test_agent_replay as _test_agent_replay
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.events import Event
from google.genai import types
import pytest
import requests

CONTRIBUTING_DIR = Path(__file__).parent.parent.parent / "contributing"
SAMPLES_DIR = CONTRIBUTING_DIR / "samples"


@pytest.fixture(autouse=True)
def _load_samples_like_adk_run():
  """Loads samples with the YAML key denylist off, matching `adk run`.

  The denylist is a hosted-web-server guard that fast_api enables globally and
  never resets, so a fast_api test earlier in the process would otherwise leave
  it on and block valid config samples here.
  """
  saved = config_agent_utils._ENFORCE_YAML_KEY_DENYLIST
  config_agent_utils._set_enforce_yaml_key_denylist(False)
  try:
    yield
  finally:
    config_agent_utils._set_enforce_yaml_key_denylist(saved)


def get_test_files():
  """Yields (sample_dir, test_file_path)."""
  if not CONTRIBUTING_DIR.exists():
    return
  # Sort files to ensure deterministic order across pytest-xdist workers
  test_files = sorted(
      CONTRIBUTING_DIR.rglob("tests/*.json"), key=lambda p: p.as_posix()
  )
  for test_file in test_files:
    sample_dir = test_file.parent.parent
    if (
        (sample_dir / "agent.py").exists()
        or (sample_dir / "__init__.py").exists()
        or (sample_dir / "root_agent.yaml").exists()
    ):
      try:
        rel_dir = sample_dir.relative_to(CONTRIBUTING_DIR)
        test_id = f"{rel_dir}/{test_file.name}"
      except ValueError:
        test_id = f"{sample_dir.name}/{test_file.name}"

      if test_file.stem.endswith("_xfail"):
        yield pytest.param(
            sample_dir, test_file, id=test_id, marks=pytest.mark.xfail
        )
      else:
        yield pytest.param(sample_dir, test_file, id=test_id)


@pytest.mark.parametrize(
    "sample_dir, test_file",
    list(get_test_files()),
)
def test_sample(sample_dir: Path, test_file: Path, monkeypatch):
  """Tests a sample by replaying exported session events."""
  _test_agent_replay(sample_dir, test_file, monkeypatch)


# Samples that cannot be loaded offline: they reach an external service, need an
# optional dependency outside [all], or are not an independently loadable root.
SKIP_LOAD = {
    "integrations/agent_registry_agent": "calls Agent Registry API at import",
    "integrations/api_registry_agent": "calls Cloud API Registry at import",
    "integrations/application_integration_agent": (
        "calls Integration Connectors API at import"
    ),
    "integrations/integration_connector_euc_agent": (
        "calls Integration Connectors API at import"
    ),
    "multimodal/static_non_text_content": (
        "uploads a file via the genai API at import"
    ),
    "integrations/authn-adk-all-in-one/adk_agents/agent_openapi_tools": (
        "needs a local identity provider server on :5000"
    ),
    "mcp/mcp_postgres_agent": (
        "needs POSTGRES_CONNECTION_STRING and a postgres server"
    ),
    "code_execution/custom_code_execution": (
        "provisions a Vertex code-interpreter extension at import"
    ),
    "code_execution/vertex_code_execution": (
        "provisions a Vertex code-interpreter extension at import"
    ),
    "integrations/crewai_tool_kwargs": (
        "needs the crewai package (not installed on every Python version)"
    ),
    "integrations/files_retrieval_agent": (
        "needs the llama-index-embeddings-google-genai package"
    ),
    "integrations/toolbox_agent": (
        "needs the toolbox-adk package and a toolbox server"
    ),
    "multimodal/computer_use": "needs the playwright package",
    "integrations/gepa": "experiment package, exposes no root_agent",
    "integrations/slack_agent": (
        "builds its agent inside main(), no module-level root_agent"
    ),
    "adk_team/adk_documentation": (
        "package dir; its child agents are the samples"
    ),
    "adk_team/adk_answering_agent/gemini_assistant": (
        "sub-agent of adk_answering_agent, not independently loadable"
    ),
    "integrations/oauth_calendar_agent": (
        "fetches the Calendar API (calendar v3) discovery doc at import"
    ),
}

# Samples whose own code is currently broken against the ADK API. Loading them
# fails today; remove the entry once the sample is fixed.
XFAIL_LOAD = {
    "integrations/jira_agent": (
        "ApplicationIntegrationToolset no longer accepts tool_name"
    ),
    "workflows/loop_config": (
        "root_agent.yaml references the nonexistent agent_class Workflow"
    ),
    "models/hello_world_litellm_add_function_to_prompt": (
        "langchain_core requires an explicit import of langchain_core.tools"
    ),
    "adk_team/adk_triaging_agent": (
        "agent.py imports adk_triaging_agent.settings, which is not present"
    ),
}

_DUMMY_ENV = {
    # Samples in this repo are trusted, so they are allowed to declare a stdio
    # MCP server in their agent config. Loading one does not start the server.
    "ADK_ALLOW_CONFIG_STDIO_MCP_SERVERS": "1",
    "GOOGLE_API_KEY": "dummy-key",
    "GEMINI_API_KEY": "dummy-key",
    "GOOGLE_CLOUD_PROJECT": "dummy-project",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "OPENAI_API_KEY": "dummy-key",
    "ANTHROPIC_API_KEY": "dummy-key",
    "GITHUB_TOKEN": "dummy-token",
    "VERTEXAI_DATASTORE_ID": "dummy-datastore",
}


def get_sample_dirs():
  """Yields a pytest param per loadable sample directory."""
  if not SAMPLES_DIR.exists():
    return
  sample_dirs = []
  for dirpath, dirnames, filenames in os.walk(SAMPLES_DIR):
    path = Path(dirpath)
    if path.name == "tests":
      dirnames[:] = []
      continue
    if any(
        f in filenames for f in ("agent.py", "__init__.py", "root_agent.yaml")
    ):
      sample_dirs.append(path)
  for sample_dir in sorted(sample_dirs):
    rel = sample_dir.relative_to(SAMPLES_DIR).as_posix()
    if rel in SKIP_LOAD:
      marks = pytest.mark.skip(reason=SKIP_LOAD[rel])
    elif rel in XFAIL_LOAD:
      marks = pytest.mark.xfail(reason=XFAIL_LOAD[rel], strict=False)
    else:
      marks = ()
    yield pytest.param(sample_dir, id=rel, marks=marks)


def _load_root_agent(sample_dir: Path):
  """Loads a sample the way `adk run` does, isolating module side effects."""
  saved_modules = set(sys.modules)
  saved_path = list(sys.path)
  sys.path.insert(0, str(sample_dir.parent))
  try:
    loader = AgentLoader(str(sample_dir.parent))
    loader.remove_agent_from_cache(sample_dir.name)
    agent_or_app = loader.load_agent(sample_dir.name)
    return (
        agent_or_app.root_agent
        if isinstance(agent_or_app, App)
        else agent_or_app
    )
  finally:
    sys.path[:] = saved_path
    # Evict only the sample's own modules so the next sample reloads cleanly,
    # while leaving third-party libraries cached (re-importing libraries with
    # global registries, e.g. opentelemetry, breaks on reload).
    prefix = sample_dir.name
    for name in set(sys.modules) - saved_modules:
      module = sys.modules.get(name)
      file = getattr(module, "__file__", None)
      if (
          name == prefix
          or name.startswith(prefix + ".")
          or (
              file
              and Path(file).resolve().is_relative_to(SAMPLES_DIR.resolve())
          )
      ):
        del sys.modules[name]


@pytest.mark.parametrize("sample_dir", list(get_sample_dirs()))
def test_sample_loads(sample_dir: Path, monkeypatch):
  """Smoke test: every sample's agent imports and constructs a root agent."""
  for key, value in _DUMMY_ENV.items():
    monkeypatch.setenv(key, value)
  import google.auth
  import google.auth.credentials
  import google.auth.transport
  from google.auth.transport import mtls

  class _DummyCredentials(google.auth.credentials.Credentials):

    def __init__(self) -> None:
      super().__init__()
      self.token: str | None = "dummy-token"

    def refresh(self, request: google.auth.transport.Request) -> None:
      self.token = "dummy-token"

  monkeypatch.setattr(
      google.auth,
      "default",
      lambda *args, **kwargs: (
          _DummyCredentials(),
          "dummy-project",
      ),
  )
  monkeypatch.setattr(
      mtls,
      "has_default_client_cert_source",
      lambda: False,
  )
  root_agent = _load_root_agent(sample_dir)
  assert root_agent is not None, f"{sample_dir} loaded no root agent"
  assert getattr(
      root_agent, "name", None
  ), f"{sample_dir} root agent has no name"


@contextlib.contextmanager
def _sample_module(sample_dir: Path, module_name: str) -> Iterator[Any]:
  """Imports one module of a sample package and evicts it afterwards."""
  prefix = sample_dir.name
  saved_path = list(sys.path)
  sys.path.insert(0, str(sample_dir.parent))
  try:
    yield importlib.import_module(f"{prefix}.{module_name}")
  finally:
    sys.path[:] = saved_path
    for name in list(sys.modules):
      if name == prefix or name.startswith(prefix + "."):
        del sys.modules[name]


@pytest.mark.parametrize(
    "failing_issue, expected_exit_code", [(None, 0), (2, 1)]
)
async def test_stale_agent_reports_failed_audits(
    failing_issue: int | None,
    expected_exit_code: int,
    monkeypatch,
    caplog,
):
  """The stale agent must not count a failed audit as processed."""
  for key, value in _DUMMY_ENV.items():
    monkeypatch.setenv(key, value)

  issues = [1, 2, 3]
  audited: list[int] = []

  class _FakeSession:
    id = "fake-session"

  class _FakeSessionService:

    async def create_session(
        self, *, user_id: str, app_name: str
    ) -> _FakeSession:
      return _FakeSession()

  class _FakeRunner:
    """Stands in for InMemoryRunner, failing the audit of one issue."""

    def __init__(self, *, agent: Any, app_name: str) -> None:
      self.session_service = _FakeSessionService()

    async def run_async(
        self, *, user_id: str, session_id: str, new_message: types.Content
    ) -> AsyncIterator[Event]:
      issue_number = int(new_message.parts[0].text.split("#")[1].rstrip("."))
      audited.append(issue_number)
      if issue_number == failing_issue:
        raise RuntimeError("model backend unavailable")
      yield Event(
          author="agent",
          content=types.Content(
              role="model", parts=[types.Part(text="No action needed.")]
          ),
      )

  with _sample_module(
      SAMPLES_DIR / "adk_team" / "adk_stale_agent", "main"
  ) as main_module:
    monkeypatch.setattr(main_module, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(main_module, "SLEEP_BETWEEN_CHUNKS", 0)
    monkeypatch.setattr(
        main_module,
        "get_old_open_issue_numbers",
        lambda owner, repo, days_old=None: list(issues),
    )
    with caplog.at_level(logging.INFO, logger="google_adk"):
      exit_code = await main_module.main()

  assert exit_code == expected_exit_code
  # Every issue is still audited: one failure must not abort the batch.
  assert sorted(audited) == issues
  expected_successes = 3 if failing_issue is None else 2
  assert f"Successfully processed {expected_successes} issues." in caplog.text
  assert ("Failed to process 1 issues." in caplog.text) == (
      failing_issue is not None
  )


@pytest.mark.parametrize(
    "failing_issue, expected_exit_code", [(None, 0), (4, 1)]
)
async def test_issue_monitoring_agent_separates_skips_from_failures(
    failing_issue: int | None,
    expected_exit_code: int,
    monkeypatch,
    caplog,
):
  """A skipped issue is not a failure, and a failed one is not a success."""
  for key, value in _DUMMY_ENV.items():
    monkeypatch.setenv(key, value)

  reviewed: list[int] = []

  class _FakeSession:
    id = "fake-session"

  class _FakeSessionService:

    async def create_session(
        self, *, user_id: str, app_name: str
    ) -> _FakeSession:
      return _FakeSession()

  class _FakeRunner:
    """Stands in for InMemoryRunner, failing the audit of one issue."""

    def __init__(self, *, agent: Any, app_name: str) -> None:
      self.session_service = _FakeSessionService()

    async def run_async(
        self, *, user_id: str, session_id: str, new_message: types.Content
    ) -> AsyncIterator[Event]:
      text = new_message.parts[0].text
      issue_number = int(text.split("#")[1].split(":")[0])
      reviewed.append(issue_number)
      if issue_number == failing_issue:
        raise RuntimeError("model backend unavailable")
      yield Event(
          author="agent",
          content=types.Content(
              role="model", parts=[types.Part(text="Not spam.")]
          ),
      )

  with _sample_module(
      SAMPLES_DIR / "adk_team" / "adk_issue_monitoring_agent", "main"
  ) as main_module:
    # 1 and 4 are audited, 2 is skipped because the bot already alerted on it,
    # 3 is skipped because only a maintainer has written on it.
    details = {
        n: {"user": {"login": "maintainer"}, "body": "tracking"} for n in (2, 3)
    }
    details[1] = {"user": {"login": "outsider"}, "body": "buy things"}
    details[4] = {"user": {"login": "outsider"}, "body": "buy more things"}
    comments = {
        2: [{
            "user": {"login": main_module.BOT_NAME},
            "body": main_module.BOT_ALERT_SIGNATURE,
        }],
        3: [{"user": {"login": "maintainer"}, "body": "still looking"}],
    }

    monkeypatch.setattr(main_module, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(main_module, "SLEEP_BETWEEN_CHUNKS", 0)
    monkeypatch.setattr(
        main_module,
        "get_repository_maintainers",
        lambda owner, repo: ["maintainer"],
    )
    monkeypatch.setattr(
        main_module, "get_target_issues", lambda owner, repo: [1, 2, 3, 4]
    )
    monkeypatch.setattr(
        main_module,
        "get_issue_details",
        lambda owner, repo, issue_number: details[issue_number],
    )
    monkeypatch.setattr(
        main_module,
        "get_issue_comments",
        lambda owner, repo, issue_number: comments.get(issue_number, []),
    )
    with caplog.at_level(logging.INFO, logger="google_adk"):
      exit_code = await main_module.main()

  assert exit_code == expected_exit_code
  # Every reviewable issue still reaches the agent: one failure must not abort
  # the batch.
  assert sorted(reviewed) == [1, 4]
  expected_successes = 2 if failing_issue is None else 1
  assert f"Successfully processed {expected_successes} issues." in caplog.text
  assert "Skipped 2 issues." in caplog.text
  assert ("Failed to process 1 issues." in caplog.text) == (
      failing_issue is not None
  )


@pytest.mark.parametrize(
    "sample_name, discovery_name",
    [
        ("adk_stale_agent", "get_old_open_issue_numbers"),
        ("adk_issue_monitoring_agent", "get_target_issues"),
    ],
)
def test_issue_discovery_propagates_page_failure(
    sample_name: str, discovery_name: str, monkeypatch
):
  """A search that dies mid-pagination must not look like a short result."""
  for key, value in _DUMMY_ENV.items():
    monkeypatch.setenv(key, value)

  full_page = [{"number": n} for n in range(1, 101)]

  def _get_request(url: str, params: dict[str, Any] | None = None) -> Any:
    if (params or {}).get("page", 1) > 1:
      raise requests.exceptions.ConnectionError("connection reset")
    return {"items": full_page} if "search" in url else full_page

  with _sample_module(
      SAMPLES_DIR / "adk_team" / sample_name, "utils"
  ) as utils_module:
    monkeypatch.setattr(utils_module, "get_request", _get_request)
    with pytest.raises(requests.exceptions.ConnectionError):
      getattr(utils_module, discovery_name)("google", "adk-python")
