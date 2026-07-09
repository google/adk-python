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

import json
import os
from pathlib import Path
import sys

from google.adk.agents import config_agent_utils
from google.adk.apps.app import App
from google.adk.cli.agent_test_runner import test_agent_replay as _test_agent_replay
from google.adk.cli.utils.agent_loader import AgentLoader
from google.genai import types
import pytest

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
  for test_file in CONTRIBUTING_DIR.rglob("tests/*.json"):
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
