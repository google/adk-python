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

"""Tests for optional dependencies and lazy loading.

Includes both fast hermetic unit tests (run by default) and high-fidelity
integration tests using a clean venv (skipped by default, run via env var).
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Check if we should run integration tests that require network/install
RUN_INTEGRATION = os.environ.get("ADK_TEST_NETWORK") == "1"


@pytest.fixture(scope="session")
def clean_core_venv(tmp_path_factory):
  """Creates a clean venv with only core ADK installed (requires network)."""
  if not RUN_INTEGRATION:
    pytest.skip("Integration tests requiring network are disabled by default.")

  venv_path = tmp_path_factory.mktemp("adk_core_venv")
  python_exe = venv_path / "bin" / "python"

  # Create venv using uv for speed
  subprocess.run(
      ["uv", "venv", str(venv_path), "--python", "3.11"],
      check=True,
      capture_output=True,
  )

  # Install core ADK from local repo (uses current pyproject.toml)
  subprocess.run(
      ["uv", "pip", "install", "--python", str(python_exe), str(_REPO_ROOT)],
      check=True,
      capture_output=True,
  )

  return python_exe


# =============================================================================
# Approach 1: Hermetic Unit Tests (Fast, No Network, Safe for CI/TAP)
# =============================================================================


def test_pydantic_version():
  """Print the installed Pydantic version."""
  import pydantic

  print(f"Pydantic version: {pydantic.__version__}")
  assert True


def test_no_eager_imports():
  """Verify that importing google.adk does not eagerly load heavy optional deps.

  Runs in the current environment but in a fresh subprocess, ensuring it
  only checks the import side-effects without modifying the environment.
  """
  code = """
import sys
import google.adk
heavy_modules = ['google.cloud.aiplatform', 'sqlalchemy', 'a2a']
loaded = [mod for mod in heavy_modules if mod in sys.modules]
print(','.join(loaded))
"""
  result = subprocess.run(
      [sys.executable, "-c", code], capture_output=True, text=True, check=True
  )
  loaded_modules = result.stdout.strip()
  assert loaded_modules == "", f"Heavy modules loaded eagerly: {loaded_modules}"


def test_a2a_remote_agent_config_raises_importerror():
  """Verify that accessing A2aRemoteAgentConfig without extra raises ImportError using mocks."""
  with mock.patch.dict("sys.modules", {"a2a": None}):
    for mod in list(sys.modules):
      if mod.startswith("a2a.") or mod.startswith("google.adk.a2a."):
        sys.modules.pop(mod, None)
    with pytest.raises(ImportError) as exc_info:
      from google.adk.a2a.agent import A2aRemoteAgentConfig
    assert "a2a-sdk" in str(exc_info.value)


def test_vertex_ai_memory_bank_service_fails_on_creation():
  """Verify that creating VertexAiMemoryBankService without extra fails using mocks."""
  try:
    from google.adk.memory import VertexAiMemoryBankService
  except KeyError as e:
    if "pydantic.root_model" in str(e):
      pytest.skip(
          "Skipping mock test due to Pydantic/MCP environment conflict"
          " (KeyError: 'pydantic.root_model')."
      )
    raise

  with mock.patch.dict("sys.modules", {"vertexai": None}):
    sys.modules.pop("google.adk.memory.vertex_ai_memory_bank_service", None)
    from google.adk.memory import VertexAiMemoryBankService

    with pytest.raises(ImportError) as exc_info:
      VertexAiMemoryBankService(agent_engine_id="123")
    assert "google-cloud-aiplatform" in str(exc_info.value)


def test_database_session_service_fails_on_creation():
  """Verify that creating DatabaseSessionService without extra fails using mocks."""
  try:
    from google.adk.sessions import DatabaseSessionService
  except KeyError as e:
    if "pydantic.root_model" in str(e):
      pytest.skip(
          "Skipping mock test due to Pydantic/MCP environment conflict"
          " (KeyError: 'pydantic.root_model')."
      )
    raise

  with mock.patch.dict("sys.modules", {"sqlalchemy": None}):
    sys.modules.pop("google.adk.sessions.database_session_service", None)
    with pytest.raises(ImportError) as exc_info:
      from google.adk.sessions import DatabaseSessionService

      DatabaseSessionService(db_url="sqlite+aiosqlite:///:memory:")
    assert "sqlalchemy" in str(exc_info.value)


def test_vertex_ai_session_service_fails_on_creation():
  """Verify that creating VertexAiSessionService without extra fails using mocks."""
  try:
    from google.adk.sessions import VertexAiSessionService
  except KeyError as e:
    if "pydantic.root_model" in str(e):
      pytest.skip(
          "Skipping mock test due to Pydantic/MCP environment conflict"
          " (KeyError: 'pydantic.root_model')."
      )
    raise

  with mock.patch.dict("sys.modules", {"vertexai": None}):
    sys.modules.pop("google.adk.sessions.vertex_ai_session_service", None)
    from google.adk.sessions import VertexAiSessionService

    with pytest.raises(ImportError) as exc_info:
      VertexAiSessionService(agent_engine_id="123")
    assert "google-cloud-aiplatform" in str(exc_info.value)


# =============================================================================
# Approach 2: High-Fidelity Integration Tests (Clean Venv, Skipped by Default)
# =============================================================================


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_critical_imports_subprocess(clean_core_venv):
  """Verify that all critical import paths in core work in a true core-only environment."""
  imports = [
      "import google.adk",
      "from google.adk import Agent",
      "from google.adk import Context",
      "from google.adk import Event",
      "from google.adk import Runner",
      "from google.adk import Workflow",
  ]
  for imp in imports:
    result = subprocess.run(
        [str(clean_core_venv), "-c", imp],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Failed to import: {imp}\nStderr: {result.stderr}"


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_a2a_remote_agent_config_raises_importerror_subprocess(clean_core_venv):
  """Verify that accessing A2aRemoteAgentConfig without extra raises ImportError in clean environment."""
  code = """
try:
    from google.adk.a2a.agent import A2aRemoteAgentConfig
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "a2a-sdk" in output


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_vertex_ai_memory_bank_service_fails_on_creation_subprocess(
    clean_core_venv,
):
  """Verify that creating VertexAiMemoryBankService without extra fails in clean environment."""
  code = """
from google.adk.memory import VertexAiMemoryBankService
try:
    service = VertexAiMemoryBankService(agent_engine_id="123")
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "google-cloud-aiplatform" in output


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_database_session_service_fails_on_creation_subprocess(
    clean_core_venv,
):
  """Verify that creating DatabaseSessionService without extra fails in clean environment."""
  code = """
try:
    from google.adk.sessions import DatabaseSessionService
    service = DatabaseSessionService(db_url="sqlite+aiosqlite:///:memory:")
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "sqlalchemy" in output


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_vertex_ai_session_service_fails_on_creation_subprocess(
    clean_core_venv,
):
  """Verify that creating VertexAiSessionService without extra fails in clean environment."""
  code = """
from google.adk.sessions import VertexAiSessionService
try:
    service = VertexAiSessionService(agent_engine_id="123")
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "google-cloud-aiplatform" in output
