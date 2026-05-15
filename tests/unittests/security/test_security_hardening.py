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

import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from google.adk.agents import config_agent_utils
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.cli.adk_web_server import RunAgentRequest
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.errors.input_validation_error import InputValidationError


class TestSecurityHardening:
  """Comprehensive security tests for Path Traversal, RCE, and Sanitization."""

  # --- Path Traversal: AgentLoader ---

  @pytest.mark.parametrize(
      "malicious_name",
      [
          "../evil",
          "..\\evil",
          "sub/../../evil",
          "valid/../..",
          "C:\\Windows\\System32\\cmd.exe",
          "/etc/passwd",
          ".",
          "..",
          " ",
          "\0",
          "%2e%2e/evil",
          "%5C..%5Cevil",
          "./../evil",
      ],
  )
  def test_agent_loader_prevents_path_traversal(self, malicious_name):
    """Verify AgentLoader rejects names that escape the agents directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
      loader = AgentLoader(temp_dir)
      with pytest.raises(ValueError) as exc_info:
        # _perform_load calls _validate_agent_name which uses a strict regex
        # for regular loading, and _validate_agent_path for YAML.
        loader.load_agent(malicious_name)
      
      # Should match either invalid name (regex) or path traversal (resolved path)
      msg = str(exc_info.value)
      assert any(
          s in msg for s in [
              "Invalid agent name",
              "resolves outside agents_dir",
              "Path traversal is not permitted",
              "Agent not found"
          ]
      )

  def test_agent_loader_yaml_validation_strict(self):
    """Verify _validate_agent_path directly with various traversal attempts."""
    with tempfile.TemporaryDirectory() as temp_dir:
      loader = AgentLoader(temp_dir)
      base_path = Path(temp_dir).resolve()
      
      # Mock the filesystem to avoid "Agent not found" errors during perform_load
      # if we were to call it directly. Instead, test the validator.
      
      # Malicious case
      with pytest.raises(ValueError, match="Path traversal is not permitted"):
        loader._validate_agent_path(str(base_path), "../outside")

      # Deeply nested traversal
      with pytest.raises(ValueError, match="Path traversal is not permitted"):
        loader._validate_agent_path(str(base_path), "a/b/../../../c")

      # Valid case
      loader._validate_agent_path(str(base_path), "valid_agent")
      loader._validate_agent_path(str(base_path), "subdir/agent")

  # --- RCE Mitigation: config_agent_utils ---

  @pytest.mark.parametrize(
      "malicious_module",
      [
          "os.system",
          "subprocess.run",
          "importlib.import_module",
          "google.adk_evil.module",
          "google.adk..evil",
          "google.adk. agents.LlmAgent", # space injection
          "google.adk.agents.LlmAgent; print('hacked')", # command separator
          "builtins.eval",
          "sys.modules",
          "__main__",
      ],
  )
  def test_config_agent_utils_rejects_unsafe_imports(self, malicious_module):
    """Verify only google.adk.* modules are allowed for dynamic resolution."""
    with pytest.raises(ValueError, match="outside the allowed namespace"):
      config_agent_utils.resolve_fully_qualified_name(malicious_module)

  def test_config_agent_utils_allows_adk_imports(self):
    """Verify ADK's own modules can still be resolved."""
    # This should not raise
    cls = config_agent_utils.resolve_fully_qualified_name("google.adk.agents.llm_agent.LlmAgent")
    from google.adk.agents.llm_agent import LlmAgent
    assert cls is LlmAgent

  def test_is_safe_module_import_logic(self):
    """Deep dive into the prefix matching logic to prevent partial name spoofs."""
    from google.adk.agents.config_agent_utils import _is_safe_module_import
    
    assert _is_safe_module_import("google.adk.agents") is True
    assert _is_safe_module_import("google.adk.utils") is True
    
    # Partial prefix spoofs
    assert _is_safe_module_import("google.adk_evil") is False
    assert _is_safe_module_import("google.adk_") is False
    
    # Empty segments
    assert _is_safe_module_import("google.adk..agents") is False
    assert _is_safe_module_import("google.adk.") is False # trailing dot splits to empty segment

  # --- API Boundary: RunAgentRequest ---

  @pytest.mark.parametrize(
      "bad_app_name",
      [
          "../evil",
          "\\path",
          "../../etc/passwd",
          "a/b",
          " ",
          "",
      ],
  )
  def test_run_agent_request_sanitization(self, bad_app_name):
    """Verify Pydantic model rejects malicious app_name at the API boundary."""
    with pytest.raises(ValidationError):
      RunAgentRequest(
          app_name=bad_app_name,
          user_id="user123",
          session_id="sess456"
      )

  def test_run_agent_request_valid(self):
    """Verify valid app names pass validation."""
    req = RunAgentRequest(
        app_name="my_secure_agent",
        user_id="user123",
        session_id="sess456"
    )
    assert req.app_name == "my_secure_agent"

  # --- Path Traversal: FileArtifactService ---

  @pytest.mark.parametrize("field_name", ["user_id", "session_id"])
  @pytest.mark.parametrize(
      "malicious_val",
      [
          "../escape",
          "..\\win",
          "sub/../../parent",
          ".",
          "..",
          "null\0byte",
          "",
      ],
  )
  async def test_artifact_service_prevents_traversal(self, tmp_path, field_name, malicious_val):
    """Verify FileArtifactService rejects traversal in user/session IDs."""
    service = FileArtifactService(root_dir=tmp_path / "artifacts")
    
    params = {
        "user_id": "valid_user",
        "session_id": "valid_sess",
        "app_name": "app",
        "filename": "file.txt",
        "artifact": mock.Mock() # mock Part
    }
    params[field_name] = malicious_val
    
    with pytest.raises(InputValidationError):
      await service.save_artifact(**params)

  # --- Symlink Awareness (Platform Dependent) ---
  
  def test_path_traversal_with_symlink_awareness(self, tmp_path):
    """Verify Path.resolve() correctly handles symlinks to prevent escaping agents_dir."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    (secret_dir / "passwords.txt").write_text("secret_data")
    
    # Create a symlink inside agents_dir pointing to sensitive data outside
    link_path = agents_dir / "malicious_link"
    try:
      os.symlink(secret_dir, link_path)
    except (OSError, NotImplementedError):
      pytest.skip("Symlinks not supported on this platform/environment")

    loader = AgentLoader(str(agents_dir))
    
    # Attempting to load "malicious_link/passwords.txt"
    # Even if it exists via symlink, the validator should eventually check it.
    # Note: AgentLoader checks if the directory exists and contains agent files.
    
    with pytest.raises(ValueError):
      # If the agent name follows identifiers only, "malicious_link" passes regex,
      # but if we try to use it to reach passwords.txt, it fails regex.
      # However, if an agent name is "malicious_link", it would load agents_dir/malicious_link/root_agent.yaml
      # if it existed. Path.resolve() would resolve it to secret_dir.
      loader._validate_agent_path(str(agents_dir), "malicious_link")
