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

from unittest import mock

from google.adk.cli.utils import agent_loader
from google.adk.cli.utils.agent_change_handler import AgentChangeEventHandler
from google.adk.cli.utils.shared_value import SharedValue
import pytest
from watchdog.events import FileModifiedEvent


class TestAgentChangeEventHandler:
  """Unit tests for AgentChangeEventHandler file extension filtering."""

  @pytest.fixture
  def mock_agent_loader(self):
    """Create a mock AgentLoader constrained to the public API."""
    return mock.create_autospec(
        agent_loader.AgentLoader, instance=True, spec_set=True
    )

  @pytest.fixture
  def handler(self, mock_agent_loader):
    """Create an AgentChangeEventHandler with mocked dependencies."""
    runners_to_clean = set()
    current_app_name_ref = SharedValue(value="test_agent")
    return AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        current_app_name_ref=current_app_name_ref,
    )

  @pytest.mark.parametrize(
      "file_path",
      [
          pytest.param("/path/to/agent.py", id="python_file"),
          pytest.param("/path/to/config.yaml", id="yaml_file"),
          pytest.param("/path/to/config.yml", id="yml_file"),
      ],
  )
  def test_on_modified_triggers_reload_for_supported_extensions(
      self, handler, mock_agent_loader, file_path
  ):
    """Verify that .py, .yaml, and .yml files trigger agent reload."""
    event = FileModifiedEvent(src_path=file_path)

    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_called_once_with(
        "test_agent"
    )
    assert (
        "test_agent" in handler.runners_to_clean
    ), f"Expected 'test_agent' in runners_to_clean for {file_path}"

  @pytest.mark.parametrize(
      "file_path",
      [
          pytest.param("/path/to/file.json", id="json_file"),
          pytest.param("/path/to/file.txt", id="txt_file"),
          pytest.param("/path/to/file.md", id="markdown_file"),
          pytest.param("/path/to/file.toml", id="toml_file"),
          pytest.param("/path/to/.gitignore", id="gitignore_file"),
          pytest.param("/path/to/file", id="no_extension"),
      ],
  )
  def test_on_modified_ignores_unsupported_extensions(
      self, handler, mock_agent_loader, file_path
  ):
    """Verify that non-py/yaml/yml files do not trigger reload."""
    event = FileModifiedEvent(src_path=file_path)

    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_not_called()
    assert not handler.runners_to_clean, (
        f"Expected runners_to_clean to be empty for {file_path}, "
        f"got {handler.runners_to_clean}"
    )
