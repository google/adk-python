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

import pytest

from google.adk.agents.config_agent_utils import check_config_for_blocked_keys


def test_check_config_for_blocked_keys_rejects_args_key():
  config = {
      "name": "test_agent",
      "model": "gemini-2.0-flash",
      "tools": [{
          "name": "some_project.tools.create_tool",
          "args": [],
      }],
  }

  with pytest.raises(ValueError, match="Blocked key 'args'"):
    check_config_for_blocked_keys(config, "root_agent.yaml")


def test_check_config_for_blocked_keys_rejects_blocked_tool_module():
  config = {
      "name": "test_agent",
      "model": "gemini-2.0-flash",
      "tools": [{
          "name": "subprocess.run",
      }],
  }

  with pytest.raises(
      ValueError, match="Blocked code reference 'subprocess.run'"
  ):
    check_config_for_blocked_keys(config, "root_agent.yaml")


def test_check_config_for_blocked_keys_allows_non_blocked_tool_reference():
  config = {
      "name": "test_agent",
      "model": "gemini-2.0-flash",
      "tools": [{
          "name": "my_project.tools.echo",
      }],
  }

  check_config_for_blocked_keys(config, "root_agent.yaml")
