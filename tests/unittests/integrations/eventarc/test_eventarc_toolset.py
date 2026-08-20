# mypy: ignore-errors
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


import unittest
from unittest import mock
import warnings

from google.adk.features._feature_registry import _WARNED_FEATURES
from google.adk.integrations.eventarc import _eventarc_toolset as eventarc_toolset
from google.adk.integrations.eventarc import EventarcCredentialsConfig
from google.adk.integrations.eventarc import EventarcToolConfig
from google.adk.integrations.eventarc import EventarcToolset
import google.oauth2.credentials


class TestEventarcToolset(unittest.IsolatedAsyncioTestCase):

  def test_initializes_with_defaults(self):
    toolset = EventarcToolset(
        credentials_config=EventarcCredentialsConfig(
            credentials=google.oauth2.credentials.Credentials(token="fake")
        )
    )
    self.assertIsInstance(toolset.tool_config, EventarcToolConfig)
    self.assertIsInstance(toolset.credentials_config, EventarcCredentialsConfig)
    self.assertEqual(len(toolset._tools), 1)
    self.assertEqual(toolset._publish_message_tool.name, "publish_message")

  def test_initializes_with_explicit_configs(self):
    tool_config = EventarcToolConfig(project_id="test-project")
    credentials_config = EventarcCredentialsConfig(
        credentials=google.oauth2.credentials.Credentials(token="fake")
    )
    toolset = EventarcToolset(
        tool_config=tool_config, credentials_config=credentials_config
    )
    self.assertEqual(toolset.tool_config.project_id, "test-project")
    self.assertIs(toolset.credentials_config, credentials_config)

  async def test_get_tools_returns_publish_message(self):
    toolset = EventarcToolset(
        credentials_config=EventarcCredentialsConfig(
            credentials=google.oauth2.credentials.Credentials(token="fake")
        )
    )
    tools = await toolset.get_tools()
    self.assertEqual(len(tools), 1)
    self.assertEqual(tools[0].name, "publish_message")

  @mock.patch.object(eventarc_toolset, "eventarc_client", autospec=True)
  async def test_close_cleans_up_clients(self, mock_client):
    toolset = EventarcToolset(
        credentials_config=EventarcCredentialsConfig(
            credentials=google.oauth2.credentials.Credentials(token="fake")
        )
    )
    mock_client.cleanup_clients = mock.AsyncMock()
    await toolset.close()
    mock_client.cleanup_clients.assert_called_once()

  def test_eventarc_toolset_experimental_warning(self):
    _WARNED_FEATURES.clear()
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      EventarcToolset(
          credentials_config=EventarcCredentialsConfig(
              credentials=google.oauth2.credentials.Credentials(token="fake")
          )
      )
      self.assertTrue(
          any("EVENTARC_TOOLSET is enabled." in str(warn.message) for warn in w)
      )

  def test_eventarc_tool_config_experimental_warning(self):
    _WARNED_FEATURES.clear()
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      EventarcToolConfig()
      self.assertTrue(
          any(
              "EVENTARC_TOOL_CONFIG is enabled." in str(warn.message)
              for warn in w
          )
      )


if __name__ == "__main__":
  unittest.main()
