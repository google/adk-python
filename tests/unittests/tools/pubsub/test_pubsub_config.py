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

from google.adk.tools.pubsub.config import PubSubToolConfig


def test_pubsub_tool_config_init():
  """Test PubSubToolConfig initialization."""
  config = PubSubToolConfig(project_id="my-project")
  assert config.project_id == "my-project"


def test_pubsub_tool_config_default():
  """Test PubSubToolConfig default initialization."""
  config = PubSubToolConfig()
  assert config.project_id is None
