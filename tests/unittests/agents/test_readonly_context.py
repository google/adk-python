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

from types import MappingProxyType
from unittest.mock import MagicMock

from google.adk.agents.readonly_context import ReadonlyContext
import pytest


@pytest.fixture
def mock_invocation_context():
  mock_context = MagicMock()
  mock_context.invocation_id = "test-invocation-id"
  mock_context.agent.name = "test-agent-name"
  mock_context.session.state = {"key1": "value1", "key2": "value2"}
  mock_context.user_id = "test-user-id"
  return mock_context


def test_invocation_id(mock_invocation_context):
  readonly_context = ReadonlyContext(mock_invocation_context)
  assert readonly_context.invocation_id == "test-invocation-id"


def test_agent_name(mock_invocation_context):
  readonly_context = ReadonlyContext(mock_invocation_context)
  assert readonly_context.agent_name == "test-agent-name"


def test_state_content(mock_invocation_context):
  readonly_context = ReadonlyContext(mock_invocation_context)
  state = readonly_context.state

  assert isinstance(state, MappingProxyType)
  assert state["key1"] == "value1"
  assert state["key2"] == "value2"


def test_user_id(mock_invocation_context):
  readonly_context = ReadonlyContext(mock_invocation_context)
  assert readonly_context.user_id == "test-user-id"
