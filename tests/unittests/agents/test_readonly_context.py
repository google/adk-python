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
