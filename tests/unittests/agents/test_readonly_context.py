from types import MappingProxyType
from unittest.mock import MagicMock

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService
import pytest


class DummyCredentialService(BaseCredentialService):
  async def load_credential(self,auth_config, tool_context):
    pass

  async def save_credential(self,auth_config, tool_context):
    pass


@pytest.fixture
def mock_invocation_context():
  mock_context = MagicMock()
  mock_context.invocation_id = "test-invocation-id"
  mock_context.agent.name = "test-agent-name"
  mock_context.session.state = {"key1": "value1", "key2": "value2"}
  mock_context.credential_service = DummyCredentialService()

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


def test_credential_service(mock_invocation_context):
  readonly_context = ReadonlyContext(mock_invocation_context)
  assert readonly_context.credential_service is not None
  assert isinstance(readonly_context.credential_service, BaseCredentialService)
