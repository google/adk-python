from collections import ChainMap
import unittest
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.sessions.session import Session


class TestReadonlyContextState(unittest.TestCase):

  def test_state_merging_precedence(self):
    # Setup
    mock_session = MagicMock(spec=Session)
    mock_session.state = {
        "persistent_key": "persistent_value",
        "conflict_key": "persistent_value",
    }

    mock_invocation_context = MagicMock(spec=InvocationContext)
    mock_invocation_context.session = mock_session
    mock_invocation_context.request_state = {
        "ephemeral_key": "ephemeral_value",
        "conflict_key": "ephemeral_value",
    }

    readonly_context = ReadonlyContext(mock_invocation_context)

    # Verify
    state = readonly_context.state

    # Check that ephemeral keys are present
    self.assertEqual(state["ephemeral_key"], "ephemeral_value")

    # Check that persistent keys are present
    self.assertEqual(state["persistent_key"], "persistent_value")

    # Check that ephemeral keys override persistent keys
    self.assertEqual(state["conflict_key"], "ephemeral_value")

    # Verify it behaves like a mapping
    self.assertIn("ephemeral_key", state)
    self.assertIn("persistent_key", state)
    self.assertEqual(state.get("ephemeral_key"), "ephemeral_value")

  def test_state_merging_empty_request_state(self):
    # Setup
    mock_session = MagicMock(spec=Session)
    mock_session.state = {"persistent_key": "persistent_value"}

    mock_invocation_context = MagicMock(spec=InvocationContext)
    mock_invocation_context.session = mock_session
    mock_invocation_context.request_state = {}

    readonly_context = ReadonlyContext(mock_invocation_context)

    # Verify
    state = readonly_context.state
    self.assertEqual(state["persistent_key"], "persistent_value")
    self.assertNotIn("ephemeral_key", state)

  def test_state_immutability(self):
    # Setup
    mock_session = MagicMock(spec=Session)
    mock_session.state = {"key": "value"}

    mock_invocation_context = MagicMock(spec=InvocationContext)
    mock_invocation_context.session = mock_session
    mock_invocation_context.request_state = {}

    readonly_context = ReadonlyContext(mock_invocation_context)
    state = readonly_context.state

    # Verify it raises TypeError on assignment
    with self.assertRaises(TypeError):
      state["key"] = "new_value"

    with self.assertRaises(TypeError):
      state["new_key"] = "value"


if __name__ == "__main__":
  unittest.main()
